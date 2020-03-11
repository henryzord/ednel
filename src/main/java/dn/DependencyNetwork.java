package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import weka.core.Instances;

import java.io.*;
import java.nio.file.Files;
import java.util.*;

public class DependencyNetwork {
    private HashMap<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private ArrayList<String> variable_names;
    private JSONObject classifiersResources;
    private float learningRate;
    private int n_generations;
    private int burn_in;
    private int thinning_factor;

    private List<Integer> sampling_order = null;

    private ArrayList<HashMap<String, AbstractVariable>> pastVariables;

    private HashMap<String, String> optionTable;
    private HashMap<String, String> lastStart;


    public DependencyNetwork(
            MersenneTwister mt, String variables_path, String options_path,
            int burn_in, int thinning_factor, float learningRate, int n_generations
    ) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.mt = mt;
        this.variables = new HashMap<>();
        this.variable_names = new ArrayList<>((int)Math.pow(algorithms.length, 2));
        this.learningRate = learningRate;
        this.n_generations = n_generations;

        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;

        this.lastStart = null;

        JSONParser jsonParser = new JSONParser();
        classifiersResources = (JSONObject)jsonParser.parse(new FileReader(options_path));

        String row;

        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables_names  = Files.list(new File(algorithms[i].toString()).toPath()).toArray();
            for(int j = 0; j < variables_names.length; j++) {
                BufferedReader csvReader = new BufferedReader(new FileReader(variables_names[j].toString()));

                row = csvReader.readLine();

                String[] header = row.split(",(?![^(]*\\))");

                HashMap<String, Boolean> isContinuous = new HashMap<>();

                String variableName = header[header.length - 2];
                isContinuous.put(variableName, false);
                this.variable_names.add(variableName);
                int n_variables_table = 1;
                String[] parentNames = new String [0];

                HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(header.length);
                table.put(variableName, new HashMap<String, ArrayList<Integer>>());

                if(header.length > 2) {
                    parentNames = new String [header.length - 2];
                    n_variables_table = n_variables_table + parentNames.length;
                    for(int k = 0; k < header.length - 2; k++) {
                        parentNames[k] = header[k];
                        isContinuous.put(parentNames[k], false);
                        table.put(parentNames[k], new HashMap<String, ArrayList<Integer>>((int)Math.pow(2, n_variables_table)));
                    }
                }

                ArrayList<Float> probabilities = new ArrayList<>((int)Math.pow(2, n_variables_table));
                ArrayList<String> values = new ArrayList<>((int)Math.pow(2, n_variables_table));

                int index = 0;

                while ((row = csvReader.readLine()) != null) {
                    String[] this_data = row.split(",(?![^(]*\\))");

                    probabilities.add(Float.valueOf(this_data[this_data.length - 1]));
                    values.add(this_data[this_data.length - 2]);

                    for(int k = 0; k < this_data.length - 1; k++) {
                        isContinuous.replace(
                                header[k],
                                isContinuous.get(header[k]) ||
                                        (this_data[k].contains("loc") && this_data[k].contains("scale"))
                        );
                        if(this_data[k].toLowerCase().equals("null")) {
                            this_data[k] = null;
                        }

                        if(!table.get(header[k]).containsKey(this_data[k])) {  // if this variable does not have this value
                            table.get(header[k]).put(this_data[k], new ArrayList<>());
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();

                for(int k = 0; k < values.size(); k++) {
                    if(values.get(k).toLowerCase().equals("null")) {
                        values.set(k, null);
                    }
                }
                
                if(isContinuous.get(variableName)) {
                    variables.put(variableName, new ContinuousVariable(variableName, parentNames, table, values, probabilities, this.mt, this.learningRate, this.n_generations));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parentNames, table, values, probabilities, this.mt, this.learningRate, this.n_generations));
                }
            }
        }

        Integer[] indices = new Integer [variables.size()];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        sampling_order = Arrays.asList(indices);
    }

    private void sampleIndividual() throws Exception {
        optionTable = new HashMap<>(this.variable_names.size());

        Collections.shuffle(this.sampling_order, new Random(mt.nextInt()));

        for(int i = 0; i < this.sampling_order.size(); i++) {
            String variableName = this.variable_names.get(this.sampling_order.get(i));

            String algorithmName = variableName.split("_")[0];

            if(optionTable.getOrDefault(algorithmName, null) == null) {
                optionTable.put(algorithmName, "");
            }

            lastStart.put(
                    variableName,
                    this.variables.get(variableName).conditionalSampling(lastStart)
            );

            if(lastStart.get(variableName) != null) {
                JSONObject optionObj = (JSONObject)classifiersResources.getOrDefault(variableName, null);
                if(optionObj == null) {
                    optionObj = (JSONObject)classifiersResources.getOrDefault(lastStart.get(variableName), null);
                }

                // checks whether this is an option
                if(optionObj != null) {
                    boolean presenceMeans = (boolean)optionObj.get("presenceMeans");
                    String optionName = (String)optionObj.get("optionName");
                    String dtype = (String)optionObj.get("dtype");

                    if(dtype.equals("np.bool")) {
                        if(lastStart.get(variableName).toLowerCase().equals("false")) {
                            if(!presenceMeans) {
                                optionTable.put(algorithmName, (optionTable.get(algorithmName) + " " + optionName).trim());
                            }
                        } else {
                            if(presenceMeans) {
                                optionTable.put(algorithmName, (optionTable.get(algorithmName) + " " + optionName).trim());
                            }
                        }
                    } else {
                        optionTable.put(algorithmName, (optionTable.get(algorithmName) + " " + optionName + " " + lastStart.get(variableName)).trim());
                    }
                }
            }
        }
    }

    public Individual[] gibbsSample(HashMap<String, String> lastStart, int sampleSize, Instances train_data) throws Exception {
        Individual[] individuals = new Individual [sampleSize];

        this.lastStart = lastStart;

        // TODO use laplace correction; check edna paper for this

        int outerCounter = 0;
        int individualCounter = 0;

        // burns some individuals
        for(int i = 0; i < burn_in; i++) {
            // updates currentLastStart and currentOptionTable
            this.sampleIndividual();
        }

        while(individualCounter < sampleSize) {
            this.sampleIndividual();
            outerCounter += 1;

            String[] options = new String [optionTable.size() * 2];
            Object[] algNames = optionTable.keySet().toArray();
            int counter = 0;
            for(int j = 0; j < algNames.length; j++) {
                options[counter] = "-" + algNames[j];
                options[counter + 1] = optionTable.get(algNames[j]);
                counter += 2;
            }
            try {
                Individual individual = new Individual(options, this.lastStart, train_data);
                if(outerCounter >= this.thinning_factor) {
                    outerCounter = 0;
                    individuals[individualCounter] = individual;
                    individualCounter += 1;
                }
            } catch(Exception e) {

            }
        }
        this.lastStart = null;
        return individuals;
    }

    void generatePermutations(List<List<Character>> lists, List<String> result, int depth, String current) {
        if (depth == lists.size()) {
            result.add(current);
            return;
        }

        for (int i = 0; i < lists.get(depth).size(); i++) {
            generatePermutations(lists, result, depth + 1, current + lists.get(depth).get(i));
        }
    }

    /**
     * Generates all possible combinations of values between a surrogate variable and its parents.
     *
     * @param variableName
     * @return
     */
    private ArrayList<HashMap<String, String>> generateCombinations(String variableName) {
        String[] parents = this.variables.get(variableName).getParents();

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>(parents.length + 1);
        Object[] thisUniqueValues = this.variables.get(variableName).getUniqueValues().toArray();
        for(Object value : thisUniqueValues) {
            HashMap<String, String> data = new HashMap<>(parents.length + 1);
            data.put(variableName, (String)value);
            combinations.add(data);
        }

        for(String parent : parents) {
            Object[] values = this.variables.get(parent).getUniqueValues().toArray();
            int outer = values.length * combinations.size();

            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>(outer);

            for(int j = 0; j < values.length; j++) {
                for(int i = 0; i < combinations.size(); i++) {
                    HashMap<String, String> local = (HashMap<String, String>) combinations.get(i).clone();
                    local.put(parent, (String)values[j]);
                    new_combinations.add(local);
                }
            }
            combinations = new_combinations;
        }
        return combinations;
    }

    public void update(Individual[] population, Integer[] sortedIndices, float selectionShare) throws Exception {
        this.updateStructure(population, sortedIndices, selectionShare);  // TODO implement
        this.updateProbabilities(population, sortedIndices, selectionShare);

    }

    public void updateProbabilities(Individual[] population, Integer[] sortedIndices, float selectionShare) throws Exception {
        for(String variableName : this.variable_names) {
            this.variables.get(variableName).updateProbabilities(population, sortedIndices, selectionShare);
        }
    }

    public void updateStructure(Individual[] population, Integer[] sortedIndices, float selectionShare) {
        // TODO implement!
    }
}
