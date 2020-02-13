package dn;

import com.google.common.collect.Collections2;
import com.sun.org.apache.xpath.internal.operations.Variable;
import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.BaselineIndividual;
import eda.Individual;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.omg.CORBA.INTERNAL;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.util.*;

public class DependencyNetwork {
    private HashMap<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private ArrayList<String> variable_names;
    private ArrayList<String> sampling_order;
    private JSONObject classifiersResources;

    public DependencyNetwork(MersenneTwister mt, String variables_path, String options_path, String sampling_order_path) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.mt = mt;
        this.variables = new HashMap<>();
        this.variable_names = new ArrayList<>((int)Math.pow(algorithms.length, 2));

        JSONParser jsonParser = new JSONParser();
        classifiersResources = (JSONObject)jsonParser.parse(new FileReader(options_path));

        this.sampling_order = new ArrayList<>();

        // reads sampling order
        String row;
        BufferedReader csvReader = new BufferedReader(new FileReader(sampling_order_path));
        while ((row = csvReader.readLine()) != null) {
            sampling_order.add(row);
        }

        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables_names  = Files.list(new File(algorithms[i].toString()).toPath()).toArray();
            for(int j = 0; j < variables_names.length; j++) {
                csvReader = new BufferedReader(new FileReader(variables_names[j].toString()));

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

                        if(!table.get(header[k]).containsKey(this_data[k])) {  // if this variable does not have this value
                            table.get(header[k]).put(this_data[k], new ArrayList<>());
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();

                if(isContinuous.get(variableName)) {
                    variables.put(variableName, new ContinuousVariable(variableName, parentNames, table, values, probabilities, this.mt));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parentNames, table, values, probabilities, this.mt));
                }
            }
        }
    }

    public Individual[] gibbsSample(HashMap<String, String> lastStart, int thinning_factor, int sampleSize, Instances train_data) throws Exception {
        Individual[] individuals = new Individual [sampleSize];

        // TODO use laplace correction; check edna paper for this

        for(int i = 0; i < sampleSize * thinning_factor; i++) {
            HashMap<String, String> optionTable = new HashMap<>(this.sampling_order.size());

            for(String variableName : this.sampling_order) {
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
            if((i % thinning_factor) == 0) {
                String[] options = new String [optionTable.size() * 2];
                Object[] algNames = optionTable.keySet().toArray();
                int counter = 0;
                for(int j = 0; j < algNames.length; j++) {
                    options[counter] = "-" + algNames[j];
                    options[counter + 1] = optionTable.get(algNames[j]);
                    counter += 2;
                }

                individuals[i / sampleSize]  = new Individual(options, lastStart, train_data);
            }
        }
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

    public void updateProbabilities(Individual[] population, Integer[] sortedIndices, int to_select) throws Exception {

        for(String variableName : this.sampling_order) {
            this.variables.get(variableName).updateProbabilities(population, sortedIndices, to_select);


            // TODO re-normalize probabilities
            int z = 0;

//            ArrayList<HashMap<String, String>> combinations = generateCombinations(variableName);
//
//            this.variables.get(variableName)
        }
    }

    public void updateStructure(Individual[] population, Integer[] sortedIndices, int to_select) {

    }

    public static void main(String[] args) throws Exception {
        Instances train_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff")));
        train_data.setClassIndex(train_data.numAttributes() - 1);

        String variables_path = "/home/henry/Projects/ednel/resources/distributions";
        String options_path = "/home/henry/Projects/ednel/resources/options.json";
        String sampling_order_path = "/home/henry/Projects/ednel/resources/sampling_order.csv";

        MersenneTwister mt = new MersenneTwister();

        DependencyNetwork dn = new DependencyNetwork(mt, variables_path, options_path, sampling_order_path);
        BaselineIndividual bi = new BaselineIndividual(train_data);
        HashMap<String, String> startPoint = bi.getCharacteristics();
        dn.gibbsSample(startPoint,10, 50, train_data);
    }
}
