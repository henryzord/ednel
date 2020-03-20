package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.individual.Individual;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import weka.core.Instances;
import org.apache.commons.math3.special.Gamma;

import java.io.*;
import java.nio.file.Files;
import java.util.*;

public class DependencyNetwork {
    private HashMap<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private ArrayList<String> variable_names;
    private JSONObject classifiersResources;
    private int burn_in;
    private int thinning_factor;

    /**
     * The number of nearest neighbors to consider when
     * calculating the mutual information between discrete and
     * continuous variables.
     */
    private int nearestNeighbor = 3;

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
                    variables.put(variableName, new ContinuousVariable(variableName, parentNames, table, values, probabilities, this.mt, learningRate, n_generations));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parentNames, table, values, probabilities, this.mt, learningRate, n_generations));
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
        int to_select = Math.round(selectionShare * sortedIndices.length);

        Individual[] fittest = new Individual[to_select];
        for(int i = 0; i < to_select; i++) {
            fittest[i] = population[sortedIndices[i]];
        }

        this.updateStructure(fittest);  // TODO implement
        this.updateProbabilities(fittest);

    }

    public void updateProbabilities(Individual[] fittest) throws Exception {
        for(String variableName : this.variable_names) {
            this.variables.get(variableName).updateProbabilities(fittest);
        }
    }

    /**
     * A modified Mutual Information metric derived from
     * J.A. Gámez, J.L. Mateo, J.M. Puerta. EDNA: Estimation of Dependenc Networks Algorithm.
     */
    private double heuristic(
        AbstractVariable pivot, HashSet<String> parentSet, AbstractVariable candidate, Individual[] fittest) throws Exception {
        double mutualInformation = this.mutualInformation(pivot, candidate, fittest);
        double localMIs = 0;
        for(String parent : parentSet) {
            localMIs += this.mutualInformation(this.variables.get(parent), candidate, fittest);
        }
        return mutualInformation - (localMIs / (parentSet.size() + 1));
    }

    /**
     * Implements a method for calculating Mutual Information between Discrete and Continuous variables
     * proposed in
     *
     * Ross, Brian C. "Mutual information between discrete and continuous data sets." PloS one 9.2 (2014).
     *
     * The specific method is Equation 5 of the paper.
     *
     * @param a Discrete variable a
     * @param b Continuous variable b
     * @param fittest data points to use for calculating mutual information
     * @return The mutual information between these two variables, given data points.
     */
    private double discreteContinuousMutualInformation(DiscreteVariable a, ContinuousVariable b, Individual[] fittest) {
        String a_name = a.getName();
        String b_name = b.getName();

        HashSet<String> a_values = a.getUniqueValues();
        HashMap<String, Integer> a_counts = new HashMap<>(a.getUniqueValues().size());
        for(String value : a_values) {
            a_counts.put(String.valueOf(value), 0);
        }
        for(Individual fit : fittest) {
            HashMap<String, String> localCharacteristics = fit.getCharacteristics();
            String a_value = String.valueOf(localCharacteristics.get(a_name));

            a_counts.put(
                a_value,
                a_counts.get(a_value) + 1
            );
        }

        int N = fittest.length;
        int k = this.nearestNeighbor;
        double mutualInformation = 0;

        for(Individual fit : fittest) {
            Float b_value = Float.valueOf(fit.getCharacteristics().get(b_name));
            int z = 0;
            z += 1;
        }


//        Gamma.digamma()

        return mutualInformation;
    }

    private double doubleDiscreteMutualInformation(AbstractVariable a, AbstractVariable b, Individual[] fittest) {
        HashMap<String, Integer> a_counts = new HashMap<>(a.getUniqueValues().size());
        HashMap<String, Integer> b_counts = new HashMap<>(b.getUniqueValues().size());
        HashMap<String, Integer> joint_counts = new HashMap<>(
                a.getUniqueValues().size() * b.getUniqueValues().size()
        );

        int total_cases = fittest.length;
        String a_name = a.getName();
        String b_name = b.getName();

        for(String value : a.getUniqueValues()) {
            a_counts.put(String.valueOf(value), 0);
        }
        for(String value : b.getUniqueValues()) {
            b_counts.put(String.valueOf(value), 0);
        }
        for(String a_value : a.getUniqueValues()) {
            for(String b_value : b.getUniqueValues()) {
                joint_counts.put(String.format("%s,%s", a_value, b_value), 0);
            }
        }

        for(Individual fit : fittest) {
            HashMap<String, String> localCharacteristics = fit.getCharacteristics();

            String a_value = String.valueOf(localCharacteristics.get(a_name));
            String b_value = String.valueOf(localCharacteristics.get(b_name));

            a_counts.put(
                a_value,
                a_counts.get(a_value) + 1
            );
            b_counts.put(
                b_value,
                b_counts.get(b_value) + 1
            );

            String joint_name = String.format("%s,%s", a_value, b_value);

            joint_counts.put(
                joint_name,
                joint_counts.get(joint_name) + 1
            );
        }

        double mutual_information = 0;
        for(String joint_name : joint_counts.keySet()) {
            String a_value = joint_name.split(",")[0];
            String b_value = joint_name.split(",")[1];

            double joint_prob = (double)joint_counts.get(joint_name) / (double)total_cases;
            double a_prob = (double)a_counts.get(a_value) / (double)total_cases;
            double b_prob = (double)b_counts.get(b_value) / (double)total_cases;

            double local = joint_prob * Math.log(joint_prob / (a_prob * b_prob));
            mutual_information += Double.isNaN(local)? 0 : local;
        }
        return mutual_information;
    }

    private double mutualInformation(AbstractVariable a, AbstractVariable b, Individual[] fittest) throws Exception {
        if(a instanceof DiscreteVariable) {
            if(b instanceof DiscreteVariable) {
                return this.doubleDiscreteMutualInformation(a, b, fittest);
            } else {  // b is instance of ContinuousVariable
                return this.discreteContinuousMutualInformation((DiscreteVariable)a, (ContinuousVariable)b, fittest);
            }
        } else {  // a is instance of ContinuousVariable
            if(b instanceof DiscreteVariable) {
                return this.discreteContinuousMutualInformation((DiscreteVariable)b, (ContinuousVariable)a, fittest);
            } else { // a and b are instances of ContinuousVariable
                throw new Exception("unsupported case.");
            }
        }
    }

    public void updateStructure(Individual[] fittest) throws Exception {
        // TODO implement!
        // candidates to be parents of a variable
        HashSet<String> staticCandSet = new HashSet<>(variable_names.size());
        staticCandSet.addAll(this.variable_names);

        for(int index : this.sampling_order) {
            String this_variable = this.variable_names.get(index);
            HashSet<String> localCandSet = (HashSet<String>)staticCandSet.clone();
            localCandSet.remove(this_variable);
            HashSet<String> parentSet = new HashSet<>();

            while(staticCandSet.size() > 0) {
                double bestHeuristic = -1;
                String bestCandidate = null;
                for(String candidate : staticCandSet) {
                    // TODO calculate mutual information, heuristic
                    double heuristic = this.heuristic(
                        this.variables.get(this_variable),
                        parentSet,
                        this.variables.get(candidate),
                        fittest
                    );
                    if(heuristic > 0) {
                        if(heuristic > bestHeuristic) {
                            bestHeuristic = heuristic;
                            bestCandidate = candidate;
                        }
                    } else {
                        localCandSet.remove(candidate);
                    }
                }
                if(bestHeuristic > 0) {
                    parentSet.add(bestCandidate);
                    localCandSet.remove(bestCandidate);
                }
            }
            // TODO now add candidates as parents of this variable!
            int z = 0;
            z += 1;
        }
    }

    public HashMap<String, AbstractVariable> getVariables() {
        return this.variables;
    }
}
