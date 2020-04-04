package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import utils.Argsorter;
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
    private int n_variables;

    /**
     * The number of nearest neighbors to consider when
     * calculating the mutual information between discrete and
     * continuous variables.
     */
    private int nearest_neighbor = 3;

    private List<Integer> sampling_order = null;

    private ArrayList<HashMap<String, AbstractVariable>> pastVariables;

    private HashMap<String, String> optionTable;
    private HashMap<String, String> lastStart;


    public DependencyNetwork(
            MersenneTwister mt, String variables_path, String options_path,
            int burn_in, int thinning_factor, float learningRate, int n_generations, int nearest_neighbor
    ) throws Exception {
        this.mt = mt;
        this.variables = new HashMap<>();
        this.variable_names = new ArrayList<>();

        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;
        this.nearest_neighbor = nearest_neighbor;

        this.lastStart = null;

        this.readVariablesFromFiles(variables_path, options_path, learningRate, n_generations);

        this.createInitialSamplingOrder(variables.size());
        this.n_variables = this.sampling_order.size();
    }

    private void readVariablesFromFiles(String variables_path, String options_path, float learningRate, int n_generations) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

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
                ArrayList<String> parents_names = new ArrayList<>();

                HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(header.length);
                table.put(variableName, new HashMap<>());

                if(header.length > 2) {
                    n_variables_table = n_variables_table + header.length - 2;
                    for(int k = 0; k < header.length - 2; k++) {
                        parents_names.add(header[k]);
                        isContinuous.put(header[k], false);
                        table.put(header[k], new HashMap<>());
                    }
                }

                ArrayList<Double> probabilities = new ArrayList<>((int)Math.pow(2, n_variables_table));
                ArrayList<String> values = new ArrayList<>((int)Math.pow(2, n_variables_table));

                int index = 0;

                while ((row = csvReader.readLine()) != null) {
                    String[] this_data = row.split(",(?![^(]*\\))");

                    probabilities.add(Double.valueOf(this_data[this_data.length - 1]));
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
                csvReader.close();  // finishes reading this file

                for(int k = 0; k < values.size(); k++) {
                    if(values.get(k).toLowerCase().equals("null")) {
                        values.set(k, null);
                    }
                }

                if(isContinuous.get(variableName)) {
                    variables.put(variableName, new ContinuousVariable(variableName, parents_names, isContinuous, table, values, probabilities, this.mt, learningRate, n_generations));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parents_names, isContinuous, table, values, probabilities, this.mt, learningRate, n_generations));
                }
            }
        }
    }

    /**
     * Creates a simple initial sampling order, with range [0, size).
     * @param size The number of variables in the graphical model.
     * @return
     */
    private void createInitialSamplingOrder(int size) {
        Integer[] indices = new Integer [size];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        sampling_order = Arrays.asList(indices);
    }

    private void sampleIndividual() throws Exception {
        optionTable = new HashMap<>(this.variable_names.size());

        // shuffles sampling order
        Collections.shuffle(this.sampling_order, new Random(mt.nextInt()));

        for(int idx : this.sampling_order) {
            String variableName = this.variable_names.get(idx);
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

    /**
     * Generates all possible combinations of values between a surrogate variable and its parents.
     *
     * @param variableName
     * @return
     */
    private ArrayList<HashMap<String, String>> generateCombinations(String variableName) {
        ArrayList<String> parents = this.variables.get(variableName).getParentsNames();

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>(parents.size() + 1);
        Object[] thisUniqueValues = this.variables.get(variableName).getUniqueValues().toArray();
        for(Object value : thisUniqueValues) {
            HashMap<String, String> data = new HashMap<>(parents.size() + 1);
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

        this.updateStructure(fittest);
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
     *
     * @param child Current child variable
     * @param parentSet Parents of this variable
     * @param candidate Candidate parent for this variable
     * @param fittest Fittest individuals from current generation
     * @return A modified mutual information metric, which is greater than zero if the candidate parent
     *         is significantly correlated to this variable, or negative otherwise
     * @throws Exception If any exception occurs
     */
    private double heuristic(
        AbstractVariable child, HashSet<String> parentSet, AbstractVariable candidate, Individual[] fittest
    ) throws Exception {
        double localMIs = 0;
        for(String parent : parentSet) {
            localMIs += this.mutualInformation(candidate, this.variables.get(parent), fittest);
        }
        return this.mutualInformation(child, candidate, fittest) - (localMIs / (parentSet.size() + 1));
    }

    /**
     * Implements a method for calculating Mutual Information between Discrete and Continuous variables
     * proposed in
     *
     * Ross, Brian C. "Mutual information between discrete and continuous data sets." PloS one 9.2 (2014).
     *
     * The specific method is Equation 5 of the paper.
     *
     * Coincidentally, is the same method used by scikit-learn:
     * https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
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
        Double[] continuousValues = new Double [fittest.length];
        String[] discreteValues = new String [fittest.length];

        int N = 0;

        // captures information
        for(int i = 0; i < fittest.length; i++) {
            HashMap<String, String> localCharacteristics = fittest[i].getCharacteristics();
            String a_value = String.valueOf(localCharacteristics.get(a_name));
            String b_value = String.valueOf(localCharacteristics.get(b_name));

            if(!b_value.toLowerCase().equals("null")) {
                discreteValues[i] = a_value;
                continuousValues[i] = Double.valueOf(b_value);

                N += 1;

                a_counts.put(
                    a_value,
                    a_counts.get(a_value) + 1
                );
            } else {
                continuousValues[i] = Double.NaN;
            }
        }

        // mutual information
        double mi = 0;

        for(int i = 0; i < continuousValues.length; i++) {
            Double thisValue = continuousValues[i];
            String thisClass = discreteValues[i];

            if(!Double.isNaN(thisValue)) {
                Double[] dists = new Double [continuousValues.length];
                for(int j = 0; j < continuousValues.length; j++) {
                    if(i == j || Double.isNaN(continuousValues[j])) {
                        dists[j] = Double.NaN;
                    } else {
                        dists[j] = Math.abs(thisValue - continuousValues[j]);
                    }
                }
                // NaN values are placed at the end of the list
                Integer[] sortedIndices = Argsorter.crescent_argsort(dists);
                int m = 0;
                int counter = 0;
                int neighbor_counter = 0;
                while(!Double.isNaN(continuousValues[sortedIndices[counter]]) && (neighbor_counter < this.nearest_neighbor)) {
                    if(discreteValues[sortedIndices[counter]].equals(thisClass)) {
                        neighbor_counter += 1;
                    }
                    m += 1;
                    counter += 1;
                }
                if((m > 0) && a_counts.get(thisClass) > 0) {
                    mi -= (
                        Gamma.digamma(a_counts.get(thisClass)) + Gamma.digamma(m)
                    ) * (1.0/(double)N);
                }
            }
        }
        return Math.max(0, Gamma.digamma(N) + mi + Gamma.digamma(this.nearest_neighbor));
    }

    /**
     * Implements a method for calculating Mutual Information between two Continuous Variables. The variables need
     * not to be Gaussians. The method was proposed in<br>
     *
     * Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger.
     * "Estimating mutual information." Physical review E 69.6 (2004): 066138.<br>
     *
     * The specific method is Equation 8 of the paper.<br>
     *
     * Coincidentally, is the same method used by scikit-learn:
     * https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
     *
     * @param a Continuous variable a
     * @param b Continuous variable b
     * @param fittest data points to use for calculating mutual information
     * @return The mutual information between these two variables, given data points.
     */
    private double continuousContinuousMutualInformation(ContinuousVariable a, ContinuousVariable b, Individual[] fittest) throws Exception {
        String a_name = a.getName();
        String b_name = b.getName();

        Double[] a_values = new Double [fittest.length];
        Double[] b_values = new Double [fittest.length];

        Double mean_a = 0.;
        Double mean_b = 0.;
        int a_values_size = 0;
        int b_values_size = 0;

        for(int i = 0; i < fittest.length; i++) {
            HashMap<String, String> localCharacteristics = fittest[i].getCharacteristics();
            String a_value = String.valueOf(localCharacteristics.get(a_name));
            String b_value = String.valueOf(localCharacteristics.get(b_name));

            if(a_value.toLowerCase().equals("null")) {
                a_values[i] = Double.NaN;
            } else {
                a_values[i] = Double.parseDouble(a_value);
                mean_a += a_values[i];
                a_values_size += 1;
            }
            if(b_value.toLowerCase().equals("null")) {
                b_values[i] = Double.NaN;
            } else {
                b_values[i] = Double.parseDouble(b_value);
                mean_b += b_values[i];
                b_values_size += 1;
            }
        }
        mean_a /= a_values_size;
        mean_b /= b_values_size;

        int non_nan_values = 0;
        // replaces missing values with mean
        for(int i = 0; i < a_values.length; i++) {
            if(Double.isNaN(a_values[i])) {
                if(!Double.isNaN((b_values[i]))) {
                    a_values[i] = mean_a;
                    non_nan_values += 1;
                }
            } else if(Double.isNaN((b_values[i]))) {
                if(!Double.isNaN(a_values[i])) {
                    b_values[i] = mean_b;
                    non_nan_values += 1;
                }
            } else {
                non_nan_values += 1;
            }
        }

        // mutual information
        double mi = 0;

        // this computation is not originally in the paper. It forces the algorithm
        // to choose a new value of k, if the previous value exceeds the number of
        // available values
        int k = Math.min(non_nan_values, this.nearest_neighbor + 1);

        // now computes euclidean distances
        Double[] distances = new Double [a_values.length];
        for(int i = 0; i < a_values.length; i++) {
            for(int j = 0; j < a_values.length; j++) {
                distances[j] = Math.sqrt(
                    Math.pow(a_values[i] - a_values[j], 2) + Math.pow(b_values[i] - b_values[j], 2)
                );
            }
            // NaN values are placed at the end of the sorting
            Integer[] sorted = Argsorter.crescent_argsort(distances);
            // does not take into account position 0, since it's the same point
            double max_a = a_values[sorted[k]];
            double max_b = b_values[sorted[k]];

            int a_count = 0;
            int b_count = 0;
            for(int j = 0; j < a_values.length; j++) {
                if(Double.isNaN(a_values[j]) || Double.isNaN(b_values[j])) {
                    break;
                }
                if(a_values[j] < max_a) {
                    a_count += 1;
                }
                if(b_values[j] < max_b) {
                    b_count += 1;
                }
            }
            mi -= (Gamma.digamma(a_count + 1) + Gamma.digamma(b_count + 1));
        }
        // TODO not sure on the value
        mi = (mi/non_nan_values) + Gamma.digamma(k) + Gamma.digamma(non_nan_values);
        return mi;
    }

    private double discreteDiscreteMutualInformation(AbstractVariable a, AbstractVariable b, Individual[] fittest) {
        HashMap<String, Integer> a_counts = new HashMap<>(a.getUniqueValues().size());
        HashMap<String, Integer> b_counts = new HashMap<>(b.getUniqueValues().size());
        HashMap<String, Integer> joint_counts = new HashMap<>(
                a.getUniqueValues().size() * b.getUniqueValues().size()
        );

        String a_name = a.getName();
        String b_name = b.getName();

        // uses additive correction
        for(String value : a.getUniqueValues()) {
            a_counts.put(String.valueOf(value), b.getUniqueValues().size());
        }
        for(String value : b.getUniqueValues()) {
            b_counts.put(String.valueOf(value), a.getUniqueValues().size());
        }
        for(String a_value : a.getUniqueValues()) {
            for(String b_value : b.getUniqueValues()) {
                joint_counts.put(String.format("%s,%s", a_value, b_value), 1);
            }
        }

        int total_cases = fittest.length + joint_counts.size();

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

        // mutual information
        double mi = 0;
        for(String joint_name : joint_counts.keySet()) {
            String a_value = joint_name.split(",")[0];
            String b_value = joint_name.split(",")[1];

            double joint_prob = (double)joint_counts.get(joint_name) / (double)total_cases;
            double a_prob = (double)a_counts.get(a_value) / (double)total_cases;
            double b_prob = (double)b_counts.get(b_value) / (double)total_cases;

            mi += joint_prob * Math.log(joint_prob / (a_prob * b_prob));
        }
        return mi;
    }

    private double mutualInformation(AbstractVariable a, AbstractVariable b, Individual[] fittest) throws Exception {
        if(a instanceof DiscreteVariable) {
            if(b instanceof DiscreteVariable) {
                return this.discreteDiscreteMutualInformation(a, b, fittest);
            } else {  // b is instance of ContinuousVariable
                // TODO maybe compute discrete mutual information
                // TODO between discrete variable and continuous, but
                // TODO treat continuous variable as (NULL, NOT NULL)
                return this.discreteContinuousMutualInformation((DiscreteVariable)a, (ContinuousVariable)b, fittest);
            }
        } else {  // a is instance of ContinuousVariable
            if(b instanceof DiscreteVariable) {
                // TODO maybe compute discrete mutual information
                // TODO between discrete variable and continuous, but
                // TODO treat continuous variable as (NULL, NOT NULL)
                return this.discreteContinuousMutualInformation((DiscreteVariable)b, (ContinuousVariable)a, fittest);
            } else { // a and b are instances of ContinuousVariable
                return this.continuousContinuousMutualInformation((ContinuousVariable)a, (ContinuousVariable)b, fittest);
            }
        }
    }

    public void updateStructure(Individual[] fittest) throws Exception {
        for(String varName : this.variable_names) {
            this.variables.get(varName).updateUniqueValues(fittest);
        }

        for(int index : this.sampling_order) {
//            // TODO remove this!
//            System.out.println("TODO remove this!!!");
//            System.out.println("come back to me later!");
//
//            this.variables.get("J48_confidenceFactorValue").updateStructure(
//                new AbstractVariable[]{this.variables.get("PART_confidenceFactorValue"), this.variables.get("PART")},
//                fittest
//            );
//            this.variables.get("J48_confidenceFactorValue").updateProbabilities(fittest);
//            this.variables.get("J48_confidenceFactorValue").updateProbabilities(fittest);
//            this.variables.get("J48_confidenceFactorValue").conditionalSampling(new HashMap<String, String>(){{put("PART_confidenceFactorValue", "0.25");}});

            String variableName = this.variable_names.get(index);

            System.out.println("on variable " + variableName);
            // TODO remove!

            // candidates to be parents of a variable
            HashSet<String> candSet = new HashSet<>(variable_names.size());
            candSet.addAll(this.variable_names);
            candSet.remove(variableName);
            HashSet<String> parentSet = new HashSet<>();

            while(candSet.size() > 0) {
                double bestHeuristic = -1;
                String bestCandidate = null;

                HashSet<String> toRemove = new HashSet<>();

                for(String candidate : candSet) {
                    double heuristic = this.heuristic(
                        this.variables.get(variableName),
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
                        toRemove.add(candidate);
                    }
                }
                candSet.removeAll(toRemove);

                if(bestHeuristic > 0) {
                    parentSet.add(bestCandidate);
                    candSet.remove(bestCandidate);
                }
            }
            AbstractVariable thisVariable = this.variables.get(variableName);
            AbstractVariable[] parents = new AbstractVariable [parentSet.size()];

            Object[] parentList = parentSet.toArray();

            // TODO reduce number of parents of a variable!!!

            for(int i = 0; i < parentSet.size(); i++) {
                parents[i] = this.variables.get((String)parentList[i]);
                System.out.println("\t" + parentList[i]); // TODO remove me!
            }

            thisVariable.updateStructure(parents, fittest);
        }
    }

    public HashMap<String, AbstractVariable> getVariables() {
        return this.variables;
    }
}
