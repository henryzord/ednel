package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
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

    /**
     * The number of nearest neighbors to consider when
     * calculating the mutual information between discrete and
     * continuous variables.
     */
    private int nearest_neighbor = 3;

    private ArrayList<ArrayList<Integer>> samplingOrder = null;

    private ArrayList<HashMap<String, AbstractVariable>> pastVariables;

    private HashMap<String, String> optionTable;
    private HashMap<String, String> lastStart;

    private int currentGenDiscardedIndividuals;
    private int currentGenEvals;
    private int currentGenConnections;

    public DependencyNetwork(
            MersenneTwister mt, String variables_path, String options_path, String sampling_order_path,
            int burn_in, int thinning_factor, float learningRate, int n_generations, int nearest_neighbor
    ) throws Exception {
        this.mt = mt;
        this.variables = new HashMap<>();
        this.variable_names = new ArrayList<>();

        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;
        this.nearest_neighbor = nearest_neighbor;

        this.currentGenEvals = 0;
        this.currentGenDiscardedIndividuals = 0;
        this.currentGenConnections = 0;

        this.lastStart = null;

        this.readVariablesFromFiles(variables_path, options_path, learningRate, n_generations);

        this.samplingOrder = this.readSamplingOrderPath(sampling_order_path, variable_names);

    }

    private static ArrayList<ArrayList<Integer>> readSamplingOrderPath(String sampling_order_path, ArrayList<String> variable_names) throws IOException, ParseException {
        JSONParser jsonParser = new JSONParser();
        JSONObject jobj = (JSONObject)jsonParser.parse(new FileReader(sampling_order_path));

        ArrayList<ArrayList<Integer>> clusters = new ArrayList<>(jobj.size());
        for(Object key : jobj.keySet()) {
            JSONArray jarr = (JSONArray)jobj.get(key);
            ArrayList<Integer> localCluster = new ArrayList<>();
            for(Object localKey : jarr.toArray()) {
                localCluster.add(variable_names.indexOf(localKey));
            }
            clusters.add(localCluster);
        }
        return clusters;
    }

    private void readVariablesFromFiles(String variables_path, String options_path, float learningRate, int n_generations) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.currentGenConnections = 0;

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
                // isContinuous.put(variableName, false);
                this.variable_names.add(variableName);
                int n_variables_table = 1;
                ArrayList<String> parents_names = new ArrayList<>();

                HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(header.length);
                table.put(variableName, new HashMap<>());

                // if there are more entries than simply the name of this variable plus column "probability"
                if(header.length > 2) {
                    n_variables_table = n_variables_table + header.length - 2;
                    for(int k = 0; k < header.length - 2; k++) {
                        parents_names.add(header[k]);
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
                        isContinuous.put(
                            header[k],
                            (this_data[k].contains("loc") && this_data[k].contains("scale")) ||  // univariate normal distribution
                                (this_data[k].contains("means"))  // multivariate normal distribution
                        );

                        // if this variable does not have this value
                        if(!table.get(header[k]).containsKey(this_data[k])) {
                            table.get(header[k]).put(this_data[k], new ArrayList<>());
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();  // finishes reading this file

                boolean amIContinuous = isContinuous.get(variableName);
                isContinuous.remove(variableName);

                if(amIContinuous) {
                    variables.put(variableName, new ContinuousVariable(variableName, parents_names, isContinuous, table, values, probabilities, this.mt, learningRate, n_generations));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parents_names, isContinuous, table, values, probabilities, this.mt, learningRate, n_generations));
                }
                this.currentGenConnections += this.variables.get(variableName).getParentsNames().size();
            }
        }
    }

    private void sampleIndividual() throws Exception {
        optionTable = new HashMap<>(this.variable_names.size());

        for(ArrayList<Integer> cluster : this.samplingOrder) {
            for(int idx : cluster) {
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
    }

    public Individual[] gibbsSample(HashMap<String, String> lastStart, int sampleSize, Instances train_data) throws Exception {
        Individual[] individuals = new Individual [sampleSize];

        this.lastStart = lastStart;
        this.currentGenDiscardedIndividuals = 0;
        this.currentGenEvals = 0;

        int outerCounter = 0;
        int individualCounter = 0;

        // shuffles sampling order
        Collections.shuffle(this.samplingOrder, new Random(mt.nextInt()));

        // burns some individuals
        for(int i = 0; i < burn_in; i++) {
            // updates currentLastStart and currentOptionTable
            this.sampleIndividual();
        }
        this.currentGenDiscardedIndividuals += burn_in;

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
                    this.currentGenEvals += 1;
                } else {
                    this.currentGenDiscardedIndividuals += 1;
                }
            } catch(Exception e) {  // invalid individual generated
                this.currentGenDiscardedIndividuals += 1;
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
     * It has the same behavior as the method used by scikit-learn:
     * https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
     *
     * @param a Values of discrete variable a
     * @param b Values of continuous variable b
     * @param k number of neighbors to consider
     * @return The mutual information between these two variables, given data points.
     */
    public static double discreteContinuousMutualInformation(String[] a, Double[] b, int k) {

        HashSet<String> aUnique = new HashSet<String>(Arrays.asList(a));
        HashMap<String, Integer> a_counts = new HashMap<>(aUnique.size());

        // captures information
        for(int i = 0; i < a.length; i++) {
            a_counts.put(
                a[i],
                a_counts.getOrDefault(a[i], 0) + 1
            );
        }

        // this implementation is very naive. A better implementation is described in
        // Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger.
        // "Estimating mutual information." Physical review E 69.6 (2004): 066138.
        // Where one sorts the continuous array by magnitude, and then looks into the vicinity of values
        // for nearest neighbors.
        // But since a.length will probably be <300 for this EDA, this code shall pass.
        int N = 0;
        double m_all = 0;
        double k_all = 0;
        double label_counts = 0;
        for(int i = 0; i < a.length; i++) {
            // ignores labels that have only one instance
            if(a_counts.get(a[i]) == 1) {
                continue;
            }
            N += 1;

            // computes distances
            Double[] dists = new Double [a.length];
            for(int j = 0; j < a.length; j++) {
                if(i == j) {
                    dists[j] = Double.MAX_VALUE;
                } else {
                    dists[j] = Math.abs(b[i] - b[j]);
                }
            }

            Integer[] sortedIndices = Argsorter.crescent_argsort(dists);
            double maxDist = Double.MAX_VALUE;
            int neighbor_counter = 0;
            for(int j = 0; j < a.length; j++) {
                if(a[sortedIndices[j]].equals(a[i])) {
                    neighbor_counter += 1;
                    if(neighbor_counter == k) {
                        maxDist = Math.nextAfter(dists[sortedIndices[j]], 0);
                        break;
                    }
                }
            }

            int m = 0;
            while((m < a.length) && (dists[sortedIndices[m]] <= maxDist)) {
                m += 1;
            }
            k_all += Gamma.digamma(neighbor_counter);
            m_all += Gamma.digamma(m + 1);
            label_counts += Gamma.digamma(a_counts.get(a[i]));
        }

        double before = Gamma.digamma(N) - (label_counts / N) + (k_all / N) - (m_all / N);
        return Math.max(0, before);
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
     * It has the same behavior as the method used by scikit-learn:
     * https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
     *
     * @param a Values of continuous variable a
     * @param b Values of continuous variable b
     * @param k number of neighbors to consider
     * @return The mutual information between these two variables, given data points.
     */
    public static double continuousContinuousMutualInformation(Double[] a, Double[] b, int k) {

        int N = a.length;
        double a_counts = 0;
        double b_counts = 0;
        for(int i = 0; i < a.length; i++) {
            Double[] dists = new Double [a.length];

            for(int j = 0; j < a.length; j++) {
                if(i == j) {
                    dists[j] = Double.MAX_VALUE;
                } else {
                    // Chebyshev distance
                    dists[j] = Math.max(
                        Math.abs(a[i] - a[j]),
                        Math.abs(b[i] - b[j])
                    );
                }
            }
            Integer[] sortedIndices = Argsorter.crescent_argsort(dists);
            double kth_distance = Math.nextAfter(dists[sortedIndices[k - 1]], 0);
            int nx = 0;
            int ny = 0;

            for(int j = 0; j < N; j++) {
                // does that for a
                if(Math.abs(a[i] - a[j]) <= kth_distance) {
                    nx += 1;
                }
                // now does that for b
                if(Math.abs(b[i] - b[j]) <= kth_distance) {
                    ny += 1;
                }
            }
            a_counts += Gamma.digamma(nx);
            b_counts += Gamma.digamma(ny);
        }

        double before = Gamma.digamma(N) + Gamma.digamma(k) - (a_counts/N) - (b_counts/N);
        return Math.max(0, before);
    }

    /**
     * This method uses natural logarithms, and Laplace correction
     * for computing mutual information between two discrete variables.
     *
     * @param a Array of discrete variable a values
     * @param b Array of discrete variable b values
     * @return Mutual information between two discrete variables.
     */
    private double discreteDiscreteMutualInformation(String[] a, String[] b) {
        HashSet<String> aUnique = new HashSet<String>(Arrays.asList(a));
        HashSet<String> bUnique = new HashSet<String>(Arrays.asList(b));

        HashMap<String, Integer> a_counts = new HashMap<>(aUnique.size());
        HashMap<String, Integer> b_counts = new HashMap<>(bUnique.size());

        HashMap<String, Integer> joint_counts = new HashMap<>(
                aUnique.size() * bUnique.size()
        );

        // uses additive correction
        for(String value : aUnique) {
            a_counts.put(String.valueOf(value), bUnique.size());
        }
        for(String value : bUnique) {
            b_counts.put(String.valueOf(value), aUnique.size());
        }
        for(String a_value : aUnique) {
            for(String b_value : bUnique) {
                joint_counts.put(String.format("%s,%s", a_value, b_value), 1);
            }
        }

        int total_cases = a.length + joint_counts.size();

        for(int i = 0; i < a.length; i++) {
            a_counts.put(
                a[i],
                a_counts.get(a[i]) + 1
            );
            b_counts.put(
                b[i],
                b_counts.get(b[i]) + 1
            );

            String joint_name = String.format("%s,%s", a[i], b[i]);

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


    /**
     * Computes mutual information between variables, regardless of the nature of each variable.
     *
     * If a variable is continuous, adds a little noise, as suggested in
     * Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger.
     * "Estimating mutual information." Physical review E 69.6 (2004): 066138.
     *
     * @param a First variable
     * @param b Second variable
     * @param fittest Array of fittest individuals from the current generation
     * @return Mutual information between variables
     * @throws Exception If any exception occurs
     */
    private double mutualInformation(AbstractVariable a, AbstractVariable b, Individual[] fittest) throws Exception {
        String a_name = a.getName();
        String b_name = b.getName();

        Double[] aContinuous = new Double [fittest.length];
        Double[] bContinuous =  new Double [fittest.length];

        String[] aDiscrete = new String [fittest.length];
        String[] bDiscrete = new String [fittest.length];

        for(int i = 0; i < fittest.length; i++) {
            HashMap<String, String> localCharacteristics = fittest[i].getCharacteristics();
            String a_value = String.valueOf(localCharacteristics.get(a_name));
            String b_value = String.valueOf(localCharacteristics.get(b_name));

            if(a instanceof DiscreteVariable) {
                aDiscrete[i] = a_value;
            } else {
                // adds noise, as suggested by Kraskov et al
                double noise = (double)mt.nextInt(10) / 10e-10;
                if(!a_value.toLowerCase().equals("null")) {
                    aContinuous[i] = Double.parseDouble(a_value) + noise;
                } else {
                    aContinuous[i] = (((ContinuousVariable)a).getMinValue() - 1)  + noise;
                }
            }
            if(b instanceof DiscreteVariable) {
                bDiscrete[i] = b_value;
            } else {
                // adds noise, as suggested by Kraskov et al
                double noise = (double)mt.nextInt(10) / 10e-10;
                if(!b_value.toLowerCase().equals("null")) {
                    bContinuous[i] = Double.parseDouble(b_value) + noise;
                } else {
                    bContinuous[i] = (((ContinuousVariable)b).getMinValue() - 1) + noise;
                }
            }
        }

        if(a instanceof DiscreteVariable) {
            if(b instanceof DiscreteVariable) {
                return this.discreteDiscreteMutualInformation(aDiscrete, bDiscrete);
            } else {  // b is instance of ContinuousVariable
                return discreteContinuousMutualInformation(aDiscrete, bContinuous, this.nearest_neighbor);
            }
        } else {  // a is instance of ContinuousVariable
            if(b instanceof DiscreteVariable) {
                return discreteContinuousMutualInformation(bDiscrete, aContinuous, this.nearest_neighbor);
            } else { // a and b are instances of ContinuousVariable
                return continuousContinuousMutualInformation(aContinuous, bContinuous, this.nearest_neighbor);
            }
        }
    }

    public void updateStructure(Individual[] fittest) throws Exception {
        this.currentGenConnections = 0;

        for(String varName : this.variable_names) {
            this.variables.get(varName).updateUniqueValues(fittest);
        }

        for(ArrayList<Integer> cluster : this.samplingOrder) {
            for (int index : cluster) {
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

                // candidates to be parents of a variable
                HashSet<String> candSet = new HashSet<>(variable_names.size());
                candSet.addAll(this.variable_names);
                candSet.remove(variableName);
                HashSet<String> parentSet = new HashSet<>();

                while (candSet.size() > 0) {
                    double bestHeuristic = -1;
                    String bestCandidate = null;

                    HashSet<String> toRemove = new HashSet<>();

                    for (String candidate : candSet) {
                        double heuristic = this.heuristic(
                                this.variables.get(variableName),
                                parentSet,
                                this.variables.get(candidate),
                                fittest
                        );
                        if (heuristic > 0) {
                            if (heuristic > bestHeuristic) {
                                bestHeuristic = heuristic;
                                bestCandidate = candidate;
                            }
                        } else {
                            toRemove.add(candidate);
                        }
                    }
                    candSet.removeAll(toRemove);

                    if (bestHeuristic > 0) {
                        parentSet.add(bestCandidate);
                        candSet.remove(bestCandidate);
                    }
                }
                AbstractVariable thisVariable = this.variables.get(variableName);
                AbstractVariable[] parents = new AbstractVariable[parentSet.size()];

                Object[] parentList = parentSet.toArray();

                for (int i = 0; i < parentSet.size(); i++) {
                    parents[i] = this.variables.get((String) parentList[i]);
                }

                thisVariable.updateStructure(parents, fittest);

                this.currentGenConnections += thisVariable.getParentsNames().size();
            }
        }
    }

    public int getCurrentGenDiscardedIndividuals() {
        return currentGenDiscardedIndividuals;
    }

    public HashMap<String, AbstractVariable> getVariables() {
        return this.variables;
    }

    public Integer getCurrentGenEvals() {
        return this.currentGenEvals;
    }

    public Integer getCurrentGenConnections() {
        return this.currentGenConnections;
    }
}
