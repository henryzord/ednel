package ednel.network;

import ednel.network.variables.AbstractVariable;
import ednel.network.variables.ContinuousVariable;
import ednel.network.variables.DiscreteVariable;
import ednel.eda.individual.Individual;
import ednel.utils.comparators.Argsorter;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import weka.core.Instances;
import org.apache.commons.math3.special.Gamma;

import java.io.*;
import java.nio.file.Files;
import java.util.*;

public class DependencyNetwork {
    private final float learningRate;
    private final int n_generations;
    private HashMap<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private JSONObject options;
    private int burn_in;
    private int thinning_factor;

    /**
     * The number of nearest neighbors to consider when
     * calculating the mutual information between discrete and
     * continuous variables.
     */
    private int nearest_neighbor;
    /**
     * Maximum number of parents that any variable may have at a given moment.
     */
    private int max_parents;

    private ArrayList<String> samplingOrder = null;

    private ArrayList<HashMap<String, AbstractVariable>> pastVariables;

    private HashMap<String, String> lastStart;

    private int currentGenDiscardedIndividuals;
    private int currentGenEvals;
    private int currentGenConnections;
//    private HashMap<String, Integer> connectionsCount;

    public DependencyNetwork(
            MersenneTwister mt, String resources_path, int burn_in, int thinning_factor,
            float learningRate, int n_generations, int nearest_neighbor, int max_parents
    ) throws Exception {
        this.mt = mt;
        this.variables = new HashMap<>();

        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;
        this.nearest_neighbor = nearest_neighbor;
        this.max_parents = max_parents;  // global max parents

        this.learningRate = learningRate;
        this.n_generations = n_generations;

        this.currentGenEvals = 0;
        this.currentGenDiscardedIndividuals = 0;
        this.currentGenConnections = 0;

        this.lastStart = null;

        this.readVariablesFromFiles(resources_path);
        this.samplingOrder = DependencyNetwork.inferSamplingOrder(this.variables);

    }

    /**
     * Infers sampling order from variables, based on the most requested variables in the dependency network.
     *
     * @param variables dictionary of names and AbstractVariable objects
     * @return sampling order
     */
    private static ArrayList<String> inferSamplingOrder(HashMap<String, AbstractVariable> variables)  {

        ArrayList<String> samplingOrder = new ArrayList<>(variables.size());
        HashSet<String> added_set = new HashSet<>();

        int added_count = 0;
        int current_vote_bar = 0;
        boolean recompute_votes = true;
        HashMap<String, Integer> votes = new HashMap<>();
        while(added_count < variables.size()) {
            if(recompute_votes) {
                votes = new HashMap<>();
                for(String var : variables.keySet()) {
                    HashSet<String> set = new HashSet<>();
                    set.addAll(variables.get(var).getFixedBlocking());
                    set.addAll(variables.get(var).getProbabilisticParents());
                    set.removeAll(added_set);  // removes variables already included

                    for(String overVar : set) {
                        if(!votes.containsKey(overVar)) {
                            votes.put(overVar, 1);
                        } else {
                            votes.put(overVar, votes.get(overVar) + 1);
                        }
                    }
                }
                recompute_votes = false;
            }
            for(String var : variables.keySet()) {
                if(votes.getOrDefault(var, 0) == current_vote_bar) {
                    samplingOrder.add(var);
                    added_set.add(var);
                    added_count += 1;
                    recompute_votes = true;
                }
            }
            current_vote_bar += 1;
        }
        Collections.reverse(samplingOrder);
        return samplingOrder;
    }

    /**
     * Reads variables from file.
     *
     * @param resources_path Path to resources folder.
     * @throws Exception If any exception occurs
     */
    private void readVariablesFromFiles(String resources_path) throws Exception {
        this.currentGenConnections = 0;

        JSONParser jsonParser = new JSONParser();
        options = (JSONObject)jsonParser.parse(new FileReader(resources_path + File.separator + "options.json"));
        JSONObject initialBlocking = (JSONObject)jsonParser.parse(new FileReader(resources_path + File.separator + "blocking.json"));
        Set<String> variables_names  = (Set<String>)initialBlocking.keySet();

        Object[] algorithmsPaths = Files.list(
                new File(resources_path + File.separator + "distributions").toPath()
        ).toArray();

        for(int i = 0; i < algorithmsPaths.length; i++) {
            Object[] variablePaths = Files.list(new File(algorithmsPaths[i].toString()).toPath()).toArray();

            for(Object variablePath : variablePaths) {
                AbstractVariable newVariable = AbstractVariable.fromPath(variablePath.toString(), initialBlocking, this.max_parents, this.mt);
                this.variables.put(newVariable.getName(), newVariable);
                this.currentGenConnections += this.variables.get(newVariable.getName()).getParentCount();
            }
        }
    }

    /**
     * Samples a single individual.
     * @return A new individual, in the format of a dictionary of characteristics.
     * @throws Exception IF any exception occurs
     */
    private HashMap<String, String> sampleIndividual() throws Exception {
        HashMap<String, String> optionTable = new HashMap<>();

        for(String variableName : this.samplingOrder) {
            String sampledValue = this.variables.get(variableName).conditionalSampling(lastStart);
            lastStart.put(variableName, sampledValue);

            if(!String.valueOf(sampledValue).equals("null")) {
                String algorithmName = this.variables.get(variableName).getAlgorithmName();

                JSONObject optionObj = (JSONObject)options.getOrDefault(variableName, null);
                if(optionObj == null) {
                    optionObj = (JSONObject)options.getOrDefault(sampledValue, null);
                }

                // checks whether this is an option
                if(optionObj != null) {
                    Boolean presenceMeans = (Boolean)optionObj.get("presenceMeans");
                    String optionName = String.valueOf(optionObj.get("optionName"));
                    String dtype = String.valueOf(optionObj.get("dtype"));

                    if(dtype.equals("np.bool")) {
                        if(String.valueOf(sampledValue).toLowerCase().equals("false")) {
                            if(!presenceMeans) {
                                optionTable.put(algorithmName, (optionTable.getOrDefault(algorithmName, "") + " " + optionName).trim());
                            }
                        }
                        if(String.valueOf(sampledValue).toLowerCase().equals("true")) {
                            if(presenceMeans) {
                                optionTable.put(algorithmName, (optionTable.getOrDefault(algorithmName, "") + " " + optionName).trim());
                            }
                        }
                    } else if(dtype.equals("dict")) {
                        JSONObject dict = (JSONObject)((new JSONParser()).parse(optionName));

                        optionTable.put(
                                algorithmName, (
                                        optionTable.getOrDefault(algorithmName, "") + " " + dict.get(sampledValue)
                                ).trim()
                        );
                    } else {
                        optionTable.put(
                                algorithmName, (
                                        optionTable.getOrDefault(algorithmName, "") + " " + optionName + " " + sampledValue
                                ).trim()
                        );
                    }
                }
            }
        }
        return optionTable;
    }

    public Individual[] gibbsSample(HashMap<String, String> lastStart, int sampleSize, Instances train_data) throws Exception {
        Individual[] individuals = new Individual [sampleSize];

        this.lastStart = lastStart;
        this.currentGenEvals = 0;

        int outerCounter = 0;
        int individualCounter = 0;

        // burns some individuals
        for(int i = 0; i < burn_in; i++) {
            // updates currentLastStart and currentOptionTable
            this.sampleIndividual();
        }
        this.currentGenDiscardedIndividuals = burn_in;

        while(individualCounter < sampleSize) {
            HashMap<String, String> optionTable = this.sampleIndividual();
            outerCounter += 1;

            // TODO only here for debugging purposes. can be removed later
//            String[] copyOptions = (String[])options.clone();
            try {
                Individual individual = new Individual(optionTable, this.lastStart, train_data);
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
        HashSet<String> parents = this.variables.get(variableName).getProbabilisticParentsNames();

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>(parents.size() + 1);
        Object[] thisUniqueValues = this.variables.get(variableName).getUniqueShadowvalues().toArray();
        for(Object value : thisUniqueValues) {
            HashMap<String, String> data = new HashMap<>(parents.size() + 1);
            data.put(variableName, value.toString());
            combinations.add(data);
        }

        for(String parent : parents) {
            Object[] values = this.variables.get(parent).getUniqueShadowvalues().toArray();
            int outer = values.length * combinations.size();

            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>(outer);

            for(int j = 0; j < values.length; j++) {
                for(int i = 0; i < combinations.size(); i++) {
                    HashMap<String, String> local = (HashMap<String, String>) combinations.get(i).clone();
                    local.put(parent, values[j].toString());
                    new_combinations.add(local);
                }
            }
            combinations = new_combinations;
        }
        return combinations;
    }

    public void update(Individual[] population, Integer[] sortedIndices, float selectionShare) throws Exception {
        int to_select = Math.round(selectionShare * sortedIndices.length);

        // collects data from fittest individuals
        Individual[] fittestIndividuals = new Individual[to_select];
        for(int i = 0; i < to_select; i++) {
            fittestIndividuals[i] = population[sortedIndices[i]];
        }

        HashMap<String, ArrayList<String>> fittestValues = new HashMap<>();
        for(String var : this.samplingOrder) {
            fittestValues.put(var, new ArrayList<>(fittestIndividuals.length));
        }
        for(Individual fit : fittestIndividuals) {
            for(String var : this.samplingOrder) {
                fittestValues.get(var).add(fit.getCharacteristics().get(var));
            }
        }

        this.updateStructure(fittestValues);
        this.updateProbabilities(fittestValues, fittestIndividuals);
        this.samplingOrder = DependencyNetwork.inferSamplingOrder(this.variables);
    }

    public void updateProbabilities(HashMap<String, ArrayList<String>> fittestValues, Individual[] fittest) throws Exception {
        for(String variableName : this.samplingOrder) {
            this.variables.get(variableName).updateProbabilities(fittestValues, fittest, this.learningRate, this.n_generations);
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
        AbstractVariable child, HashSet<String> parentSet, AbstractVariable candidate,
        HashMap<String, ArrayList<String>> fittest
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
                a_counts.get(String.valueOf(a[i])) + 1
            );
            b_counts.put(
                b[i],
                b_counts.get(String.valueOf(b[i])) + 1
            );

            String joint_name = String.format("%s,%s", String.valueOf(a[i]), String.valueOf(b[i]));

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
    private double mutualInformation(
            AbstractVariable a, AbstractVariable b, HashMap<String, ArrayList<String>> fittest
    ) throws Exception {
        String a_name = a.getName();
        String b_name = b.getName();

        int n_data = fittest.get(a_name).size();

        Double[] aContinuous = new Double [n_data];
        Double[] bContinuous =  new Double [n_data];

        String[] aDiscrete = new String [n_data];
        String[] bDiscrete = new String [n_data];

        int a_lack_count = 0;
        int b_lack_count = 0;

        for(int i = 0; i < n_data; i++) {
            String a_value = String.valueOf(fittest.get(a_name).get(i));
            String b_value = String.valueOf(fittest.get(b_name).get(i));

            if(a instanceof DiscreteVariable) {
                aDiscrete[i] = a_value;
            } else {
                // adds noise, as suggested by Kraskov et al
                double noise = (double)mt.nextInt(10) / 10e-10;
                if(!a_value.toLowerCase().equals("null")) {
                    aContinuous[i] = Double.parseDouble(a_value) + noise;
                } else {
                    aContinuous[i] = (((ContinuousVariable)a).getMinValue() - 1)  + noise;
                    a_lack_count += 1;
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
                    b_lack_count += 1;
                }
            }
        }

        // the two variables are continuous, but they severely lack data to compute mutual information (e.g. only
        // one data point for each distribution)
        if((n_data - a_lack_count) <= 1 || (n_data - b_lack_count) <= 1) {
            return 0;
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

    /**
     * Updates structure of dependency network/probabilistic graphical model.
     * Uses an heuristic to detect which variables are more suitable to be parents of a given child variable.
     *
     * @param fittest HashMap of variable values for fittest individuals.
     * @throws Exception
     */
    private void updateStructure(HashMap<String, ArrayList<String>> fittest) throws Exception {
        this.currentGenConnections = 0;

        for(String variableName : this.samplingOrder) {
            // candidates to be parents of a variable
            HashSet<String> candSet = new HashSet<>(this.samplingOrder.size());
            candSet.addAll(this.variables.keySet());  // adds all variables
            candSet.remove(variableName);  // removes itself, otherwise makes no sense
            candSet.removeAll(this.variables.get(variableName).getFixedCannotLink());  // removes cannot link variables

            // probabilistic parent set starts empty
            HashSet<String> parentSet = new HashSet<>();

            while ((candSet.size() > 0) && (parentSet.size() < this.max_parents)) {
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
            // TODo now the magic happens

            AbstractVariable thisVariable = this.variables.get(variableName);
            AbstractVariable[] mutableParents = new AbstractVariable[parentSet.size()];

            Object[] parentList = parentSet.toArray();
            for (int i = 0; i < parentSet.size(); i++) {
                mutableParents[i] = this.variables.get((String)parentList[i]);
            }

            thisVariable.updateStructure(mutableParents, fittest);

            this.currentGenConnections += thisVariable.getParentCount();
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
