package ednel.network;

import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.network.variables.AbstractVariable;
import ednel.network.variables.ContinuousVariable;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import smile.validation.AdjustedMutualInformation;

import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import static ednel.utils.MyMathUtils.lfactorial;
import static java.lang.Math.exp;
import static java.lang.Math.log;

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
    private static ArrayList<String> inferSamplingOrder(HashMap<String, AbstractVariable> variables) {
        ArrayList<String> samplingOrder = new ArrayList<>(variables.size());
        HashSet<String> added_set = new HashSet<>();

        HashMap<String, HashSet<String>> parentsOf = new HashMap<>();

        // computes votes for all variables
        HashMap<String, Integer> votes = new HashMap<>();
        for(String var : variables.keySet()) {
            HashSet<String> set = new HashSet<>();
            set.addAll(variables.get(var).getFixedBlocking());
            set.addAll(variables.get(var).getProbabilisticParents());
            set.removeAll(added_set);  // removes variables already included
            parentsOf.put(var, set);

            for(String overVar : set) {
                if(!votes.containsKey(overVar)) {
                    votes.put(overVar, 1);
                } else {
                    votes.put(overVar, votes.get(overVar) + 1);
                }
            }
        }

        // tries to add variables with the least amount of parents
        while(added_set.size() < variables.size()) {  // while there are still variables to add
            String best_candidate = null;
            int best_missing_parents = Integer.MAX_VALUE;
            int best_voting = 0;

            boolean addedAny = false;
            for(String var : variables.keySet()) {
                if(added_set.contains(var)) {
                    continue;
                }
                Set<String> intersection = new HashSet<>(parentsOf.get(var)); // use the copy constructor
                intersection.retainAll(added_set);

                int missing_parents = parentsOf.get(var).size() - intersection.size();

                // if there are no missing parents, add right away
                if(missing_parents == 0) {
                    addedAny = true;
                    samplingOrder.add(var);
                    added_set.add(var);
                } else if((missing_parents < best_missing_parents) && (votes.getOrDefault(var, 0) >= best_voting)) {
                    best_missing_parents = missing_parents;
                    best_candidate = var;
                    best_voting = votes.getOrDefault(var, 0);
                }
            }
            // will have to make sacrifices
            if(!addedAny) {
                samplingOrder.add(best_candidate);
                added_set.add(best_candidate);
            }
        }
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
        Set<String> variables_names = (Set<String>)initialBlocking.keySet();

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
     *
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
                    String algorithmOptions = optionTable.getOrDefault(algorithmName, "");

                    if(algorithmOptions.contains(variableName)) {
                        optionTable.put(algorithmName, algorithmOptions.replace(variableName, sampledValue));
                    }

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

    public HashMap<String, Object[]> gibbsSample(HashMap<String, String> lastStart, int sampleSize, FitnessCalculator fc, int seed) throws Exception {
        Individual[] individuals = new Individual[sampleSize];
        Double[] fitnesses = new Double[sampleSize];

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

            try {
                if(outerCounter >= this.thinning_factor) {
                    Individual individual = new Individual(optionTable, this.lastStart);
                    fitnesses[individualCounter] = fc.evaluateEnsemble(seed, individual);
                    individuals[individualCounter] = individual;

                    outerCounter = 0;

                    individualCounter += 1;
                    this.currentGenEvals += 1;
                } else {
                    this.currentGenDiscardedIndividuals += 1;
                }
            } catch (Exception e) {  // invalid individual generated
                this.currentGenDiscardedIndividuals += 1;
            }
        }
        this.lastStart = null;

        return new HashMap<String, Object[]>(){{
            put("population", individuals);
            put("fitnesses", fitnesses);
        }};
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
                    HashMap<String, String> local = (HashMap<String, String>)combinations.get(i).clone();
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
     * J.A. Gámez, J.L. Mateo, J.M. Puerta. EDNA: Estimation of Dependency Networks Algorithm.
     *
     * @param child     Current child variable
     * @param parentSet Parents of this variable
     * @param candidate Candidate parent for this variable
     * @param fittest   Fittest individuals from current generation
     * @return A modified mutual information metric, which is greater than zero if the candidate parent
     * is significantly correlated to this variable, or negative otherwise
     * @throws Exception If any exception occurs
     */
    private double heuristic(
            AbstractVariable child, HashSet<String> parentSet, AbstractVariable candidate,
            HashMap<String, ArrayList<String>> fittest
    ) throws Exception {
        double localMIs = 0;
        for(String parent : parentSet) {
            localMIs += this.getAdjustedMutualInformation(candidate, this.variables.get(parent), fittest);
        }
        return this.getAdjustedMutualInformation(child, candidate, fittest) - (localMIs / (parentSet.size() + 1));
    }

    /**
     * Computes (unadjusted) mutual information between two discrete variables.
     *
     * @param contigencyMatrix The contigency matrix for two discrete variables.
     * @param N                number of combinations in contigencyMatrix
     * @param ais              Number of occurrences in fittest population for each a value
     * @param bjs              Number of occurrences in fittest population for each b value
     * @return (unadjusted) mutual information between two discrete variables.
     */
    private static double getMutualInformation(HashMap<String, HashMap<String, Integer>> contigencyMatrix, int N, HashMap<String, Integer> ais, HashMap<String, Integer> bjs) {
        double mi = 0;
        for(String ai : contigencyMatrix.keySet()) {
            for(String bj : contigencyMatrix.get(ai).keySet()) {
                double p_ab = (contigencyMatrix.get(ai).get(bj) / (double)N);
                double p_a = ais.get(ai) / (double)N;
                double p_b = bjs.get(bj) / (double)N;

                mi += p_ab * log(p_ab / (p_a * p_b));
            }
        }
        return mi;
    }

    /**
     * Generates a contigency table for discrete variables a and b. That is, for each discrete value that a can assume,
     * computes how many times (for each b value) that combination occurred in the fittest population.
     * <p>
     * Null values are not taken into consideration, since no variable can (probabilistic) assume a null value.
     * <p>
     * The table is - atcd  implementation level - a dictionary of dictionaries, where the super-dictionary contain variable
     * a values and subdictionary contain variable b values. The value of the subdictionary is the number of occurences
     * of (A=a, B=b) values in the fittest population.
     *
     * @param a       First discrete variable
     * @param b       Second discrete variable
     * @param fittest Fittest population
     * @return contigency table for variables a and b.
     * @throws Exception if any of the variables is not Discrete.
     */
    private static HashMap<String, HashMap<String, Integer>> getContigencyMatrix(
            AbstractVariable a, AbstractVariable b, HashMap<String, ArrayList<String>> fittest) throws Exception {

        if(a instanceof ContinuousVariable || b instanceof ContinuousVariable) {
            throw new Exception("Both variables must be discrete!");
        }

        HashMap<String, HashMap<String, Integer>> table = new HashMap<>();

        ArrayList<String> a_values = fittest.get(a.getName());
        ArrayList<String> b_values = fittest.get(b.getName());

        if(a_values.size() != b_values.size()) {
            throw new Exception("Bad built fittest population dictionary!");
        }

        for(int i = 0; i < a_values.size(); i++) {
            if(String.valueOf(a_values.get(i)).equals("null") || String.valueOf(b_values.get(i)).equals("null")) {
                continue;
            }

            if(!table.containsKey(a_values.get(i))) {
                table.put(a_values.get(i), new HashMap<>());
            }
            HashMap<String, Integer> subDic = table.get(a_values.get(i));

            if(!subDic.containsKey(b_values.get(i))) {
                subDic.put(b_values.get(i), 1);
            } else {
                subDic.put(b_values.get(i), subDic.get(b_values.get(i)) + 1);
            }
            table.put(a_values.get(i), subDic);
        }
        return table;
    }

    /**
     * The expected Mutual Information value for two random discrete variables, as given by
     * https://en.wikipedia.org/wiki/Adjusted_mutual_information#Adjustment_for_chance
     * <p>
     * This means that mutual information will be zero if the two variables are correlated by chance, and 1
     * for perfect correlation.
     *
     * The code is an slight adaptation from slime implementation:
     * https://github.com/haifengl/smile/blob/1826b2f0fd9ba57ec0956792f00a419e950c850f/core/src/main/java/smile/validation/AdjustedMutualInformation.java#L133
     *
     * @return Adjusted Mutual Information
     */
    private static double getExpectedMutualInformation(int N, HashMap<String, Integer> ais, HashMap<String, Integer> bjs) {
        double expect = 0.0;
        for(String a_val : ais.keySet()) {
            for(String b_val : bjs.keySet()) {
                int ai = ais.get(a_val);
                int bj = bjs.get(b_val);

                int lower_limit = Math.max(1, ai + bj - N);
                int upper_limit = Math.min(ai, bj);
                for(int nij = lower_limit; nij <= upper_limit; nij++) {
                    expect += (nij / (double)N) * log((N * nij) / (double)(ai * bj)) * exp(
                            (lfactorial(ai) + lfactorial(bj) + lfactorial(N - ai) + lfactorial(N - bj))
                            - (lfactorial(N) + lfactorial(nij) + lfactorial(ai - nij) + lfactorial(bj - nij) +
                                    lfactorial(N - ai - bj + nij))
                    );
                }
            }
        }
        return expect;
    }

    /**
     * Calculates entropy for a given discrete variable.
     *
     * @param counts All values that a can assume, with the count of times a assumed that value, in the relevant elite subset.
     * @param N      number of individuals in relevant elite.
     * @return The entropy for the given variable
     */
    private static double getEntropy(HashMap<String, Integer> counts, int N) {
        double entropy = 0;
        for(String key : counts.keySet()) {
            entropy += (counts.get(key) / (double)N) * log(counts.get(key) / (double)N);
        }

        return -entropy;
    }

    /**
     * Computes adjusted mutual information between two discrete variables.
     *
     * @param a       First variable
     * @param b       Second variable
     * @param fittest Array of fittest individuals from the current generation
     * @return Adjusted mutual information between variables
     * @throws Exception If any exception occurs
     */
    private double getAdjustedMutualInformation(
            AbstractVariable a, AbstractVariable b, HashMap<String, ArrayList<String>> fittest
    ) throws Exception {

        HashMap<String, HashMap<String, Integer>> contigencyMatrix = DependencyNetwork.getContigencyMatrix(a, b, fittest);
        int N = 0;  // number of individuals in relevant-elite

        HashMap<String, Integer> ais = new HashMap<>();
        HashMap<String, Integer> bjs = new HashMap<>();

        for(String a_val : contigencyMatrix.keySet()) {
            for(String b_val : contigencyMatrix.get(a_val).keySet()) {
                int nij = contigencyMatrix.get(a_val).get(b_val);

                N += nij;

                if(!ais.containsKey(a_val)) {
                    ais.put(a_val, 0);
                }
                ais.put(a_val, ais.get(a_val) + nij);

                if(!bjs.containsKey(b_val)) {
                    bjs.put(b_val, 0);
                }
                bjs.put(b_val, bjs.get(b_val) + nij);
            }
        }

        double mi = DependencyNetwork.getMutualInformation(contigencyMatrix, N, ais, bjs);
        double e_mi = DependencyNetwork.getExpectedMutualInformation(N, ais, bjs);
        double a_entropy = DependencyNetwork.getEntropy(ais, N);
        double b_entropy = DependencyNetwork.getEntropy(bjs, N);

        return Math.min(1, Math.max(0, mi - e_mi / (Math.max(a_entropy, b_entropy) - e_mi)));
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
            candSet.removeAll(this.variables.get(variableName).getCannotLink());  // removes cannot link variables

            // probabilistic parent set starts empty
            HashSet<String> parentSet = new HashSet<>();

            while((candSet.size() > 0) && (parentSet.size() < this.max_parents)) {
                double bestHeuristic = -1;
                String bestCandidate = null;

                HashSet<String> toRemove = new HashSet<>();

                for(String candidate : candSet) {
                    if(this.variables.get(candidate).getCannotLink().contains(variableName)) {
                        toRemove.add(candidate);
                    } else {
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
                }
                candSet.removeAll(toRemove);

                if(bestHeuristic > 0) {
                    parentSet.add(bestCandidate);
                    candSet.remove(bestCandidate);
                }
            }
            AbstractVariable thisVariable = this.variables.get(variableName);
            AbstractVariable[] mutableParents = new AbstractVariable[parentSet.size()];

            Object[] parentList = parentSet.toArray();
            for(int i = 0; i < parentSet.size(); i++) {
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

    public ArrayList<String> getSamplingOrder() {
        return this.samplingOrder;
    }
}

