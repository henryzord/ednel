package ednel.network;

import ednel.eda.individual.*;
import ednel.network.variables.AbstractVariable;
import org.apache.commons.math3.random.MersenneTwister;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;

import java.security.InvalidParameterException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.TimeoutException;

import static ednel.utils.MyMathUtils.lfactorial;
import static java.lang.Math.exp;
import static java.lang.Math.log;

public class DependencyNetwork {
    private final double learningRate;

    private HashMap<String, AbstractVariable> variables;

    /** A dictionary of dictionaries. Each key is a variable, and each value another dictionary. In this second dictionary,
     * keys are the type of relationship the variable has with its children: whether deterministic or probabilistic. The
     * values of this second dictionary are ArrayList of children names. */
    private HashMap<String, HashMap<String, ArrayList<String>>> graph;

    private MersenneTwister mt;
    private int burn_in;
    private int thinning_factor;
    /** Maximum number of parents that any variable may have at a given moment. */
    private int max_parents;

    /** Number of generations needed to perform learning of structure of graphical model */
    private final int delay_structure_learning;

    private HashMap<String, ArrayList<String>> bufferStructureLearning;

    private ArrayList<String> samplingOrder = null;

    // counters
    /** Discarded individuals of this generation */
    private int currentGenDiscardedIndividuals;
    private int currentGenEvals;
    private int currentGenConnections;
    private double currentGenMeanHeuristic;

    private HashMap<String, ArrayList<String>> lastFittestValues;

    private boolean no_cycles;
    private OptionHandler optionHandler;
    private final int timeout_individual;

    public DependencyNetwork(
            MersenneTwister mt, int burn_in, int thinning_factor, boolean no_cycles,
            double learningRate, int max_parents, int delay_structure_learning, int timeout_individual
    ) throws Exception {
        this.mt = mt;
        this.variables = new HashMap<>();

        this.delay_structure_learning = delay_structure_learning;

        this.bufferStructureLearning = new HashMap<>();

        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;

        this.no_cycles = no_cycles;

        this.max_parents = max_parents;  // global max parents

        this.learningRate = learningRate;

        this.timeout_individual = timeout_individual;

        this.currentGenEvals = 0;
        this.currentGenDiscardedIndividuals = 0;
        this.currentGenConnections = 0;
        this.currentGenMeanHeuristic = 0;

        this.lastFittestValues = null;

        this.readVariablesFromFiles();
        this.graph = DependencyNetwork.generateDeterministicGraph(this.variables);
        this.samplingOrder = DependencyNetwork.inferSamplingOrder(this.graph);
    }

    /**
     * Updates graph attribute of this object, but only with deterministic relationships
     */
    private static HashMap<String, HashMap<String, ArrayList<String>>> generateDeterministicGraph(HashMap<String, AbstractVariable> variables) {
        HashMap<String, HashMap<String, ArrayList<String>>> graph = new HashMap<>();

        for(AbstractVariable variable : variables.values()) {
            if(!graph.containsKey(variable.getName())) {
                HashMap<String, ArrayList<String>> temp = new HashMap<>();
                temp.put("deterministic", new ArrayList<>());
                temp.put("probabilistic", new ArrayList<>());
                graph.put(variable.getName(), temp);
            }
            for(String parentName : variable.getDeterministicParents()) {
                HashMap<String, ArrayList<String>> temp;
                if(!graph.containsKey(parentName)) {
                    temp = new HashMap<>();
                    temp.put("deterministic", new ArrayList<>());
                    temp.put("probabilistic", new ArrayList<>());
                } else {
                    temp = graph.get(parentName);
                }
                ArrayList<String> alc = temp.get("deterministic");
                alc.add(variable.getName());
                temp.put("deterministic", alc);
                graph.put(parentName, temp);
            }
        }
        return graph;
    }

    /**
     *
     * Gets deterministic children of all variables.
     *
     * @param variables A HashMap where the key is the variable name and the value a corresponding AbstractVariable
     *                  object
     * @return A HashMap where the key is the parent variable and the value a HashSet with deterministic children
     * of that variable.
     */
    private static HashMap<String, HashSet<String>> getEveryonesDeterministicChildren(HashMap<String, AbstractVariable> variables) {
        HashMap<String, HashSet<String>> children = new HashMap<>(variables.size());

        HashSet<String> toAdd = new HashSet<>();
        toAdd.addAll(variables.keySet());

        for(String var : variables.keySet()) {
            HashSet<String> parents = new HashSet<>();
            parents.addAll(variables.get(var).getDeterministicParents());
            for(String parent : parents) {
                HashSet<String> initial = children.getOrDefault(parent, new HashSet<String>());
                initial.add(var);
                children.put(parent, initial);
                toAdd.remove(parent);
            }
        }
        for(String var : toAdd) {
            children.put(var, new HashSet<>());
        }

        return children;
    }

    /**
     *
     * Gets deterministic parents of all variables.
     *
     * @param variables A HashMap where the key is the variable name and the value a corresponding AbstractVariable
     *                  object
     * @return A HashMap where the key is the child variable and the value a HashSet with deterministic parents
     * of that variable.
     */
    private static HashMap<String, HashSet<String>> getEveryonesDeterministicParents(HashMap<String, AbstractVariable> variables) {
        HashMap<String, HashSet<String>> parents = new HashMap<>(variables.size());

        for(String var : variables.keySet()) {
            HashSet<String> allParents = new HashSet<>();
            allParents.addAll(variables.get(var).getDeterministicParents());
            parents.put(var, allParents);
        }
        return parents;
    }

    /**
     *
     * Gets all parents (both deterministic and probabilistic) of all variables.
     *
     * @param variables A HashMap where the key is the variable name and the value a corresponding AbstractVariable
     *                  object
     * @return A HashMap where the key is the child variable and the value a HashSet with all parents (both
     * deterministic and probabilistic) of that variable.
     */
    private static HashMap<String, HashSet<String>> getEveryonesAllParents(HashMap<String, AbstractVariable> variables) {
        HashMap<String, HashSet<String>> parents = DependencyNetwork.getEveryonesDeterministicParents(variables);

        for(String var : variables.keySet()) {
            HashSet<String> local = parents.get(var);
            local.addAll(variables.get(var).getProbabilisticParents());
            parents.put(var, local);
        }
        return parents;
    }

    /**
     *
     * Gets all children (both deterministic and probabilistic) of all variables.
     *
     * @param variables A HashMap where the key is the variable name and the value a corresponding AbstractVariable
     *                  object
     * @return A HashMap where the key is the parent variable and the value a HashSet with all children (both
     * deterministic and probabilistic) of that variable.
     */
    private static HashMap<String, HashSet<String>> getEveryonesAllChildren(HashMap<String, AbstractVariable> variables) {
        HashMap<String, HashSet<String>> children = DependencyNetwork.getEveryonesDeterministicChildren(variables);

        for(String var : variables.keySet()) {
            HashSet<String> parents = variables.get(var).getProbabilisticParents();
            for(String parent : parents) {
                HashSet<String> local = children.getOrDefault(parent, new HashSet<String>());
                local.add(var);
                children.put(parent, local);
            }
        }
        return children;
    }

    /**
     * Infers sampling order from variables, based on the most requested variables in the dependency network.
     *
     * @param graph HashMap where keys are variable names and entries the list of its children
     * @return an ArrayList with the inferred sampling order
     */
    private static ArrayList<String> inferSamplingOrder(HashMap<String, HashMap<String, ArrayList<String>>> graph) {
        ArrayList<String> samplingOrder = new ArrayList<>(graph.size());
        HashSet<String> added_set = new HashSet<>();

        // parentsOf has the parents of a given variable
        HashMap<String, HashSet<String>> probParentsOf = new HashMap<>();
        HashMap<String, HashSet<String>> detParentsOf = new HashMap<>();

        for(String var : graph.keySet()) {
            ArrayList<String> probChildren = graph.get(var).get("probabilistic");
            ArrayList<String> detChildren = graph.get(var).get("deterministic");

            if(!probParentsOf.containsKey(var)) {
                probParentsOf.put(var, new HashSet<>());
            }
            if(!detParentsOf.containsKey(var)) {
                detParentsOf.put(var, new HashSet<>());
            }

            for(String child : probChildren) {
                HashSet<String> toAdd;
                if(probParentsOf.containsKey(child)) {
                    toAdd = probParentsOf.get(child);
                } else {
                    toAdd = new HashSet<>();
                }
                toAdd.add(var);
                probParentsOf.put(child, toAdd);
            }

            for(String child : detChildren) {
                HashSet<String> toAdd;
                if(detParentsOf.containsKey(child)) {
                    toAdd = detParentsOf.get(child);
                } else {
                    toAdd = new HashSet<>();
                }
                toAdd.add(var);
                detParentsOf.put(child, toAdd);
            }
        }

        int n_variables = graph.size();

        ArrayList<String> shuffableVariables = new ArrayList<>(graph.keySet());
        Collections.shuffle(shuffableVariables);

        // tries to add variables with the least amount of parents
        while(added_set.size() < n_variables) {  // while there are still variables to add
            ArrayList<String> added_now = new ArrayList<>();

            String best_candidate = null;
            int best_missing_prob = Integer.MAX_VALUE;
            int best_missing_det = Integer.MAX_VALUE;
            int best_voting = 0;

            boolean addedAny = false;
            for(String var : shuffableVariables) {
                Set<String> alreadyAddedProbParents = new HashSet<>(probParentsOf.get(var));
                Set<String> alreadyAddedDetParents = new HashSet<>(detParentsOf.get(var));
                alreadyAddedProbParents.retainAll(added_set);
                alreadyAddedDetParents.retainAll(added_set);

                int missing_prob_parents = probParentsOf.get(var).size() - alreadyAddedProbParents.size();
                int missing_det_parents = detParentsOf.get(var).size() - alreadyAddedDetParents.size();

                // if there are no missing parents, add right away
                if(missing_prob_parents == 0 && missing_det_parents == 0) {
                    addedAny = true;
                    samplingOrder.add(var);
                    added_set.add(var);
                    added_now.add(var);
                } else if(missing_det_parents <= best_missing_det) {
                    if(missing_prob_parents <= best_missing_prob) {
                        if((probParentsOf.get(var).size() + detParentsOf.get(var).size()) >= best_voting) {
                            best_missing_prob = missing_prob_parents;
                            best_missing_det = missing_det_parents;
                            best_candidate = var;
                            best_voting = probParentsOf.get(var).size() + detParentsOf.get(var).size();
                        }
                    }
                }
            }
            // will have to make sacrifices
            if(!addedAny) {
                samplingOrder.add(best_candidate);
                added_set.add(best_candidate);
                added_now.add(best_candidate);
            }
            shuffableVariables.removeAll(added_now);
        }
        return samplingOrder;
    }

    /**
     * Reads variables from csv files. CSV files are within jar of project.
     *
     * @throws Exception If any exception occurs
     */
    private void readVariablesFromFiles() throws Exception {
        this.currentGenConnections = 0;

        // method for reading from same jar
        this.optionHandler = new OptionHandler();

        Reflections reflections = new Reflections("distributions", new ResourcesScanner());
        Set<String> variablePaths = reflections.getResources(x -> true);

        for(Object variablePath : variablePaths) {
            AbstractVariable variable = AbstractVariable.fromPath(variablePath.toString(), this.mt);

            this.variables.put(variable.getName(), variable);
            this.currentGenConnections += this.variables.get(variable.getName()).getParentCount();
        }
    }

    /**
     * Samples a single individual.
     *
     * @return A HashMap with two items: the updated lastStart point in the solution space, and the option table
     *         generated for the sampled individual.
     * @throws Exception IF any exception occurs
     */
    private HashMap<String, HashMap<String, String>> sampleIndividual(HashMap<String, String> lastStart) throws Exception {
        HashMap<String, String> optionTable = new HashMap<>();

        for(String variableName : this.samplingOrder) {
            String sampledValue = this.variables.get(variableName).conditionalSampling(lastStart);
            lastStart.put(variableName, sampledValue);

            if(!String.valueOf(sampledValue).equals("null")) {
                String algorithmName = this.variables.get(variableName).getAlgorithmName();
                optionTable = this.optionHandler.handle(optionTable, variableName, algorithmName, sampledValue);
            }
        }
        HashMap<String, HashMap<String, String>> components = new HashMap<>();
        components.put("lastStart", lastStart);
        components.put("optionTable", optionTable);

        return components;
    }

    public Individual[] gibbsSampleAndAssignFitness(
            HashMap<String, String> lastStart, int sampleSize, FitnessCalculator fc, int seed,
            LocalDateTime start, Integer timeout
    ) throws Exception {
        Individual[] individuals = new Individual[sampleSize];

        this.currentGenEvals = 0;

        // burns some individuals
        for(int i = 0; i < burn_in; i++) {
            // updates currentLastStart and currentOptionTable
            HashMap<String, HashMap<String, String>> components = this.sampleIndividual(lastStart);
            lastStart = components.get("lastStart");
        }
        this.currentGenDiscardedIndividuals = burn_in;

        HashMap<String, String> initialSearchPoint = (HashMap<String, String>)lastStart.clone();

        int thinning_counter = 0;
        int individual_counter = 0;
        int inner_invalid_streak = 0;
        int outer_invalid_streak = 0;

        while(individual_counter < sampleSize) {  // while there are still individuals to sample
            LocalDateTime t1 = LocalDateTime.now();
            boolean overTime = (timeout > 0) && ((int)start.until(t1, ChronoUnit.SECONDS) > timeout);
            if(overTime) {
                break;
            }

            HashMap<String, HashMap<String, String>> components = this.sampleIndividual(lastStart);
            HashMap<String, String> optionTable = components.get("optionTable");
            lastStart = components.get("lastStart");
            thinning_counter += 1;

            // if this individual should be taken into account; otherwise
            // discards because thinning_factor is larger than zero
            if(thinning_counter >= this.thinning_factor) {
                try {
                    Individual individual = new Individual(optionTable, lastStart);
                    individual.setFitness(
                            fc.evaluateEnsemble(seed, individual, this.timeout_individual, false)
                    );

                    individuals[individual_counter] = individual;

                    thinning_counter = 0;
                    individual_counter += 1;

                    inner_invalid_streak = 0;
                    outer_invalid_streak = 0;

                    this.currentGenEvals += 1;
                } catch (InvalidParameterException | EmptyEnsembleException | NoAggregationPolicyException | TimeoutException e) {
                    // invalid individual generated
                    this.currentGenDiscardedIndividuals += 1;

                    inner_invalid_streak += 1;
                    if(inner_invalid_streak >= 5) {  // 5 is an arbitrary parameter
                        if(individual_counter > 0) {
                            lastStart = individuals[individual_counter - 1].getCharacteristics();
                            outer_invalid_streak += 1;
                            inner_invalid_streak = 0;
                        } else {
                            inner_invalid_streak = 0;
                            outer_invalid_streak = 5;
                        }

                        if(outer_invalid_streak >= 5) {
                            lastStart = initialSearchPoint;
                            outer_invalid_streak = 0;
                        }
                    }
                }
            } else {
                this.currentGenDiscardedIndividuals += 1;
            }
        }

        // copies sampled individuals
        Individual[] to_return =  new Individual [individual_counter];
        for(int i = 0; i < individual_counter; i++) {
            to_return[i] = individuals[i];
        }
        return to_return;
    }

    /**
     * TODO unverified! Generates all possible combinations of values between a surrogate variable and its parents.
     *
     * @param variableName Name of child variable
     * @return An ArrayList of HashMaps, where each HashMap is a dictionary where keys are variable names and values
     * the values assumed by variables in that combination. The size of the ArrayList gives how many combinations
     * are possible.
     */
    private ArrayList<HashMap<String, String>> generateCombinations(String variableName) {
        HashSet<String> parents = (HashSet<String>)this.variables.get(variableName).getDeterministicParents().clone();
        parents.addAll(this.variables.get(variableName).getProbabilisticParents());

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>(parents.size() + 1);
        ArrayList<String> thisUniqueValues = this.variables.get(variableName).getUniqueValues();
        for(String value : thisUniqueValues) {
            HashMap<String, String> data = new HashMap<>(parents.size() + 1);
            data.put(variableName, value);
            combinations.add(data);
        }

        for(String parent : parents) {
            ArrayList<String> values = this.variables.get(parent).getUniqueValues();
            int outer = values.size() * combinations.size();

            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>(outer);

            for(int j = 0; j < values.size(); j++) {
                for(int i = 0; i < combinations.size(); i++) {
                    HashMap<String, String> local = (HashMap<String, String>)combinations.get(i).clone();
                    local.put(parent, values.get(j));
                    new_combinations.add(local);
                }
            }
            combinations = new_combinations;
        }
        return combinations;
    }


    /**
     * Creates a HashMap, where each key is a variable name, and each value an ArrayList with the values of the
     * individuals that updated the GM probabilities in the current generation.
     *
     * @param selectionShare Proportion of individuals that were selected to update GM probabilities
     * @param sortedIndices Indices of individuals in the population, sorted in descendent order of fitness
     * @param population Actual individuals
     * @return A HashMap where each key is a variable in GM and the value an ArrayList of values for that variable in
     *         last generation's fittest population (as per selectionShare)
     */
    private HashMap<String, ArrayList<String>> getFittestIndividualsValues(float selectionShare, Integer[] sortedIndices, Individual[] population) {
        int to_select = Math.round(selectionShare * sortedIndices.length);

        Individual[] fittestIndividuals = new Individual[to_select];
        for(int i = 0; i < to_select; i++) {
            fittestIndividuals[i] = population[sortedIndices[i]];
        }

        HashMap<String, ArrayList<String>> currFittestValues = new HashMap<>();
        for(String var : this.samplingOrder) {
            currFittestValues.put(var, new ArrayList<>(fittestIndividuals.length));
        }
        for(Individual fit : fittestIndividuals) {
            for(String var : this.samplingOrder) {
                currFittestValues.get(var).add(fit.getCharacteristics().get(var));
            }
        }
        return currFittestValues;
    }

    public void update(Individual[] population, Integer[] sortedIndices, float selectionShare, int generation) throws Exception {
        HashMap<String, ArrayList<String>> currFittestValues = this.getFittestIndividualsValues(selectionShare, sortedIndices, population);

        if(this.learningRate > 0) {
            for(String var : this.samplingOrder) {
                ArrayList<String> values = this.bufferStructureLearning.getOrDefault(var, new ArrayList<>());
                values.addAll(currFittestValues.get(var));
                this.bufferStructureLearning.put(var, values);
            }
        }

        // only updates structure if there is a previous fittest population
        if((this.max_parents > 0) &&
                (generation > 0 && ((this.delay_structure_learning <= 1) || (generation % this.delay_structure_learning) == 0))) {
            this.updateStructure(this.bufferStructureLearning);
            this.bufferStructureLearning = new HashMap<>();
        }
        this.lastFittestValues = currFittestValues;

        this.updateProbabilities(currFittestValues, this.lastFittestValues);

        this.samplingOrder = DependencyNetwork.inferSamplingOrder(this.graph);
    }

    /**
     * Updates probabilities of this Dependency Network, using the pre-established structure (whether learnt in this
     * generation or not).
     * @param currFittestValues A HashMap where each key is a variable name and each value the array of values from the
     *                          current sub-population of fittest individuals.
     * @param lastFittestValues A HashMap where each key is a variable name and each value the array of values from the
     *                          previous sub-population of fittest individuals.
     * @throws Exception If any exceptions occurs.
     */
    public void updateProbabilities(
            HashMap<String, ArrayList<String>> currFittestValues,
            HashMap<String, ArrayList<String>> lastFittestValues
    ) throws Exception {

        for(String variableName : this.samplingOrder) {
            this.variables.get(variableName).updateProbabilities(currFittestValues, lastFittestValues, this.learningRate);
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
     * @param currentGenFittest HashMap of variable values for fittest individuals.
     * @throws Exception
     */
    private void updateStructure(HashMap<String, ArrayList<String>> currentGenFittest) throws Exception {
        this.currentGenConnections = 0;

        this.graph = DependencyNetwork.generateDeterministicGraph(this.variables);

        Collections.shuffle(this.samplingOrder);  // adds randomness to the process

        double n_computed_amis = 0;
        this.currentGenMeanHeuristic = 0;

        for(String variableName : this.samplingOrder) {
            HashSet<String> candSet = new HashSet<>(this.variables.keySet());  // candidates to be parents of a variable
            candSet.remove(variableName);  // removes itself, otherwise makes no sense
            candSet.removeAll(this.variables.get(variableName).getDeterministicParents());  // removes deterministic parents

            // removes deterministic children of this variable
            for(String child : graph.get(variableName).get("deterministic")) {
                candSet.remove(child);
            }

            // probabilistic parent set starts empty
            HashSet<String> probParentSet = new HashSet<>();

            // while there are still candidates, and space for probabilistic parents
            while((candSet.size() > 0) && (probParentSet.size() < this.max_parents)) {
                double bestHeuristic = -1;
                String bestCandidate = null;

                HashSet<String> toRemove = new HashSet<>();

                for(String candidate : candSet) {
                    if(this.no_cycles && DependencyNetwork.doesItInsertCycle(candidate, variableName, this.graph)) {
                        toRemove.add(candidate);
                    } else {
                        double heuristic = this.heuristic(
                                this.variables.get(variableName),
                                probParentSet,
                                this.variables.get(candidate),
                                currentGenFittest
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

                if(bestHeuristic >= 0.1) {
                    this.currentGenMeanHeuristic = this.currentGenMeanHeuristic + bestHeuristic;
                    n_computed_amis = n_computed_amis + 1.0;

                    probParentSet.add(bestCandidate);

                    HashMap<String, ArrayList<String>> temp = this.graph.get(bestCandidate);
                    ArrayList<String> updatedChildren = temp.get("probabilistic");
                    updatedChildren.add(variableName);
                    temp.put("probabilistic", updatedChildren);
                    this.graph.put(bestCandidate, temp);
                }
                candSet.remove(bestCandidate);
            }

            AbstractVariable thisVariable = this.variables.get(variableName);

            HashSet<String> thisVariableDetParents = this.variables.get(variableName).getDeterministicParents();

            AbstractVariable[] all_parents = new AbstractVariable[probParentSet.size() + thisVariableDetParents.size()];

            Object[] parentList = probParentSet.toArray();
            int parent_counter;
            for(parent_counter = 0; parent_counter < probParentSet.size(); parent_counter++) {
                all_parents[parent_counter] = this.variables.get((String)parentList[parent_counter]);
            }
            for(String detParentName : thisVariableDetParents) {
                all_parents[parent_counter] = this.variables.get(detParentName);
                parent_counter += 1;
            }
            thisVariable.updateStructure(all_parents);

            this.currentGenConnections += thisVariable.getParentCount();
        }
        this.currentGenMeanHeuristic /= n_computed_amis <= 0.0? 1.0 : n_computed_amis;
    }

    /**
     * Checks if connecting parent to child introduces a cycle in the graph.
     * @param parent Parent variable
     * @param child Child variable
     * @return true if introduces cycle, false otherwise
     */
    private static boolean doesItInsertCycle(String parent, String child, HashMap<String, HashMap<String, ArrayList<String>>> graph) {
        HashSet<String> visited = new HashSet<>();
        visited.add(parent);
        visited.add(child);

        Stack<String> stack = new Stack<>();

        String[] childrenType = new String[]{"deterministic", "probabilistic"};
        for(String childType : childrenType) {
            stack.addAll(graph.get(parent).get(childType));
            stack.addAll(graph.get(child).get(childType));
        }

        while(stack.size() > 0) {
            String current = stack.pop();
            if(visited.contains(current)) {
                return true;
            }
            visited.add(current);

            for(String childType : childrenType) {
                ArrayList<String> next = graph.get(current).get(childType);
                for(String item : next) {
                    if(visited.contains(item)) {
                        return true;
                    } else {
                        stack.addAll(graph.get(item).get(childType));
                    }
                }
            }
        }
        return false;
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

    public Double getCurrentGenMeanHeuristic() {
        return this.currentGenMeanHeuristic;
    }

    public OptionHandler getOptionHandler() {
        return this.optionHandler;
    }
}

