package ednel.network.variables;

import ednel.eda.individual.Individual;
import ednel.utils.CombinationNotPresentException;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.json.simple.JSONObject;
import org.apache.commons.math3.random.MersenneTwister;

import java.io.*;
import java.util.*;

public abstract class AbstractVariable {
    protected String name;

    /**
     * Mutable parents from previous generation.
     */
    protected HashSet<String> lastgen_prob_parents;

    /**
     * Probabilistic parents of this variable.
     */
    protected HashSet<String> prob_parents;

    protected HashMap<String, Boolean> isParentContinuous;
    protected int max_parents;

    /**
     * Unique values this variable can have. Does not contain any null values.
     * If there is a null value among the values, create a new
     * Shadowvalue for it:
     *
     * Shadowvalue sv = new Shadowvalue(
     *     String.class.getMethod("toString"),
     *     null
     * );
     *
     */
    protected HashSet<Shadowvalue> uniqueShadowvalues;
    protected HashMap<Shadowvalue, Double> unconditionalProbabilities;

    /**
     * Should not contain any null values.
     * If there is a null value among the values, create a new
     * Shadowvalue for it:
     *
     * Shadowvalue sv = new Shadowvalue(
     *     String.class.getMethod("toString"),
     *     null
     * );
     *
     */
    protected ArrayList<Shadowvalue> values;
    protected ArrayList<Double> probabilities;

//    protected ArrayList<Shadowvalue> oldValues;
//    protected ArrayList<Double> oldProbabilities;

    /**
     * A variable has a set of "cannot link" variables, which this variable has no probabilistic relationship with them.
     * For example: J48_pruning is probabilistically independent of J48; it is actually deterministically dependent
     * on J48 being true.
     * There are no values if J48 = false for J48_pruning.
     *
     * This attribute contains names of variables that can never be linked to the current variable.
     */
    protected HashSet<String> fixedCannotLink;

    /**
     * A variable has a set of "cannot link" variables, which this variable has no probabilistic relationship with them.
     * For example: J48_pruning is probabilistically independent of J48; it is actually deterministically dependent
     * on J48 being true.
     * There are no values if J48 = false for J48_pruning.
     *
     * This attribute contains names of variables that can change from one generation to another.
     */
    protected HashSet<String> mutableCannotLink;

    /**
     * Blocking variables are variables that must be sampled before this variable can be sampled.
     * If any of the blocking variables is not present in the current generation of the Gibbs Sampler,
     * this variable should not be sampled too.
     *
     * This attribute contains information on variables that never leave to block the current variable.
     */
    protected HashMap<String, ArrayList<String>> fixedBlocking;

    /**
     * Blocking variables are variables that must be sampled before this variable can be sampled.
     * If any of the blocking variables is not present in the current generation of the Gibbs Sampler,
     * this variable should not be sampled too.
     *
     * This attribute contains information on variables that may change from one generation to another.
     */
    protected HashMap<String, ArrayList<String>> mutableBlocking;

    /**
     * Should not contain null values. If a null value is present, then it is "null" (i.e. a string)
     */
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;

    protected String[] variablesCombinations;

    protected MersenneTwister mt;

    protected static String[] singleParentVariables = {"J48_reducedErrorPruning", "J48_unpruned", "PART_reducedErrorPruning", "PART_unpruned"};

    public AbstractVariable(
            String name, JSONObject fixedBlocking, HashMap<String, Boolean> isParentContinuous,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> initial_values, ArrayList<Double> probabilities,
            MersenneTwister mt, int max_parents) throws Exception {

        this.name = name;
        this.fixedBlocking = AbstractVariable.readBlockingJSONObject(fixedBlocking);
        this.mutableBlocking = new HashMap<>();

        this.lastgen_prob_parents = new HashSet<>();
        this.prob_parents = new HashSet<>();

        this.isParentContinuous = isParentContinuous;


        this.values = new ArrayList<>(initial_values.size());
        this.uniqueShadowvalues = new HashSet<>();

        if(this.getClass().equals(DiscreteVariable.class)) {
            for (String value : initial_values) {
                Shadowvalue sv = new Shadowvalue(
                        String.class.getMethod("toString"),
                        value
                );
                this.values.add(sv);
                this.uniqueShadowvalues.add(sv);
            }
        } else {
                for(int i = 0; i < initial_values.size(); i++) {
                    HashMap<String, Double> properties = ((ContinuousVariable)this).fromStringToProperty(initial_values.get(i));
                    ((ContinuousVariable)this).a_max = properties.get("a_max");
                    ((ContinuousVariable)this).a_min = properties.get("a_min");
                    ((ContinuousVariable)this).scale = properties.get("scale");
                    ((ContinuousVariable)this).loc_init = properties.get("loc");
                    ((ContinuousVariable)this).scale_init = properties.get("scale_init");

                    Shadowvalue sv;
                    if(properties.containsKey("means")) {
                        throw new Exception("not implemented yet!");  // TODO implement
                    } else {
                        sv = new ShadowNormalDistribution(this.mt, properties);
                    }
                    this.values.add(sv);
                    this.uniqueShadowvalues.add(sv);
                }
            }

        this.setUnconditionalProbabilities();

        this.probabilities = probabilities;

        this.fixedCannotLink = new HashSet<>();
        this.fixedCannotLink.addAll(this.fixedBlocking.keySet());

        String algorithmName = this.getAlgorithmName();

        if(!this.name.equals(algorithmName)) {
            this.fixedCannotLink.add(algorithmName);
        }

        this.table = table;

        this.initVariablesCombinations();

        this.mt = mt;

        this.max_parents = max_parents;
        for(String otherName : singleParentVariables) {
            if(otherName.equals(name)) {
                this.max_parents = 1;
                break;
            }
        }
    }

    /**
     * Converts a JSONObject that stores blocking variables-values pairs to a HashMap.
     * @param obj JSONObject with blocking values
     * @return HashMap with blocking values
     */
    private static HashMap<String, ArrayList<String>> readBlockingJSONObject(JSONObject obj) {
        HashMap<String, ArrayList<String>> map = new HashMap<>();

        for(Object var : obj.keySet()) {
            map.put((String)var, (ArrayList<String>)obj.get(var));
        }
        return map;
    }

    /**
     * Checks whether the variables that the current variable deterministically depends are present in the current
     * sample of Gibbs Sampler.
     * @param lastStart Current configuration of values to variables in the Gibbs Sampler.
     * @return True if a value can be sampled from this variable; false otherwise.
     */
    private boolean passBlockingTest(HashMap<String, String> lastStart) {
        // boolean proceed = true;
        //            for(String block : this.variables.get(variableName).getAllBlocking()) {
        //                // if variable was not sampled, or classifier of that variable is absent in ensemble
        //                // (also works if variable name is algorithm name)
        //                if(String.valueOf(lastStart.get(block)).equals("null") || lastStart.get(AbstractVariable.getAlgorithmName(block)).equals("false")) {
        //                    lastStart.put(variableName, null);
        //                    proceed = false;
        //                    break;
        //                }
        //            }

        ArrayList<HashMap<String, ArrayList<String>>> toIter = new ArrayList<HashMap<String, ArrayList<String>>>(){{
            add(fixedBlocking);
            add(mutableBlocking);
        }};

        for(HashMap<String, ArrayList<String>> block : toIter) {
            for(String var : block.keySet()) {
                boolean pass = false;
                for(String value : block.get(var)) {
                    if(String.valueOf(lastStart.getOrDefault(var, null)).equals(value)) {
                        pass = true;
                        break;
                    }
                }
                if(!pass) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Given any name of a variable, Returns the algorithm to which the variable is surrogate
     */
    public static String getAlgorithmName(String variableName) {
        return variableName.split("_")[0];
    }

    /**
     * Returns the algorithm to which this variable is surrogate
     */
    public String getAlgorithmName() {
        return this.name.split("_")[0];
    }

    /**
     * For each entry in the probabilities attribute in this object, writes the combination of probabilistic parents
     * parents/this variable values (in this order) that is associated with that probability.
     */
    protected void initVariablesCombinations() {
        int n_combinations = 1;
        for(String val : this.table.keySet()) {
            n_combinations *= this.table.get(val).size();
        }

        this.variablesCombinations = new String [n_combinations];
        Arrays.fill(this.variablesCombinations, "");
        this.addToVariablesCombinations(this.prob_parents);
        this.addToVariablesCombinations(new HashSet<String>(){{
            add(name);
        }});
    }

    private void addToVariablesCombinations(HashSet<String> variables) {
        for(String parent : variables) {
            for(String val : this.table.get(parent).keySet()) {
                Iterator<Integer> iter = this.table.get(parent).get(val).iterator();
                while(iter.hasNext()) {
                    this.variablesCombinations[iter.next()] += val + ",";
                }
            }
        }
    }

    /**
     * Samples a value from this variable, unconditioned to any probabilistic parent.
     * Only deterministic parents apply.
     * @return A value from the unconditional distribution.
     */
    public String unconditionalSampling(HashMap<String, String> lastStart) {
        try {
            int[] indices = new int[this.uniqueShadowvalues.size()];
            double[] localProbs = new double[this.uniqueShadowvalues.size()];
            Object[] keySetArray = this.uniqueShadowvalues.toArray();
            for(int i = 0; i < keySetArray.length; i++) {
                indices[i] = i;
                localProbs[i] = this.unconditionalProbabilities.get((Shadowvalue)keySetArray[i]);
            }

            EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
            Shadowvalue val = (Shadowvalue)keySetArray[localDist.sample()];
            if(!val.toString().equals("null")) {
                return val.getValue();
            }
        } catch(MathArithmeticException mae) {
            System.out.println("Variable: " + this.name + " value: " + lastStart.get(this.name));
            System.out.println("probabilistic parents: ");
            for(String parent : this.prob_parents) {
                System.out.println("\t" + parent + " value: " + lastStart.get(parent));
            }
            throw mae;
        }
        return null;
    }

    /**
     * Samples a new value for this variable, based on conditions.
     * @param lastStart Last values from the Dependency Network.
     * @return A new value for this variable.
     */
    public String conditionalSampling(HashMap<String, String> lastStart) {
        // if any of the variables that the current variable depends is null, skip sampling and set it to null too
        if(!this.passBlockingTest(lastStart)) {
            return null;
        }

        // tries to sample a value given probabilistic parent values.
        // if probabilistic parents are absents from current sample, samples an unconditional value
        try {
            int[] indices = this.getArrayOfIndices(
                    this.table, this.probabilities.size(), lastStart, null, false, prob_parents
            );

            double[] localProbs = new double [indices.length];
            for(int i = 0; i < localProbs.length; i++) {
                localProbs[i] = probabilities.get(indices[i]);
            }
            // samples values based on probabilities
            try {
                EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
                Shadowvalue val = this.values.get(localDist.sample());
                if(!val.toString().equals("null")) {
                    return val.getValue();
                }
            } catch(MathArithmeticException mae) {
                System.out.println("Variable: " + this.name + " value: " + lastStart.get(this.name));
                System.out.println("probabilistic parents: ");
                for(String parent : this.prob_parents) {
                    System.out.println("\t" + parent + " value: " + lastStart.get(parent));
                }
                throw mae;
            }
        } catch(CombinationNotPresentException cnp) {  // probabilistic parent combination of values not present
            return unconditionalSampling(lastStart);
        }
        return null;
    }

    /**
     * Given values for parent variables of the current variable, gets the indices
     * on the table that correspond to those values.
     * Already treats continuous variables (that may not have an exact value in the table entry).
     * @param conditions A HashMap where each entry is a parent name, and each value its value in
     *                   The Gibbs sampling process.
     * @param variableValue Value of the current variable.
     * @param useVariableValue Whether to take into account the value of the current variable or not.
     *                      If true, this method will search for the table index that matches exactly the
     *                      parent values PLUS this variable value. If false, will returh a set of indices
     *                      where parent values match.
     * @return An array of indices
     */
    protected int[] getArrayOfIndices(
            HashMap<String, HashMap<String, ArrayList<Integer>>> table, int n_values,
            HashMap<String, String> conditions, String variableValue, boolean useVariableValue,
            HashSet<String> allParents) throws CombinationNotPresentException {

        HashSet<Integer> intersection = this.getSetOfIndices(table, n_values, conditions, variableValue, useVariableValue, allParents);

        int[] intIndices = new int [intersection.size()];
        Object[] values = intersection.toArray();
        for(int i = 0; i < values.length; i++) {
            intIndices[i] = (int)values[i];
        }
        return intIndices;
    }

    /**
     * Returns the indices on this variable table where variableName is NOT null.
     * @param variableName Variable queried
     * @return All the indices in this.table where variableNmae is not null.
     */
    protected ArrayList<Integer> notNullLoc(String variableName) {
        ArrayList<Integer> localIndices = new ArrayList<>();
        HashMap<String, ArrayList<Integer>> valuesDict = this.table.get(variableName);
        for(String val : valuesDict.keySet()) {
            if(!String.valueOf(val).equals("null")) {
                localIndices.addAll(valuesDict.get(val));
            }
        }
        return localIndices;
    }

    /**
     * Casts to string a combination of variables and variable values.
     * @return
     */
    private static String conditionsToString(HashMap<String, String> conditions) {
        StringBuilder concatenated = new StringBuilder("");
        if(conditions != null) {
            for(String key : conditions.keySet()) {
                concatenated.append(String.format("%s=%s ", key, conditions.get(key)));
            }
        }
        return concatenated.toString();
    }

    /**
     * Given values for parent variables of the current variable, gets the indices
     * on the table that correspond to those values.
     * Already treats continuous variables (that may not have an exact value in the table entry).
     * @param conditions A HashMap where each entry is a parent name, and each value its value in
     *                   The Gibbs sampling process.
     * @param variableValue Value of the current variable.
     * @param locOnVariable Whether to take into account the value of the current variable or not.
     *                      If true, this method will search for the table index that matches exactly the
     *                      parent values PLUS this variable value. If false, will returh a set of indices
     *                      where parent values match.
     * @return A set of indices
     * @throws CombinationNotPresentException If the combination of values is not present in the probability table.
     */
    protected HashSet<Integer> getSetOfIndices(
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            int n_values,
            HashMap<String, String> conditions,
            String variableValue, boolean locOnVariable,
            HashSet<String> allParents
    ) throws CombinationNotPresentException {

        if(locOnVariable && variableValue == null) {
            throw new CombinationNotPresentException(
                    "combination of values not present in probability table: " + AbstractVariable.conditionsToString(conditions)
            );
        }

        HashSet<Integer> intersection = new HashSet<>();
        HashSet<Integer> allNumbers = new HashSet<>();
        // adds all indices to answer
        for(int i = 0; i < n_values; i++) {
            intersection.add(i);
            allNumbers.add(i);
        }

        ArrayList<Integer> localIndices;
        for (Iterator<String> it = allParents.iterator(); it.hasNext(); ) {
            String parentName = it.next();
            String parentVal = conditions.get(parentName);
            if(parentVal == null) {
                throw new CombinationNotPresentException(
                        "combination of values not present in probability table: " + AbstractVariable.conditionsToString(conditions)
                );
            }

            if (this.isParentContinuous.get(parentName)) {
                if (!String.valueOf(parentVal).equals("null")) {
                    localIndices = this.notNullLoc(parentName);
                } else {
                    localIndices = table.get(parentName).get(String.valueOf(parentVal));
                }
            } else {
                localIndices = table.get(parentName).get(String.valueOf(parentVal));
            }
            if(localIndices == null) {
                throw new CombinationNotPresentException(
                        "combination of values not present in probability table: " + AbstractVariable.conditionsToString(conditions)
                );
            } else {
                intersection.retainAll(new HashSet<>(localIndices));
            }
        }
        // whether to find indices that match exactly this variable's value.
        // if true, returns a single index; otherwise, returns an array of indices
        if(locOnVariable) {
            if(this instanceof ContinuousVariable) {
                if(!String.valueOf(variableValue).equals("null")) {
                    localIndices = this.notNullLoc(this.getName());
                } else {
                    localIndices = table.get(this.name).get(String.valueOf(variableValue));
                }
            } else {
                localIndices = table.get(this.name).get(String.valueOf(variableValue));
            }
            intersection.retainAll(new HashSet<>(localIndices));
        }
        if(intersection.isEmpty()) {
            throw new CombinationNotPresentException(
                    "combination of values not present in probability table: " + AbstractVariable.conditionsToString(conditions)
            );
        }

        return intersection;
    }

    /**
     * Just clears the parent_names, values, probabilities and table properties.
     * Updates the parent set of this variable, based on the fittest individuals from a generation.
     * @param mutableParents New parents of this variable
     * @param fittest Fittest individuals from the lattest generation
     * @throws Exception Exception thrower for children classes
     */
    public void updateStructure(AbstractVariable[] mutableParents, HashMap<String, ArrayList<String>> fittest) throws Exception {
        this.lastgen_prob_parents = (HashSet<String>)this.prob_parents.clone();
        for(String mutable_parent : this.prob_parents) {
            this.isParentContinuous.remove(mutable_parent);
        }

        this.prob_parents.clear();
        this.mutableBlocking.clear();

        // if variable is not among fixed parents
        for(AbstractVariable par : mutableParents) {
            // adds algorithm of parent variable to mutable blocking variables
            this.mutableBlocking.put(par.getAlgorithmName(), new ArrayList<String>(){{add("true");}});

            this.prob_parents.add(par.getName());
            this.isParentContinuous.put(par.getName(), par instanceof ContinuousVariable);
        }

//        this.oldTable = (HashMap<String, HashMap<String, ArrayList<Integer>>>) this.table.clone();
//        this.oldValues = (ArrayList<Shadowvalue>) this.values.clone();
//        this.oldProbabilities = (ArrayList<Double>) this.probabilities.clone();

        // only updates structure if probabilistic parents changed
        if(!this.sameParentsFromLastGeneration()) {
            this.setValues(fittest);

            this.probabilities.clear();
            this.table.clear();
            this.values.clear();

            // continuous variable code
            HashMap<String, HashSet<Shadowvalue>> valuesCombinations = Combinator.getUniqueValuesFromVariables(mutableParents);
            // adds unique values of this variable
            valuesCombinations.put(this.getName(), this.getUniqueShadowvalues());

            this.updateTable(valuesCombinations);
            this.initVariablesCombinations();
            this.setValues(fittest);
        }
    }

    protected static int countDiscrete(AbstractVariable[] parents) {
        int count = 0;

        for(AbstractVariable parent : parents) {
            if(parent instanceof DiscreteVariable) {
                count += 1;
            }
        }
        return count;
    }

    /**
     * Gets only discrete variables from the list provided.
     * @param variables An array of variables to be proceeded.
     * @return The same array, with only discrete variables now.
     */
    protected static AbstractVariable[] getOnlyDiscrete(AbstractVariable[] variables) {
        int countDiscreteParents = AbstractVariable.countDiscrete(variables);
        AbstractVariable[] discreteParents = null;
        if(countDiscreteParents != variables.length) {
            discreteParents = new AbstractVariable [countDiscreteParents];
            int counter = 0;
            for(int i = 0; i < variables.length; i++) {
                if(variables[i] instanceof DiscreteVariable) {
                    discreteParents[counter] = variables[i];
                    counter += 1;
                }
            }
        } else {
            discreteParents = variables;
        }
        return discreteParents;
    }

    protected int countContinuousParents(AbstractVariable[] parents) {
        return parents.length - countDiscrete(parents);
    }

    protected boolean allContinuousParents(AbstractVariable[] parents) {
        return countDiscrete(parents) == 0;
    }

    protected boolean allDiscreteParents(AbstractVariable[] parents) {
        return countDiscrete(parents) != parents.length;
    }

    /**
     * Updates the table attribute of this variable. Automatically treats values for continuous variables.
     *
     * @param dictUnique A dictionary, where each entry is the set of unique values
     *               of a variable (which its name is the key).
     */
    protected void updateTable(HashMap<String, HashSet<Shadowvalue>> dictUnique) {
        Object[] keys = dictUnique.keySet().toArray();

        int n_combinations = 1;
        int[] n_unique = new int [dictUnique.size()];
        int[] repeat_every = new int [dictUnique.size()];

        this.table = new HashMap<>(dictUnique.size());
        for(int i = dictUnique.size() - 1; i >= 0; i--) {
            HashMap<String, ArrayList<Integer>> variableDict = new HashMap<>(dictUnique.get((String)keys[i]).size());

            for(Iterator<Shadowvalue> it = dictUnique.get((String)keys[i]).iterator(); it.hasNext(); ) {
                String val = it.next().toString();
                variableDict.put(val, new ArrayList<>());
            }
            table.put((String)keys[i], variableDict);

            // computes the period of appearance of variable's dictUnique
            n_unique[i] = variableDict.size();
            n_combinations *= n_unique[i];
            if(i == (dictUnique.size() - 1)) {
                repeat_every[i] = 1;
            } else {
                repeat_every[i] = repeat_every[i + 1] * n_unique[i + 1];
            }
        }

        for(int i = 0; i < keys.length; i++) {
            HashMap<String, ArrayList<Integer>> local = table.get((String)keys[i]);
            Object[] subKeys = local.keySet().toArray();

            // updates
            int counter = 0;
            while(counter < n_combinations) {
                for(int j = 0; j < subKeys.length; j++) {
                    for(int k = 0; k < repeat_every[i]; k++) {
                        local.get((String)subKeys[j]).add(counter);
                        counter += 1;
                    }
                }
            }
            // re-inserts
            table.put((String)keys[i], local);
        }
    }

    /**
     * Makes carousel spin
     * @param carousel
     * @param sizes
     * @return
     */
    private int []advanceCarousel(int[] carousel, int[] sizes) {
        for (int j = sizes.length - 1; j >= 0; j--) {
            carousel[j] += 1;
            if (carousel[j] >= sizes[j]) {
                carousel[j] = 0;
            } else {
                break;
            }
        }
        return carousel;
    }

    /**
     * a modified way of performing learning rate.
     */
    private void addResidualProbabilities() throws Exception {
        // TODO reactivate later!
//        String[] fixedNames = new String [fixed_parents.size() + 1];
//        int fixedCounter = 0;
//        for(String fixPar : fixed_parents) {
//            fixedNames[fixedCounter] = fixPar;
//            fixedCounter += 1;
//        }
//        fixedNames[fixed_parents.size()] = this.getName();
//
//        HashMap<String, Object[]> variablesArrayOfValues = new HashMap<>();
//        int[] sizes = new int [fixedNames.length];
//        int[] carousel = new int [fixedNames.length];
//
//        int n_combinations = 1;
//        for(int i = 0; i < fixedNames.length; i++) {
//            Object[] uniqueVals = this.table.get(fixedNames[i]).keySet().toArray();
//            variablesArrayOfValues.put(fixedNames[i], uniqueVals);
//            sizes[i] = uniqueVals.length;
//            carousel[i] = 0;
//            n_combinations *= sizes[i];
//        }
//        int counter = 0;
//        while(counter < n_combinations) {
//            HashMap<String, String> cond = new HashMap<>();
//
//            for (int i = 0; i < fixedNames.length; i++) {
//                String name = fixedNames[i];
//                cond.put(name, (String) variablesArrayOfValues.get(name)[carousel[i]]);
//            }
//            carousel = advanceCarousel(carousel, sizes);
//            counter += 1;
//
//            HashSet<Integer> newIndices = this.getSetOfIndices(this.table, this.probabilities.size(), cond, cond.get(this.getName()), true, new HashSet<>(this.fixed_parents));
//            HashSet<Integer> oldIndices = this.getSetOfIndices(this.oldTable, this.oldProbabilities.size(), cond, cond.get(this.getName()), true, new HashSet<>(this.fixed_parents));
//
//            double meanOldProbability = 0;
//            for(int index : oldIndices) {
//                meanOldProbability += this.oldProbabilities.get(index);
//            }
//            // uses equation of PBIL
//            if(oldIndices.size() > 0) {
//                meanOldProbability = (1 - this.learningRate) * (meanOldProbability / oldIndices.size());
//            }
//            for(int index : newIndices) {
//                this.probabilities.set(index, meanOldProbability);
//            }
//        }
//
//        throw new Exception("now take into account previous gen mutable parents!!!");
    }

    /**
     * Update probabilities of this Variable based on the fittest population of a generation.
     *
     * @param fittestValues Fittest individuals from the last population
     * @throws Exception If any exception occurs
     */
    public void updateProbabilities(HashMap<String, ArrayList<String>> fittestValues, Individual[] fittest, float learningRate, int n_generations) throws Exception {
        int n_continuous = 0;

        int n_fittest = fittestValues.get(this.getName()).size();

        // TODO laplace correction might be needed next. come here and set value to 1 if probability distributions sum up to zero
        ArrayList<Double> relevantFittestCounts = new ArrayList<>(Collections.nCopies(this.values.size(), 1.0));  // set 1 for laplace correction; 0 otherwise
        double relevantFittestSize = this.values.size();  // TODO set to this.values.size() for laplace correction; 0 otherwise

        ArrayList<Double> oldProbabilities = sameParentsFromLastGeneration()? (ArrayList<Double>)this.probabilities.clone() : null;

        this.probabilities = new ArrayList<>(Collections.nCopies(this.values.size(), 0.0));
        this.setUnconditionalProbabilities(fittestValues);

        HashMap<String, Integer> ddd = new HashMap<>();

        for (Iterator<String> it = prob_parents.iterator(); it.hasNext(); ) {
            String par = it.next();
            if(this.isParentContinuous.get(par)) {
                ddd.put(par, n_continuous);
                n_continuous += 1;
            }
        }
        if(this instanceof ContinuousVariable) {
            ddd.put(this.getName(), n_continuous);
            n_continuous += 1;
        }

        // keeps track of double values for each one of the
        // combination of values in the fittest population
        double[][][] dda = new double[n_continuous][this.values.size()][n_fittest];
        int[][] ddc = new int[n_continuous][this.values.size()];
        for(int i = 0; i < n_continuous; i++) {
            Arrays.fill(ddc[i], 0);;
        }

        // for each individual in the fittest population:
        // locates the index that represents its combination of values
        // adds 1 to the counter of occurrences
        for(int i = 0; i < n_fittest; i++) {
            try {
                // TODo if self is null, don't even search it

                int[] indices = this.getArrayOfIndices(
                        this.table,
                        this.values.size(),
                        fittest[i].getCharacteristics(),
                        fittest[i].getCharacteristics().get(this.getName()),
                        true,
                        prob_parents
                );

                // this individual shall return only one index.
                // if returns more than one, then there's an error in the code
                if(indices.length != 1) {
                    throw new Exception("unexpected behaviour!");
                }
                // registers this individual occurrence
                relevantFittestCounts.set(indices[0], relevantFittestCounts.get(indices[0]) + 1);
                relevantFittestSize += 1;  // registers that the combination of values exist

                // annotates float values, if any
                for(String var : ddd.keySet()) {
                    String val = fittestValues.get(var).get(i);
                    if(val != null) {
                        // dda[index of continuous variable][index of combination][fittest counter]
                        dda[ddd.get(var)][indices[0]][ddc[ddd.get(var)][indices[0]]
                                ] = Double.parseDouble(val);
                        ddc[ddd.get(var)][indices[0]] += 1;
                    }
                }
            } catch (CombinationNotPresentException eae) {
                // nothing happens
            }
        }

        boolean sameParents = this.sameParentsFromLastGeneration();
        HashMap<HashMap<String, String>, HashSet<Integer>> parentValIndices = this.iterateOverParentValues();

        // TODO what if variable does not have parents?
        // TODO has to iterate over all values. but then indices (below) should be adapted too

        // now has to iterate over parent values, because probabilities
        // are normalized based on exact parent values
        for(HashMap<String, String> parentCombination : parentValIndices.keySet()) {
//            // updates normal distributions
            if(this instanceof ContinuousVariable) {
                throw new Exception("not implemented yet!");
//                ((ContinuousVariable)this).updateNormalDistributions(parentValIndices.get(parentCombination), dda, ddc, ddd, n_generations);
            }

            // iterates over child values
            for(Shadowvalue selfValue : this.uniqueShadowvalues) {
                HashSet<Integer> indices = this.getSetOfIndices(
                        this.table, this.values.size(), parentCombination,
                        selfValue.toString(), true, prob_parents
                );

                if(indices.size() != 1) {
                    throw new Exception("should have return only one index!");
                }

                Integer index = (Integer)indices.toArray()[0];

                double old_value = sameParents ? (1 - learningRate) * oldProbabilities.get(index) : 0;
                double new_value = (sameParents ? learningRate : 1) * relevantFittestCounts.get(index) / relevantFittestSize;

                this.probabilities.set(index, old_value + new_value);

            }
        }

        // normalizes everything - if needed
        this.normalizeProbabilities(parentValIndices);

        for(int i = 0; i < this.probabilities.size(); i++) {
            if(Double.isNaN(this.probabilities.get(i))) {
                throw new Exception("unexpected behavior!");
//                this.probabilities.set(i, 0.0);
            }
        }
    }

    private void setUnconditionalProbabilities(HashMap<String, ArrayList<String>> fittestValues) {
        this.unconditionalProbabilities = new HashMap<>();

        ArrayList<String> assumedValues = fittestValues.get(this.getName());
        long sum = 0;
        for(Shadowvalue unique : this.uniqueShadowvalues) {
            int count = 0;
            String uniqueStr = unique.toString();
            for(int i = 0; i < assumedValues.size(); i++) {
                if(String.valueOf(assumedValues.get(i)).equals(uniqueStr)) {
                    count += 1;
                }
            }
            sum += count;
            this.unconditionalProbabilities.put(unique, (double)count);
        }

        // normalizes
        for(Shadowvalue unique : this.uniqueShadowvalues) {
            this.unconditionalProbabilities.put(unique, this.unconditionalProbabilities.get(unique) / sum);
        }
    }
    private void setUnconditionalProbabilities() {
        this.unconditionalProbabilities = new HashMap<>();
        for(Shadowvalue unique : this.uniqueShadowvalues) {
            this.unconditionalProbabilities.put(unique, 1./this.uniqueShadowvalues.size());
        }
    }

    /**
     * Checks whether probabilistic parents have changed from last generation to current.
     */
    private boolean sameParentsFromLastGeneration() {
        return this.lastgen_prob_parents.containsAll(this.prob_parents) &&
                this.prob_parents.containsAll(this.lastgen_prob_parents);
    }

    /**
     * Iterates over all probabilistic parent values for current generation.
     *
     * Generates a dictionary where the key is a HashMap with a value assignment for each probabilistic parent variable,
     * and the value is an ArrayList to where those values appear in the probabilistic table.
     *
     * What if this variable does not have probabilistic parents? Then it will return all indices in the table.
     *
     * @return Returns a dictionary where the key is the combination of parent values (excluded child, i.e this variable
     * values) and the dictionary value for that key is the set of indices where that combination of parent values happen.
     */
    private HashMap<HashMap<String, String>, HashSet<Integer>> iterateOverParentValues() {
        HashMap<HashMap<String, String>, HashSet<Integer>> results = new HashMap<>();
        if(this.prob_parents.size() == 0) {
            HashSet<Integer> allIndices = new HashSet<>();
            for(int i = 0; i < this.values.size(); i++) {
                allIndices.add(i);
            }
            results.put(new HashMap<String, String>(), allIndices);
            return results;
        }

        HashMap<String, Object[]> parentsArrayOfValues = new HashMap<>();
        int[] sizes = new int [prob_parents.size() + 1];
        int[] carousel = new int [prob_parents.size() + 1];

        Object[] prob_parents_array = this.prob_parents.toArray();
        int parent_combinations = 1;
        for(int i = 0; i < prob_parents_array.length; i++) {
            Object[] parentUnVal = this.table.get((String)prob_parents_array[i]).keySet().toArray();
            parentsArrayOfValues.put(
                    (String)prob_parents_array[i],
                    parentUnVal
            );
            sizes[i] = parentUnVal.length;
            parent_combinations *= sizes[i];
            carousel[i] = 0;
        }

        int counter = 0;
        while(counter < parent_combinations) {
            HashMap<String, String> cond = new HashMap<>();

            // adds a new combination of values for parents
            for (int i = 0; i < prob_parents_array.length; i++) {
                String parentName = (String) prob_parents_array[i];
                cond.put(
                        parentName,
                        (String) parentsArrayOfValues.get(parentName)[carousel[i]]
                );
            }
            carousel = advanceCarousel(carousel, sizes);
            counter += 1;

            try {
                HashSet<Integer> indices = this.getSetOfIndices(
                        this.table, this.values.size(), cond, null, false, this.prob_parents
                );
                results.put(cond, indices);
            } catch(CombinationNotPresentException cnp) {
                System.err.println("unexpected behavior!");
            }

        }
        return results;
    }

    /**
     * Call this function right after updating probabilities.
     * Will re-normalize them. Prevents probabilities from vanishing in later generations.
     */
    private void normalizeProbabilities(HashMap<HashMap<String, String>, HashSet<Integer>> parentValIndices) {
        for(HashSet<Integer> indices : parentValIndices.values()) {
            double sum = 0;

            for(int index : indices) {
                sum += this.probabilities.get(index);
            }

            for(int index : indices) {
                this.probabilities.set(index, this.probabilities.get(index) / sum);
            }
        }
    }


    public HashMap<String, Double> getTablePrettyPrint() {
        ArrayList<String> lines = new ArrayList<>(this.values.size());
        for(int i = 0; i < this.values.size(); i++) {
            lines.add("");
        }

        Object[][] toProcess = {prob_parents.toArray(), new String[]{name}};

        for(Object[] current : toProcess) {
            for(Object variableName : current) {
                HashMap<String, ArrayList<Integer>> variableValues = this.table.get((String)variableName);
                for(String variableVal : variableValues.keySet()) {
                    for(Integer index : variableValues.get(variableVal)) {
                        String oldLine = lines.get(index);
                        String candidate = String.format(Locale.US, "%s=%s", (String)variableName, variableVal);
                        lines.set(index, oldLine + (oldLine.length() > 0? "," : "") + candidate);
                    }
                }
            }
        }
        HashMap<String, Double> pairwise = new HashMap<>(lines.size());
        for(int i = 0; i < lines.size(); i++) {
            pairwise.put(lines.get(i), probabilities.get(i));
        }

        return pairwise;
    }

    /**
     *
     * @return An array with unique names of parents.
     */
    public HashSet<String> getProbabilisticParentsNames() {
        return this.prob_parents;
    }

    public String getName() {
        return this.name;
    }

    public int getParentCount() {
        return this.prob_parents.size();
    }

    public int getMaxParents() {
        return max_parents;
    }

    /**
     * Returns all variables that must be sampled before this variable, either fixed or mutable.
     */
    public HashSet<String> getAllBlocking() {
        HashSet<String> set = new HashSet<>();
        set.addAll(this.fixedBlocking.keySet());
        set.addAll(this.mutableBlocking.keySet());
        return set;
    }

    public HashSet<String> getProbabilisticParents() {
        return this.prob_parents;
    }

    public HashSet<String> getFixedCannotLink() {
        return this.fixedCannotLink;
    }

    public HashSet<Shadowvalue> getUniqueShadowvalues() {
        return uniqueShadowvalues;
    }

    /**
     * Updates values array.
     *
     * @param fittest dictionary where the keys are the variable names and the value (of the dict) an array of values
     *                assumed by that variable in the fittest population.
     * @throws Exception
     */
    public void setValues(HashMap<String, ArrayList<String>> fittest) throws Exception {
        HashMap<String, ArrayList<Integer>> thisVariableIndices = this.table.get(this.getName());
        Object[] vals = thisVariableIndices.keySet().toArray();

        for(Object val : vals) {
            ArrayList<Integer> indices = thisVariableIndices.get((String)val);
            Shadowvalue thissv = null;
            for (Iterator<Shadowvalue> it = this.uniqueShadowvalues.iterator(); it.hasNext(); ) {
                Shadowvalue sv = it.next();
                if(sv.toString().equals(String.valueOf(val))) {
                    thissv = sv;
                    break;
                }
            }

            int max_index = Collections.max(indices);
            for(int i = this.values.size(); i <= max_index; i++) {
                this.values.add(null);
            }
            for(Integer index : indices) {
                this.values.set(index, thissv);
            }
        }
    }

    public static AbstractVariable fromPath(
            String path, JSONObject initialBlocking, int max_parents, MersenneTwister mt) throws Exception {
        BufferedReader csvReader = new BufferedReader(new FileReader(path));

        String variableName = path.substring(
                path.lastIndexOf(File.separator) + 1,
                path.lastIndexOf(".")
        );

        String row = csvReader.readLine();
        String[] header = row.split(",(?![^(]*\\))");

        HashMap<String, Boolean> isContinuous = new HashMap<>();

        int n_variables_table = 1;
        HashSet<String> parents_names = new HashSet<>();

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
                        isContinuous.getOrDefault(header[k], false) ||
                                (this_data[k].contains("loc") && this_data[k].contains("scale")) ||  // univariate normal distribution
                                this_data[k].contains("means")  // multivariate normal distribution
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
            return new ContinuousVariable(
                variableName, (JSONObject)initialBlocking.get(variableName), isContinuous, table,
                values, probabilities, mt, max_parents
            );
        }
        return new DiscreteVariable(
            variableName, (JSONObject)initialBlocking.get(variableName), isContinuous, table,
            values, probabilities, mt, max_parents
        );
    }
}


