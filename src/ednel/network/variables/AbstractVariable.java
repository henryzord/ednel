package ednel.network.variables;

import ednel.eda.individual.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.*;

public abstract class AbstractVariable {
    protected String name;

    protected ArrayList<String> mutable_parents;
    protected ArrayList<String> fixed_parents;
    protected HashMap<String, Boolean> isParentContinuous;
    protected int max_parents;

    /**
     * Should not contain any null values.
     * If there is a null value, cast it to String:
     * String.valueOf(null)
     */
    protected HashSet<String> uniqueValues;

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
    protected HashSet<Shadowvalue> uniqueShadowvalues;

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

    protected ArrayList<Shadowvalue> oldValues;
    protected ArrayList<Double> oldProbabilities;

    /**
     * Should not contain null values. If a null value is present, then it is "null" (i.e. a string)
     */
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> oldTable;

    protected String[] variablesCombinations;

    protected MersenneTwister mt;

    protected double learningRate;
    protected int n_generations;

    protected static String[] singleParentVariables = {"J48_reducedErrorPruning", "J48_unpruned", "PART_reducedErrorPruning", "PART_unpruned"};

    public AbstractVariable(
            String name, ArrayList<String> fixed_parents, HashMap<String, Boolean> isParentContinuous,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            HashSet<String> uniqueValues, HashSet<Shadowvalue> uniqueshadowValues,
            ArrayList<Shadowvalue> values, ArrayList<Double> probabilities,
            MersenneTwister mt, double learningRate, int n_generations, int max_parents) throws Exception {

        this.name = name;
        this.fixed_parents = fixed_parents;
        this.mutable_parents = new ArrayList<>();
        this.isParentContinuous = isParentContinuous;

        this.uniqueValues = uniqueValues;
        this.uniqueShadowvalues = uniqueshadowValues;

        this.values = values;
        this.probabilities = probabilities;

        this.table = table;

        this.initVariablesCombinations();

        this.mt = mt;
        this.learningRate = learningRate;
        this.n_generations = n_generations;

        this.max_parents = max_parents;
        for(String otherName : singleParentVariables) {
            if(otherName.equals(name)) {
                this.max_parents = 1;
                break;
            }
        }
    }

    /**
     * For each entry in the probabilities attribute in this object, writes the combination of fixed parents/mutable
     * parents/this variable values (in this order) that is associated with that probability.
     */
    protected void initVariablesCombinations() {
        int n_combinations = 1;
        for(String val : this.table.keySet()) {
            n_combinations *= this.table.get(val).size();
        }

        this.variablesCombinations = new String [n_combinations];
        Arrays.fill(this.variablesCombinations, "");
        this.addToVariablesCombinations(this.fixed_parents);
        this.addToVariablesCombinations(this.mutable_parents);
        this.addToVariablesCombinations(new ArrayList<String>(){{
            add(name);
        }});
    }

    private void addToVariablesCombinations(ArrayList<String> variables) {
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
     * Samples a new value for this variable, based on conditions.
     * @param lastStart Last values from the Dependency Network.
     * @return A new value for this variable.
     */
    public String conditionalSampling(HashMap<String, String> lastStart) {
        HashSet<String> allParents = new HashSet<>(this.fixed_parents);
        allParents.addAll(this.mutable_parents);

        int[] indices = this.getArrayOfIndices(this.table, this.probabilities.size(), lastStart, null, false, allParents);

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
            System.out.println("Variable: " + this.getName() + " value: " + lastStart.get(this.getName()));  // TODO remove!
            System.out.println("fixed parents: " );
            for(String parent : this.fixed_parents) {
                System.out.println("\t" + parent + " value: " + lastStart.get(parent));
            }
            System.out.println("mutable parents: ");
            for(String parent : this.mutable_parents) {
                System.out.println("\t" + parent + " value: " + lastStart.get(parent));
            }
            throw mae;
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
            HashSet<String> allParents) {

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
     */
    protected HashSet<Integer> getSetOfIndices(
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            int n_values,
            HashMap<String, String> conditions,
            String variableValue, boolean locOnVariable,
            HashSet<String> allParents
    ) {

        HashSet<Integer> intersection = new HashSet<>();
        HashSet<Integer> allNumbers = new HashSet<>();
        // adds all indices to answer
        for(int i = 0; i < n_values; i++) {
            intersection.add(i);
            allNumbers.add(i);
        }

        ArrayList<Integer> localIndices = null;
        for (Iterator<String> it = allParents.iterator(); it.hasNext(); ) {
            String parentName = it.next();
            String parentVal = conditions.get(parentName);

            if (this.isParentContinuous.get(parentName)) {
                if (!String.valueOf(parentVal).equals("null")) {
                    localIndices = this.notNullLoc(parentName);
                } else {
                    localIndices = table.get(parentName).get(String.valueOf(parentVal));
                }
            } else {
                localIndices = table.get(parentName).get(String.valueOf(parentVal));
            }
//            if(localIndices != null) {
            try {
                intersection.retainAll(new HashSet<>(localIndices));
            } catch (NullPointerException npe) {
                System.out.println("Variable: " + this.getName());  // TODO remove!
                System.out.println("fixed parents: ");
                for (String parent : this.fixed_parents) {
                    System.out.println("\t" + parent);
                }
                System.out.println("mutable parents: ");
                for (String parent : this.mutable_parents) {
                    System.out.println("\t" + parent);
                }
                throw npe;
//            }
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
        return intersection;
    }

    /**
     * Just clears the parent_names, values, probabilities and table properties.
     * Updates the parent set of this variable, based on the fittest individuals from a generation.
     * @param mutableParents New parents of this variable
     * @param fittest Fittest individuals from the lattest generation
     * @throws Exception Exception thrower for children classes
     */
    public void updateStructure(AbstractVariable[] mutableParents, AbstractVariable[] fixedParents,
                                HashMap<String, ArrayList<String>> fittest) throws Exception {

        for(String mutable_parent : this.mutable_parents) {
            this.isParentContinuous.remove(mutable_parent);
        }
        this.mutable_parents.clear();

        // if variable is not among fixed parents
        for(AbstractVariable par : mutableParents) {
            if(this.fixed_parents.indexOf(par.getName()) == -1) {
                this.mutable_parents.add(par.getName());
                this.isParentContinuous.put(par.getName(), par instanceof ContinuousVariable);
            }
        }

        this.oldTable = (HashMap<String, HashMap<String, ArrayList<Integer>>>) this.table.clone();
        this.oldValues = (ArrayList<Shadowvalue>) this.values.clone();
        this.oldProbabilities = (ArrayList<Double>) this.probabilities.clone();

        this.probabilities.clear();
        this.table.clear();
        this.values.clear();

        // continuous variable code
        HashMap<String, HashSet<String>> valuesCombinations = Combinator.getUniqueValuesFromVariables(mutableParents);
        valuesCombinations.putAll(Combinator.getUniqueValuesFromVariables(fixedParents));
        // adds unique values of this variable
        valuesCombinations.put(this.getName(), this.getUniqueValues());

        this.updateTable(valuesCombinations);
        this.initVariablesCombinations();
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
    protected void updateTable(HashMap<String, HashSet<String>> dictUnique) throws Exception {
        Object[] keys = dictUnique.keySet().toArray();

        int n_combinations = 1;
        int[] n_unique = new int [dictUnique.size()];
        int[] repeat_every = new int [dictUnique.size()];

        this.table = new HashMap<>(dictUnique.size());
        for(int i = dictUnique.size() - 1; i >= 0; i--) {
            HashMap<String, ArrayList<Integer>> variableDict = new HashMap<>(dictUnique.get((String)keys[i]).size());

            for(Iterator<String> it = dictUnique.get((String)keys[i]).iterator(); it.hasNext(); ) {
                String val = it.next();
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
    private void addResidualProbabilities() {
        String[] fixedNames = new String [fixed_parents.size() + 1];
        for(int i = 0; i < fixed_parents.size(); i++) {
            fixedNames[i] = fixed_parents.get(i);
        }
        fixedNames[fixed_parents.size()] = this.getName();

        HashMap<String, Object[]> variablesArrayOfValues = new HashMap<>();
        int[] sizes = new int [fixedNames.length];
        int[] carousel = new int [fixedNames.length];

        int n_combinations = 1;
        for(int i = 0; i < fixedNames.length; i++) {
            Object[] uniqueVals = this.table.get(fixedNames[i]).keySet().toArray();
            variablesArrayOfValues.put(fixedNames[i], uniqueVals);
            sizes[i] = uniqueVals.length;
            carousel[i] = 0;
            n_combinations *= sizes[i];
        }
        int counter = 0;
        while(counter < n_combinations) {
            HashMap<String, String> cond = new HashMap<>();

            for (int i = 0; i < fixedNames.length; i++) {
                String name = fixedNames[i];
                cond.put(name, (String) variablesArrayOfValues.get(name)[carousel[i]]);
            }
            carousel = advanceCarousel(carousel, sizes);
            counter += 1;

            HashSet<Integer> newIndices = this.getSetOfIndices(this.table, this.probabilities.size(), cond, cond.get(this.getName()), true, new HashSet<>(this.fixed_parents));
            HashSet<Integer> oldIndices = this.getSetOfIndices(this.oldTable, this.oldProbabilities.size(), cond, cond.get(this.getName()), true, new HashSet<>(this.fixed_parents));

            double meanOldProbability = 0;
            for(int index : oldIndices) {
                meanOldProbability += this.oldProbabilities.get(index);
            }
            // uses equation of PBIL
            if(oldIndices.size() > 0) {
                meanOldProbability = (1 - this.learningRate) * (meanOldProbability / oldIndices.size());
            }
            for(int index : newIndices) {
                this.probabilities.set(index, meanOldProbability);
            }
        }
    }

    /**
     * Updates shadow values for discrete variables
     * @return
     */
    private int updateShadowValues() {
        HashMap<String, ArrayList<Integer>> thisVariableIndices = this.table.get(this.getName());
        Object[] vals = thisVariableIndices.keySet().toArray();

        int n_combinations = 0;
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
            n_combinations = Math.max(max_index, n_combinations);
            for(int i = this.values.size(); i <= max_index; i++) {
                this.values.add(null);
            }
            for(Integer index : indices) {
                this.values.set(index, thissv);
            }
        }
        n_combinations += 1;
        return n_combinations;
    }


    /**
     * Update probabilities of this Variable based on the fittest population of a generation.
     *
     * @param fittestValues Fittest individuals from the last population
     * @throws Exception If any exception occurs
     */
    public void updateProbabilities(HashMap<String, ArrayList<String>> fittestValues, Individual[] fittest) throws Exception {
        int n_combinations = updateShadowValues();
        double countsSum = 0;
        int n_continuous = 0;
        int n_fittest = fittestValues.get(this.getName()).size();

        ArrayList<Double> counts = new ArrayList<>(Collections.nCopies(n_combinations, 0.0));  // set 1 for laplace correction; 0 otherwise
        this.probabilities = new ArrayList<>(Collections.nCopies(n_combinations, 0.0));

        HashMap<String, Integer> ddd = new HashMap<>();
        HashSet<String> allParents = new HashSet<>();
        allParents.addAll(this.mutable_parents);
        allParents.addAll(this.fixed_parents);

        for (Iterator<String> it = allParents.iterator(); it.hasNext(); ) {
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
        double[][][] dda = new double[n_continuous][n_combinations][n_fittest];
        int[][] ddc = new int[n_continuous][n_combinations];
        for(int i = 0; i < n_continuous; i++) {
            Arrays.fill(ddc[i], 0);;
        }

        // for each individual in the fittest population:
        // locates the index that represents its combination of values
        // adds 1 to the counter of occurrences
        for(int i = 0; i < n_fittest; i++) {
            int[] indices = this.getArrayOfIndices(
                    this.table,
                    this.probabilities.size(),
                    fittest[i].getCharacteristics(),
                    fittest[i].getCharacteristics().get(this.getName()),
                    true,
                    allParents
            );

            // this individual shall return only one index. if returns more than one, then there's an error
            // in the code
            if(indices.length > 1) {
                throw new Exception("unexpected behaviour!");
            }
            // registers this individual occurrence
            counts.set(indices[0], counts.get(indices[0]) + 1);
            countsSum += 1;

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
        }

        addResidualProbabilities();  // TODO change formula here later!!!

        // updates probabilities
        // also updates normal distributions, if this variable is continuous

        // indices now has the set of indices that parents match cond, but child variable has all its values
        HashMap<HashMap<String, String>, HashSet<Integer>> parentValIndices = this.iterateOverParentValues();
        for(HashSet<Integer> indices : parentValIndices.values()) {
            // updates normal distributions
            if(this instanceof ContinuousVariable) {
                ((ContinuousVariable)this).updateNormalDistributions(indices, dda, ddc, ddd);
            }

            for(int index : indices) {
                this.probabilities.set(
                        index,
                        this.probabilities.get(index) + this.learningRate * (counts.get(index) / countsSum)
                );
            }
        }

        normalizeProbabilities(parentValIndices);

        for(int i = 0; i < this.probabilities.size(); i++) {
            if(Double.isNaN(this.probabilities.get(i))) {
                this.probabilities.set(i, 0.0);
            }
        }
    }

    /**
     * Iterates over all parent values (either fixed or mutable).
     *
     * @return Returns a dictionary where the key is the combination of parent values (excluded child, i.e this variable
     * values) and the dictionary value for that key is the set of indices where that combination of parent values happen.
     */
    private HashMap<HashMap<String, String>, HashSet<Integer>> iterateOverParentValues() {
        HashSet<String> allParents = new HashSet<>();
        allParents.addAll(this.mutable_parents);
        allParents.addAll(this.fixed_parents);

        HashMap<String, Object[]> parentsArrayOfValues = new HashMap<>();
        int[] sizes = new int [allParents.size()];
        int[] carousel = new int [allParents.size()];

        Object[] allParentsArray = allParents.toArray();
        int parent_combinations = 1;
        for(int i = 0; i < allParentsArray.length; i++) {
            Object[] parentUnVal = this.table.get((String)allParentsArray[i]).keySet().toArray();
            parentsArrayOfValues.put(
                    (String)allParentsArray[i],
                    parentUnVal
            );
            sizes[i] = parentUnVal.length;
            parent_combinations *= sizes[i];
            carousel[i] = 0;
        }

        HashMap<HashMap<String, String>, HashSet<Integer>> results = new HashMap<>(parent_combinations);

        int counter = 0;
        while(counter < parent_combinations) {
            HashMap<String, String> cond = new HashMap<>();

            // adds a new combination of values for parents
            for (int i = 0; i < allParentsArray.length; i++) {
                String parentName = (String) allParentsArray[i];
                cond.put(
                        parentName,
                        (String) parentsArrayOfValues.get(parentName)[carousel[i]]
                );
            }
            carousel = advanceCarousel(carousel, sizes);
            counter += 1;

            HashSet<Integer> indices = this.getSetOfIndices(
                this.table, this.probabilities.size(), cond, null, false, allParents
            );
            results.put(cond, indices);
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


    public abstract void updateUniqueValues(HashMap<String, ArrayList<String>> fittest);

    public HashMap<String, Double> getTablePrettyPrint() {
        ArrayList<String> lines = new ArrayList<>(this.values.size());
        for(int i = 0; i < this.values.size(); i++) {
            lines.add("");
        }

        ArrayList<ArrayList<String>> toProcess = new ArrayList<ArrayList<String>>(){{
            add(mutable_parents);
            add(fixed_parents);
            add(new ArrayList<String>(){{
                add(name);
            }});
        }};

        for(ArrayList<String> current : toProcess) {
            for(String variableName : current) {
                HashMap<String, ArrayList<Integer>> variableValues = this.table.get(variableName);
                for(String variableVal : variableValues.keySet()) {
                    for(Integer index : variableValues.get(variableVal)) {
                        String oldLine = lines.get(index);
                        String candidate = String.format(Locale.US, "%s=%s", variableName, variableVal);
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
    public ArrayList<String> getMutableParentsNames() {
        return this.mutable_parents;
    }

    public ArrayList<String> getFixedParentsNames() {
        return this.fixed_parents;
    }

    public HashSet<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public String getName() {
        return this.name;
    }

    public int getParentCount() {
        return this.mutable_parents.size() + this.fixed_parents.size();
    }

    public int getMaxParents() {
        return max_parents;
    }
}
