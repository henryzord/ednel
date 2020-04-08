package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;

public abstract class AbstractVariable {
    protected String name;
    protected ArrayList<String> parents_names;
    protected HashMap<String, Boolean> isParentContinuous;

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

    /**
     * Can contain null values.
     */
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;

    protected MersenneTwister mt;

    protected double learningRate;
    protected int n_generations;

    public AbstractVariable(
        String name, ArrayList<String> parents_names, HashMap<String, Boolean> isParentContinuous,
        HashMap<String, HashMap<String, ArrayList<Integer>>> table,
        HashSet<String> uniqueValues, HashSet<Shadowvalue> uniqueshadowValues,
        ArrayList<Shadowvalue> values, ArrayList<Double> probabilities,
        MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        this.name = name;
        this.parents_names = parents_names;
        this.isParentContinuous = isParentContinuous;

        this.uniqueValues = uniqueValues;
        this.uniqueShadowvalues = uniqueshadowValues;

        this.values = values;
        this.probabilities = probabilities;

        this.table = table;

        this.mt = mt;
        this.learningRate = learningRate;
        this.n_generations = n_generations;
    }

    /**
     * Samples a new value for this variable, based on conditions.
     * @param lastStart Last values from the Dependency Network.
     * @return A new value for this variable.
     */
    public String conditionalSampling(HashMap<String, String> lastStart) {
        int[] indices = this.getArrayOfIndices(lastStart, null, false);

        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            localProbs[i] = probabilities.get(indices[i]);
        }
        // samples values based on probabilities
        EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
        Shadowvalue val = this.values.get(localDist.sample());

        if(!val.toString().equals("null")) {
            return val.getValue();
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
    protected int[] getArrayOfIndices(HashMap<String, String> conditions, String variableValue, boolean useVariableValue) {
        HashSet<Integer> intersection = this.getSetOfIndices(conditions, variableValue, useVariableValue);

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
    private ArrayList<Integer> notNullLoc(String variableName) {
        ArrayList<Integer> localIndices = new ArrayList<>();
        HashMap<String, ArrayList<Integer>> subKeys = this.table.get(variableName);
        for(String key : subKeys.keySet()) {
            if(key != null) {
                localIndices.addAll(subKeys.get(key));
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
    protected HashSet<Integer> getSetOfIndices(HashMap<String, String> conditions,
                                               String variableValue, boolean locOnVariable) {

        HashSet<Integer> intersection = new HashSet<>();
        HashSet<Integer> allNumbers = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
            allNumbers.add(i);
        }

        ArrayList<Integer> localIndices = null;
        for(int i = 0; i < this.parents_names.size(); i++) {
            String parentName = this.parents_names.get(i);
            String parentVal = conditions.get(this.parents_names.get(i));

            if(this.isParentContinuous.get(parentName)) {
                if(!String.valueOf(parentVal).equals("null")) {
                    localIndices = this.notNullLoc(parentName);
                } else {
                    localIndices = this.table.get(parentName).get(parentVal);
                }
            } else {
                localIndices = this.table.get(parentName).get(String.valueOf(parentVal));
            }
//            if(localIndices != null) {
            intersection.retainAll(new HashSet<>(localIndices));
//            }
        }
        if(locOnVariable) {
            if(this instanceof ContinuousVariable) {
                if(!String.valueOf(variableValue).equals("null")) {
                    localIndices = this.notNullLoc(this.getName());
                } else {
                    localIndices = this.table.get(this.name).get(variableValue);
                }
            } else {
                localIndices = this.table.get(this.name).get(String.valueOf(variableValue));
            }
            intersection.retainAll(new HashSet<>(localIndices));
        }
        return intersection;
    }

    /**
     * Just clears the parent_names, values, probabilities and table properties.
     * Updates the parent set of this variable, based on the fittest individuals from a generation.
     * @param parents New parents of this variable
     * @param fittest Fittest individuals from the lattest generation
     * @throws Exception Exception thrower for children classes
     */
    public void updateStructure(AbstractVariable[] parents, Individual[] fittest) throws Exception {
        this.isParentContinuous.clear();
        this.parents_names.clear();
        this.table.clear();
        this.values.clear();
        this.probabilities.clear();
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
     * Updates the table property of this variable. Automatically treats values for continuous variables.
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
     * Update probabilities of this Variable based on the fittest population of a generation.
     * @param fittest Fittest individuals from the last population
     * @throws Exception If any exception occurs
     */
    public void updateProbabilities(Individual[] fittest) throws Exception {
        HashMap<String, ArrayList<Integer>> thisVariable = this.table.get(this.getName());
        Object[] vals = thisVariable.keySet().toArray();

        // counts how many combinations of values there is for this variable
        // also updates shadow values
        int n_combinations = 0;
        for(Object val : vals) {
            ArrayList<Integer> indices = thisVariable.get((String)val);
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
                this.probabilities.add(1.0);
            }
            for(Integer index : indices) {
                this.values.set(index, thissv);
            }
        }
        n_combinations += 1;

        HashMap<String, Integer> ddd = new HashMap<>();
        int n_continuous = 0;
        for(String par : this.parents_names) {
            if(this.isParentContinuous.get(par)) {
                ddd.put(par, n_continuous);
                n_continuous += 1;
            }
        }
        if(this instanceof ContinuousVariable) {
            ddd.put(this.getName(), n_continuous);
            n_continuous += 1;
        }

        // keeps track of floats values for each one of the
        // combination of values in the fittest population
        double[][][] dda = new double[n_continuous][n_combinations][fittest.length];
        int[][] ddc = new int[n_continuous][n_combinations];
        for(int i = 0; i < n_continuous; i++) {
            for(int j = 0; j < n_combinations; j++) {
                ddc[i][j] = 0;
            }
        }

        for(int i = 0; i < fittest.length; i++) {
            int[] indices = this.getArrayOfIndices(fittest[i].getCharacteristics(), fittest[i].getCharacteristics().get(this.getName()), true);

            if(indices.length > 1) {
                throw new Exception("unexpected behaviour!");
            }

            // annotates float values
            for(String var : ddd.keySet()) {
                String val = fittest[i].getCharacteristics().get(var);
                if(val != null) {
                    dda[ddd.get(var)][indices[0]][ddc[ddd.get(var)][indices[0]]] = Double.parseDouble(val);
                    ddc[ddd.get(var)][indices[0]] += 1;
                }
            }
            this.probabilities.set(indices[0], this.probabilities.get(indices[0]) + 1);
        }

        // updates probabilities
        // also updates normal distributions, if this variable is continuous
        HashMap<String, Object[]> uniqueValues = new HashMap<>();
        int[] sizes = new int [this.parents_names.size()];
        int[] carousel = new int [this.parents_names.size()];

        for(int i = 0; i < this.parents_names.size(); i++) {
            Object[] parentUnVal = this.table.get(this.parents_names.get(i)).keySet().toArray();
            uniqueValues.put(
                this.parents_names.get(i),
                parentUnVal
            );
            sizes[i] = parentUnVal.length;
            carousel[i] = 0;
        }

        n_combinations /= this.uniqueValues.size();
        int counter = 0;
        while(counter < n_combinations) {
            HashMap<String, String> cond = new HashMap<>();

            for(int i = 0; i < parents_names.size(); i++) {
                String parentName = this.parents_names.get(i);
                cond.put(
                    parentName,
                    (String)uniqueValues.get(parentName)[carousel[i]]
                );
            }
            for(int j = sizes.length - 1; j >= 0; j--) {
                carousel[j] += 1;
                if(carousel[j] >= sizes[j]) {
                    carousel[j] = 0;
                } else {
                    break;
                }
            }
            counter += 1;

            HashSet<Integer> indices = this.getSetOfIndices(cond, null, false);

            // updates normal distributions
            if(this instanceof ContinuousVariable) {
                HashSet<Integer> nnSet = new HashSet<>(this.notNullLoc(this.getName()));
                nnSet.retainAll(indices);
                int idx = (int)nnSet.toArray()[0];

                int multivariate = 0;
                Object[] keys = ddd.keySet().toArray();
                boolean[] insert = new boolean[ddd.size()];
                int min_size = Integer.MAX_VALUE;
                for(int i = 0; i < ddd.size(); i++) {
                    if(ddc[ddd.get((String)keys[i])][idx] > 0) {
                        multivariate += 1;
                        insert[i] = true;

                        min_size = Math.min(
                            min_size,
                            ddc[ddd.get((String)keys[i])][idx]
                        );
                    } else {
                        insert[i] = false;
                    }
                }
                if(multivariate > 1) {
                    double[][] toProcess = new double[multivariate][];

                    int adder = 0;
                    for(int i = 0; i < ddd.size(); i++) {
                        if(insert[i]) {
                            toProcess[adder] = Arrays.copyOfRange(
                                dda[ddd.get((String)keys[i])][idx],
                                0,
                                min_size
                            );
                            adder += 1;
                        }

                    }
                    this.values.set(
                        idx,
                        new ShadowMultivariateNormalDistribution(
                            this.mt,
                            toProcess,
                            ((ContinuousVariable) this).a_min,
                            ((ContinuousVariable) this).a_max,
                            ((ContinuousVariable) this).scale_init
                        )
                    );
                } else {
                    double[] data = Arrays.copyOfRange(
                        dda[ddd.get(this.getName())][idx],
                        0,
                        ddc[ddd.get(this.getName())][idx]
                    );
                    DescriptiveStatistics ds = new DescriptiveStatistics(data);
                    double loc = ds.getMean();
                    double scale = ((ContinuousVariable) this).scale - ((ContinuousVariable) this).scale_init / ((ContinuousVariable) this).n_generations;
                    if(Double.isNaN(loc)) {
                        // this is an extreme case, where the normal distribution is reset
                        loc = ((ContinuousVariable) this).loc_init;
                    }

                    this.values.set(
                        idx,
                        new ShadowNormalDistribution(
                            this.mt,
                            loc,
                            scale,
                            ((ContinuousVariable) this).a_min,
                            ((ContinuousVariable) this).a_max,
                            ((ContinuousVariable) this).scale_init
                        )
                    );
                }
            }

            double sum = 0;
            for(int index : indices) {
                sum += this.probabilities.get(index);
            }
            for(int index : indices) {
                this.probabilities.set(index, this.probabilities.get(index) / sum);
            }
        }
    }

    // setters and updaters

    public abstract void updateUniqueValues(Individual[] fittest);

    // Getters

    /**
     *
     * @return An array with unique names of parents.
     */
    public ArrayList<String> getParentsNames() {
        return parents_names;
    }

    public HashSet<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public String getName() {
        return name;
    }

}
