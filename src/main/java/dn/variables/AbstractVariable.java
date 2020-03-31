package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public abstract class AbstractVariable {
    protected String name;
    protected ArrayList<String> parents_names;
    protected ArrayList<Boolean> isParentDiscrete;

    protected HashSet<ShadowValue> uniqueValues;

    protected ArrayList<ShadowValue> values;
    protected ArrayList<Double> probabilities;

    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;

    protected MersenneTwister mt;

    protected double learningRate;
    protected int n_generations;

    public AbstractVariable(
            String name, ArrayList<String> parents_names, ArrayList<Boolean> isParentDiscrete,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            HashSet<ShadowValue> uniqueValues, ArrayList<ShadowValue> values, ArrayList<Double> probabilities,
            MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        this.name = name;
        this.parents_names = parents_names;
        this.isParentDiscrete = isParentDiscrete;

        this.uniqueValues = uniqueValues;

        this.values = values;
        this.probabilities = probabilities;

        this.table = table;

        this.mt = mt;
        this.learningRate = learningRate;
        this.n_generations = n_generations;
    }

    public abstract String[] unconditionalSampling(int sample_size) throws Exception;

    protected int conditionalSamplingIndex(HashMap<String, String> lastStart) {
        int[] indices = this.getIndices(lastStart, null, false);

        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            localProbs[i] = probabilities.get(indices[i]);
        }

        EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
        return localDist.sample();
    }

    public HashMap<String, HashMap<String, ArrayList<Integer>>> getTable() {
        return this.table;
    }

    /**
     * Update probabilities of this Variable based on the fittest population of a generation.
     * @param fittest
     * @throws Exception
     */
    public void updateProbabilities(Individual[] fittest) throws Exception {
        ArrayList<Float> occurs = new ArrayList<>();
        for(int i = 0; i < probabilities.size(); i++) {
            occurs.add((float)0.0);
        }

        // updates a continuous variable
        if(this.getClass().equals(ContinuousVariable.class)) {
            // gets the count of occurrences
            for(int i = 0; i < fittest.length; i++) {
                HashSet<Integer> parentIndices = this.getSetOfIndices(
                        fittest[i].getCharacteristics(),
                        null,
                        false
                );
                HashSet<Integer> nullIndices = this.getSetOfIndices(
                        fittest[i].getCharacteristics(),
                        null,
                        true
                );

                parentIndices.retainAll(nullIndices);

                for(Object index : parentIndices.toArray()) {
                    occurs.set((int)index, occurs.get((int)index) + 1);
                }
            }
        } else {  // updates a discrete variable
            // gets the count of occurrences
            for(int i = 0; i < fittest.length; i++) {
                int[] indices = this.getIndices(
                        fittest[i].getCharacteristics(),
                        fittest[i].getCharacteristics().get(this.name),
                        true
                );
                for(int index : indices) {
                    occurs.set(index, occurs.get(index) + 1);
                }
            }
        }

        // generates combinations of values
        ArrayList<HashMap<String, String>> combinations = new ArrayList<>();
        for(Object value : this.getUniqueValues().toArray()) {
            HashMap<String, String> local = new HashMap<>();
            local.put(this.name, null);
            combinations.add(local);
        }
        for(String parent : this.parents_names) {
            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>();
            for(int i = 0; i < combinations.size(); i++) {
                Object[] parentUniqueVals = this.table.get(parent).keySet().toArray();
                for(int j = 0; j < parentUniqueVals.length; j++) {
                    HashMap<String, String> local = (HashMap<String, String>)combinations.get(i).clone();
                    local.put(parent, (String)parentUniqueVals[j]);
                    new_combinations.add(local);
                }
            }
            combinations = new_combinations;
        }

        // calculates the sum
        for(int i = 0; i < combinations.size(); i++) {
            int[] indices = this.getIndices(combinations.get(i), null, false);
            double  sum = 0, newSum = 0, newValue, rest;
            for(int j = 0; j < indices.length; j++) {
                sum += occurs.get(indices[j]);
            }

            if(sum > 0) {
                // updates probabilities using learning rate and relative frequencies
                for(int j = 0; j < indices.length; j++) {
                    newValue = ((1.0 - this.learningRate) * this.probabilities.get(indices[j]) +
                            this.learningRate * occurs.get(indices[j]) / sum);
                    newSum += newValue;
                    this.probabilities.set(
                            indices[j],
                            newValue

                    );
                }
                rest = 1 - newSum;
                newSum = 0;
                for(int j = 0; j < indices.length; j++) {
                    this.probabilities.set(
                        indices[j],
                        this.probabilities.get(indices[j]) + rest / indices.length
                    );
                    newSum += this.probabilities.get(indices[j]);
                }

                if(Math.abs(1 - newSum) > 0.01) {
                    throw new Exception("does not sum up to 1!");
                }
            }
        }  // ends calculating the sum

        // clears NaN values
        // TODO throw away this code later
        for(int i = 0; i < probabilities.size(); i++) {
            if(Double.isNaN(probabilities.get(i)) || probabilities.get(i) < 0) {
                probabilities.set(i, 0.0);
            }
        }
    }

    /**
     * Samples a new value for this variable, based on conditions.
     * @param lastStart Last values from the Dependency Network.
     * @return A new value for this variable.
     */
    public String conditionalSampling(HashMap<String, String> lastStart) {
        ShadowValue val = this.values.get(this.conditionalSamplingIndex(lastStart));
        if(val != null) {
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
     * @param locOnVariable Whether to take into account the value of the current variable or not.
     *                      If true, this method will search for the table index that matches exactly the
     *                      parent values PLUS this variable value. If false, will returh a set of indices
     *                      where parent values match.
     * @return A set of indices
     */
    protected HashSet<Integer> getSetOfIndices(HashMap<String, String> conditions,
                                               String variableValue, boolean locOnVariable) {
        HashSet<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        ArrayList<Integer> localIndices;
        for(int i = 0; i < this.parents_names.size(); i++) {
            localIndices = this.table.get(this.parents_names.get(i)).get(conditions.get(this.parents_names.get(i)));
            // TODo here. code breaks when using continuous parents
//            throw new Exception("use inverse selection for continuous variables that are parents.");
            if(localIndices != null) {
                intersection.retainAll(new HashSet<>(localIndices));
            }
        }
        if(locOnVariable) {
            localIndices = this.table.get(this.name).get(variableValue);
            intersection.retainAll(new HashSet<>(localIndices));
        }
        return intersection;
    }

    /**
     * Get indices in the table object that correspond to a given assignment of values to the set
     * of parents for this variable.
     * @return
     */
    protected int[] getIndices(HashMap<String, String> conditions, String variableValue, boolean locOnVariable) {
        HashSet<Integer> intersection = this.getSetOfIndices(conditions, variableValue, locOnVariable);

        Object[] indices = intersection.toArray();
        int[] intIndices = new int [indices.length];
        for(int i = 0; i < indices.length; i++) {
            intIndices[i] = (int)indices[i];
        }

        return intIndices;
    }

    /**
     * Updates the parent set of this variable, based on the fittest individuals from a generation.
     * @param fittest
     * @throws Exception
     */
    public void updateStructure(AbstractVariable[] parents, Individual[] fittest) throws Exception {
        this.parents_names = new ArrayList<>(parents.length);
        for(int i = 0; i < parents.length; i++) {
            this.parents_names.add(parents[i].getName());
        }
        this.table.clear();
        this.values.clear();
        this.probabilities.clear();
    }

    protected int countDiscreteParents(AbstractVariable[] parents) {
        int count = 0;

        for(AbstractVariable parent : parents) {
            if(parent instanceof DiscreteVariable) {
                count += 1;
            }
        }
        return count;
    }

    /**
     * Removes discrete continuous parents from the set of parents_names of this variable.
     * Also returns the variables that are only discrete.
     * @param parents
     * @return
     */
    protected AbstractVariable[] removeContinuousParents(AbstractVariable[] parents) {
        int countDiscreteParents = this.countDiscreteParents(parents);
        AbstractVariable[] discreteParents = null;
        if(countDiscreteParents != parents.length) {
            discreteParents = new AbstractVariable [countDiscreteParents];
            int counter = 0;
            for(int i = 0; i < parents.length; i++) {
                if(parents[i] instanceof DiscreteVariable) {
                    discreteParents[counter] = parents[i];
                    counter += 1;
                } else {
                    this.parents_names.remove(parents[i].getName());
                }
            }
        } else {
            discreteParents = parents;
        }
        return discreteParents;
    }

    protected int countContinuousParents(AbstractVariable[] parents) {
        return parents.length - countDiscreteParents(parents);
    }

    protected boolean allContinuousParents(AbstractVariable[] parents) {
        return countDiscreteParents(parents) == 0;
    }

    protected boolean allDiscreteParents(AbstractVariable[] parents) {
        return countDiscreteParents(parents) != parents.length;
    }

    /**
     * Generates the product between lists, stored in a list of lists.
     *
     * @param values A list of lists.
     * @return The product of each list with every other list, in the list of lists.
     */
    public ArrayList<ArrayList<String>> generateCombinations(ArrayList<ArrayList<String>> values) {
        int n_combinations = 1;
        int[] n_unique = new int [values.size()];
        int[] repeat_every = new int [n_unique.length];

        for(int i = 0; i < values.size(); i++) {
            n_unique[i] = values.get(i).size();
            n_combinations *= n_unique[i];
        }
        ArrayList<ArrayList<String>> combinations = new ArrayList<>();
        for(int i = 0; i < n_combinations; i++) {
            combinations.add(new ArrayList<>());
        }

        // computes the period of appearance of variable's values
        repeat_every[repeat_every.length - 1] = 1;
        for(int i = repeat_every.length - 2; i >= 0; i--) {
            repeat_every[i] = repeat_every[i + 1] * n_unique[i + 1];
        }

        for(int i = 0; i < values.size(); i++) {
            combinations = this.addVariableToCombinations(values.get(i), combinations, repeat_every[i]);
        }
        return combinations;
    }

    public ArrayList<ArrayList<String>> generateCombinations(HashMap<String, HashSet<String>> values) {
        ArrayList<ArrayList<String>> allValues = new ArrayList<>(values.size());
        Object[] keys = values.keySet().toArray();
        for(int i = 0; i < keys.length; i++) {
            Object[] localArray = values.get(keys[i]).toArray();
            ArrayList<String> stringValues = new ArrayList<>(localArray.length);
            for(int j = 0; j < localArray.length; j++) {
                stringValues.add((String)localArray[j]);
            }
            allValues.add(stringValues);
        }
        return generateCombinations(allValues);
    }

    /**
     * Generates a list of combinations of values between variables.
     * Values of this variable (if selfInclude = true) are placed in the end of the list.
     *
     * @param parents Parents of this variable.
     * @param selfInclude Whether to include values of this variable. The values will be placed
     *                    at the end of the list.
     * @return Combinations of values between the (discrete) variables.
     */
    public ArrayList<ArrayList<String>> generateCombinations(AbstractVariable[] parents, boolean selfInclude) throws Exception {
        ArrayList<ArrayList<String>> allValues = new ArrayList<>(parents.length);

        Method getUniqueValues = AbstractVariable.class.getMethod("getUniqueValues");
        ArrayList<HashSet<String>> localValues = new ArrayList<>(parents.length + 1);
        for(int i = 0; i < parents.length; i++) {
            localValues.add((HashSet<String>)getUniqueValues.invoke(parents[i], null));
        }
        if(selfInclude) {
            localValues.add((HashSet<String>) getUniqueValues.invoke(this, null));
        }

        for(int i = 0; i < localValues.size(); i++) {
            ArrayList<String> local = new ArrayList<>();
            String[] dummy = new String[localValues.get(i).size()];
            localValues.toArray(dummy);
            local.addAll(localValues.get(i));
            allValues.add(local);
        }
        return generateCombinations(allValues);
    }

    /**
     * Given a list of combinations of unique values between variables, adds a new variable to the
     * roster of combinations.
     * @param values List of values to be added to list of combinations
     * @param combinations List of combinations.
     * @param repeat_every Number of times to repeat each value in the combinations array.
     * @return The list of combinations updated.
     */
    private ArrayList<ArrayList<String>> addVariableToCombinations(
        ArrayList<String> values, ArrayList<ArrayList<String>> combinations, int repeat_every) {

        int n_combinations = combinations.size();

        int global_counter = 0;
        while(global_counter < n_combinations) {
            for (Iterator<String> it = values.iterator(); it.hasNext(); ) {
                String value = it.next();
                for(int j = 0; j < repeat_every; j++) {
                    combinations.get(global_counter).add(value);
                    global_counter += 1;
                }
            }
        }
        return combinations;
    }

    public ArrayList<String> getParentsNames() {
        return parents_names;
    }

    /**
     * Gets the values that this variable can assume.
     * @return
     */
    public HashSet<ShadowValue> getUniqueValues() {
        return this.uniqueValues;
    }

    public String getName() {
        return name;
    }

    protected HashMap<String, HashSet<String>> getUniqueValuesFromVariables(AbstractVariable[] parents, boolean selfInclude) {

        HashMap<String, HashSet<String>> uniqueValues = new HashMap<>(parents.length + 1);

        for (AbstractVariable parent : parents) {
            uniqueValues.put(parent.getName(), parent.getUniqueValues());
        }
        if(selfInclude) {
            uniqueValues.put(this.getName(), this.getUniqueValues());
        }
        return uniqueValues;
    }

    protected void updateTableEntries(HashMap<String, HashSet<String>> eouniqueValues) throws Exception {
        ArrayList<ArrayList<String>> combinations = this.generateCombinations(eouniqueValues);

        int n_combinations = combinations.size();

        this.values = new ArrayList<>(n_combinations);
        this.probabilities = new ArrayList<>(n_combinations);

        ArrayList<String> indices = new ArrayList<>(eouniqueValues.size());

        this.table = new HashMap<>(n_combinations);
        for(String key : eouniqueValues.keySet()) {
            if(!key.equals(this.getName())) {
                Object[] localUniqueValues = eouniqueValues.get(key).toArray();
                indices.add(key);

                this.table.put(key, new HashMap<>(localUniqueValues.length));
                for(Object value : localUniqueValues) {
                    this.table.get(key).put((String)value, new ArrayList<>());
                }
            }
        }
        this.table.put(this.name, new HashMap<>(eouniqueValues.get(this.name).size()));
        for(Object value : eouniqueValues.get(this.name).toArray()) {
            this.table.get(this.name).put((String)value, new ArrayList<>());
        }
        indices.add(this.name);

        for(int i = 0; i < combinations.size(); i++) {
            ArrayList<String> values = combinations.get(i);
            int j;

            for(j = 0; j < values.size() - 1; j++) {
                table.get(indices.get(j)).get(values.get(j)).add(i);
            }
            j = values.size() - 1;
            this.values.add(
                new ShadowValue(
                    String.class.getMethod("toString"),
                    values.get(j)
                )
            );
            this.probabilities.add(0.0);
            table.get(indices.get(j)).get(values.get(j)).add(i);
        }
    }

}
