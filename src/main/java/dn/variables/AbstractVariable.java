package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public abstract class AbstractVariable {
    protected String name;
    protected ArrayList<String> parents_names;
    protected HashMap<String, Boolean> isParentContinuous;

    protected HashSet<String> uniqueValues;

    protected ArrayList<Shadowvalue> values;
    protected ArrayList<Double> probabilities;

    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;

    protected MersenneTwister mt;

    protected double learningRate;
    protected int n_generations;

    public AbstractVariable(
        String name, ArrayList<String> parents_names, HashMap<String, Boolean> isParentContinuous,
        HashMap<String, HashMap<String, ArrayList<Integer>>> table,
        HashSet<String> uniqueValues, ArrayList<Shadowvalue> values, ArrayList<Double> probabilities,
        MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        this.name = name;
        this.parents_names = parents_names;
        this.isParentContinuous = isParentContinuous;

        this.uniqueValues = uniqueValues;

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
     * @return An array of indices
     */
    protected int[] getArrayOfIndices(HashMap<String, String> conditions, String variableValue, boolean locOnVariable) {
        HashSet<Integer> intersection = this.getSetOfIndices(conditions, variableValue, locOnVariable);

        int[] intIndices = new int [intersection.size()];
        Object[] values = intersection.toArray();
        for(int i = 0; i < values.length; i++) {
            intIndices[i] = (int)values[i];
        }
        return intIndices;
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

        ArrayList<Integer> localIndices = null;
        for(int i = 0; i < this.parents_names.size(); i++) {
            String parentName = this.parents_names.get(i);
            String parentVal = conditions.get(this.parents_names.get(i));

            if(this.isParentContinuous.get(parentName)) {
                if(parentVal != null) {
                    localIndices = new ArrayList<>();
                    HashMap<String, ArrayList<Integer>> subKeys = this.table.get(parentName);
                    for(String key : subKeys.keySet()) {
                        if(key != null) {
                            localIndices.addAll(subKeys.get(key));
                        }
                    }
                } else {
                    localIndices = this.table.get(parentName).get(parentVal);
                }
            } else {
                localIndices = this.table.get(parentName).get(parentVal);
            }
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

        HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(dictUnique.size());
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
     * @param fittest
     * @throws Exception
     */
    public void updateProbabilities(Individual[] fittest) throws Exception {
        //        int n_combinations = new_table.size();
//
//        ArrayList<String> indices = new ArrayList<>(dictUnique.size());
//
//        for(int i = 0; i < new_table.size(); i++) {
//            ArrayList<String> values = new_table.get(i);
//            int j;
//
//            for(j = 0; j < values.size() - 1; j++) {
//                table.get(indices.get(j)).get(values.get(j)).add(i);
//            }
//            j = values.size() - 1;
//            this.values.add(
//                new Shadowvalue(
//                    String.class.getMethod("toString"),
//                    values.get(j)
//                )
//            );
//            this.probabilities.add(0.0);
//            table.get(indices.get(j)).get(values.get(j)).add(i);
//        }
        
//        ArrayList<Double> occurs = new ArrayList<>();
//        for(int i = 0; i < probabilities.size(); i++) {
//            occurs.add(0.0);
//        }
//
//        // updates a continuous variable
//        if(this.getClass().equals(ContinuousVariable.class)) {
//            // gets the count of occurrences
//            for(int i = 0; i < fittest.length; i++) {
//                HashSet<Integer> parentIndices = this.getSetOfIndices(
//                    fittest[i].getCharacteristics(),
//                    null,
//                    false
//                );
//                HashSet<Integer> nullIndices = this.getSetOfIndices(
//                    fittest[i].getCharacteristics(),
//                    null,
//                    true
//                );
//
//                parentIndices.retainAll(nullIndices);
//
//                for(Object index : parentIndices.toArray()) {
//                    occurs.set((int)index, occurs.get((int)index) + 1);
//                }
//            }
//        } else {  // updates a discrete variable
//            // gets the count of occurrences
//            for(int i = 0; i < fittest.length; i++) {
//                int[] indices = this.getArrayOfIndices(
//                    fittest[i].getCharacteristics(),
//                    fittest[i].getCharacteristics().get(this.name),
//                    true
//                );
//                for(int index : indices) {
//                    occurs.set(index, occurs.get(index) + 1);
//                }
//            }
//        }
//
//        // TODO update this!
////        throw new Exception("update this!");
//
//        // generates combinations of values
//        ArrayList<HashMap<String, String>> combinations = new ArrayList<>();
//        for(Object value : this.getUniqueValues().toArray()) {
//            HashMap<String, String> local = new HashMap<>();
//            local.put(this.name, null);
//            combinations.add(local);
//        }
//        for(String parent : this.parents_names) {
//            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>();
//            for(int i = 0; i < combinations.size(); i++) {
//                Object[] parentUniqueVals = this.table.get(parent).keySet().toArray();
//                for(int j = 0; j < parentUniqueVals.length; j++) {
//                    HashMap<String, String> local = (HashMap<String, String>)combinations.get(i).clone();
//                    local.put(parent, (String)parentUniqueVals[j]);
//                    new_combinations.add(local);
//                }
//            }
//            combinations = new_combinations;
//        }
//
//        // calculates the sum
//        for(int i = 0; i < combinations.size(); i++) {
//            int[] indices = this.getArrayOfIndices(combinations.get(i), null, false);
//            double  sum = 0, newSum = 0, newValue, rest;
//            for(int j = 0; j < indices.length; j++) {
//                sum += occurs.get(indices[j]);
//            }
//
//            if(sum > 0) {
//                // updates probabilities using learning rate and relative frequencies
//                for(int j = 0; j < indices.length; j++) {
//                    newValue = ((1.0 - this.learningRate) * this.probabilities.get(indices[j]) +
//                        this.learningRate * occurs.get(indices[j]) / sum);
//                    newSum += newValue;
//                    this.probabilities.set(
//                        indices[j],
//                        newValue
//
//                    );
//                }
//                rest = 1 - newSum;
//                newSum = 0;
//                for(int j = 0; j < indices.length; j++) {
//                    this.probabilities.set(
//                        indices[j],
//                        this.probabilities.get(indices[j]) + rest / indices.length
//                    );
//                    newSum += this.probabilities.get(indices[j]);
//                }
//
//                if(Math.abs(1 - newSum) > 0.01) {
//                    throw new Exception("does not sum up to 1!");
//                }
//            }
//        }  // ends calculating the sum
//
//        // clears NaN values
//        // TODO throw away this code later
//        for(int i = 0; i < probabilities.size(); i++) {
//            if(Double.isNaN(probabilities.get(i)) || probabilities.get(i) < 0) {
//                probabilities.set(i, 0.0);
//            }
//        }
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
