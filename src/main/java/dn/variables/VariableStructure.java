package dn.variables;

import eda.individual.Individual;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public abstract class VariableStructure {

    protected String name;
    protected String[] parents;
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;
    protected ArrayList<Float> probabilities;
    protected ArrayList<String> values;
    protected HashSet<String> uniqueValues;

    public VariableStructure(String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
                             ArrayList<String> values, ArrayList<Float> probabilities) {

        this.name = name;
        this.parents = parents;
        this.table = table;
        this.values = values;
        this.probabilities = probabilities;
        this.uniqueValues = new HashSet<>(this.values);
    }

    protected HashSet<Integer> getSetOfIndices(HashMap<String, String> conditions, String variableValue, boolean locOnVariable) {
        HashSet<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        ArrayList<Integer> localIndices;
        for(int i = 0; i < this.parents.length; i++) {
            localIndices = this.table.get(this.parents[i]).get(conditions.get(this.parents[i]));
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
    public void updateStructure(VariableStructure[] parents, Individual[] fittest) throws Exception {
        this.parents = null;
        this.table.clear();
        this.values.clear();
        this.probabilities.clear();
    }

    protected int countDiscreteParents(VariableStructure[] parents) {
        int count = 0;

        for(VariableStructure parent : parents) {
            if(parent instanceof DiscreteVariable) {
                count += 1;
            }
        }
        return count;
    }

    protected int countContinuousParents(VariableStructure[] parents) {
        return parents.length - countDiscreteParents(parents);
    }

    protected boolean allContinuousParents(VariableStructure[] parents) {
        return countDiscreteParents(parents) == 0;
    }

    protected boolean allDiscreteParents(VariableStructure[] parents) {
        return countDiscreteParents(parents) != parents.length;
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
    public ArrayList<ArrayList<String>> generateCombinations(VariableStructure[] parents, boolean selfInclude) {
        int n_combinations = 1;
        int[] n_unique = new int [parents.length + (selfInclude? 1 : 0)];
        int[] repeat_every = new int [n_unique.length];

        for(int i = 0; i < parents.length; i++) {
            n_unique[i] = parents[i].getUniqueValues().size();
            n_combinations *= n_unique[i];
        }
        if(selfInclude) {
            n_combinations *= this.getUniqueValues().size();
            n_unique[parents.length] = this.getUniqueValues().size();
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

        for(int i = 0; i < parents.length; i++) {
            combinations = this.addVariableToCombinations(parents[i], combinations, repeat_every[i]);
        }
        if(selfInclude) {
            combinations = this.addVariableToCombinations(this, combinations, repeat_every[parents.length]);
        }
        return combinations;
    }

    /**
     * Given a list of combinations of unique values between variables, adds a new variable to the
     * roster of combinations.
     * @param var Variable to have its values added.
     * @param combinations List of combinations.
     * @param repeat_every Number of times to repeat each value in the combinations array.
     * @return The list of combinations updated.
     */
    private ArrayList<ArrayList<String>> addVariableToCombinations(VariableStructure var, ArrayList<ArrayList<String>> combinations, int repeat_every) {
        Object[] uniqueValues = var.getUniqueValues().toArray();
        int n_combinations = combinations.size();

        int global_counter = 0;
        while(global_counter < n_combinations) {
            for(Object value : uniqueValues) {
                for(int j = 0; j < repeat_every; j++) {
                    combinations.get(global_counter).add((String)value);
                    global_counter += 1;
                }
            }
        }
        return combinations;
    }

    public void setParents(VariableStructure[] parents) {
        this.parents = new String [parents.length];
        for(int i = 0; i < parents.length; i++) {
            this.parents[i] = parents[i].getName();
        }
    }

    public void removeContinuousParents() {

    }

    public String[] getParents() {
        return parents;
    }

    /**
     * Gets the values that this variable can assume.
     * @return
     */
    public HashSet<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public String getName() {
        return name;
    }

}
