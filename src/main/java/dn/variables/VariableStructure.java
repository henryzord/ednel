package dn.variables;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

import com.sun.org.apache.xpath.internal.operations.Variable;
import eda.individual.Individual;

public abstract class VariableStructure {
    protected String name;
    protected ArrayList<String> parents_names;
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;
    protected ArrayList<Float> probabilities;
    protected ArrayList<String> values;
    protected HashSet<String> uniqueValues;

    public VariableStructure(String name, ArrayList<String> parents_names, HashMap<String,
        HashMap<String, ArrayList<Integer>>> table, ArrayList<String> values, ArrayList<Float> probabilities) {

        this.name = name;
        this.parents_names = parents_names;
        this.table = table;
        this.values = values;
        this.probabilities = probabilities;
        this.uniqueValues = new HashSet<>(this.values);
    }

    protected HashSet<Integer> getSetOfIndices(HashMap<String, String> conditions,
                                               String variableValue, boolean locOnVariable) {
        HashSet<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        ArrayList<Integer> localIndices;
        for(int i = 0; i < this.parents_names.size(); i++) {
            localIndices = this.table.get(this.parents_names.get(i)).get(conditions.get(this.parents_names.get(i)));
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
        this.parents_names = new ArrayList<>(parents.length);
        for(int i = 0; i < parents.length; i++) {
            this.parents_names.add(parents[i].getName());
        }
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

    /**
     * Removes discrete continuous parents from the set of parents_names of this variable.
     * Also returns the variables that are only discrete.
     * @param parents
     * @return
     */
    protected VariableStructure[] removeContinuousParents(VariableStructure[] parents) {
        int countDiscreteParents = this.countDiscreteParents(parents);
        VariableStructure[] discreteParents = null;
        if(countDiscreteParents != parents.length) {
            discreteParents = new VariableStructure [countDiscreteParents];
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
    public ArrayList<ArrayList<String>> generateCombinations(VariableStructure[] parents, boolean selfInclude) throws Exception {
        ArrayList<ArrayList<String>> allValues = new ArrayList<>(parents.length);

        Method getUniqueValues = VariableStructure.class.getMethod("getUniqueValues");
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
    public HashSet<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public String getName() {
        return name;
    }

    protected HashMap<String, HashSet<String>> getUniqueValuesFromVariables(VariableStructure[] parents, boolean selfInclude) {

        HashMap<String, HashSet<String>> uniqueValues = new HashMap<>(parents.length + 1);

        for(int i = 0; i < parents.length; i++) {
            uniqueValues.put(parents[i].getName(), parents[i].getUniqueValues());
        }
        if(selfInclude) {
            uniqueValues.put(this.getName(), this.getUniqueValues());
        }
        return uniqueValues;
    }

    protected void updateTableEntries(HashMap<String, HashSet<String>> eouniqueValues) {
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
            this.values.add(values.get(j));
            this.probabilities.add((float)0.0);
            table.get(indices.get(j)).get(values.get(j)).add(i);

        }
    }
}
