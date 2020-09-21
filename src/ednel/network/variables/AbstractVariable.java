package ednel.network.variables;

import ednel.network.variables.statistics.StatisticsHandler;
import ednel.utils.Combination;
import ednel.utils.CombinationNotPresentException;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.exception.NotANumberException;
import org.apache.commons.math3.random.MersenneTwister;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class AbstractVariable {
    protected HashSet<String> all_parents;
    protected String name;

    /** Deterministic parents of this variable. */
    protected HashSet<String>  det_parents;

    /** Probabilistic parents of this variable. */
    protected HashSet<String> prob_parents;

    protected ArrayList<String> uniqueValues;

    /** ArrayList of indices that point to uniqueValues ArrayList. */
    protected ArrayList<Integer> indices;

    /** Probabilities associated with each entry in table. */
    protected ArrayList<Double> probabilities;

    /** Probability table. May Contain null values. */
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;

    protected MersenneTwister mt;

    /** number of entries in probability table */
    protected int n_combinations;

    private HashSet<Integer> all_indices;

    protected StatisticsHandler bs;

    public AbstractVariable(
            String name, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Double> probabilities, MersenneTwister mt) throws Exception {

        this.name = name;

        this.det_parents = new HashSet<>();
        this.det_parents.addAll(table.keySet());  // parents in initial probability table are deterministic parents
        this.det_parents.remove(this.name);
        this.prob_parents = new HashSet<>();
        this.all_parents = (HashSet<String>)this.det_parents.clone();

        this.probabilities = probabilities;
        this.table = table;

        HashSet<String> uniqueValuesSet = new HashSet<>();
        uniqueValuesSet.addAll(values);

        this.uniqueValues = new ArrayList<>(uniqueValuesSet.size());

        for (Object value : uniqueValuesSet.toArray()) {
            this.uniqueValues.add((String)value);
        }

        if(this.det_parents.size() > 1) {
            throw new Exception("Dependency Network must start with at most one deterministic parent!");
        }

        this.bs = new StatisticsHandler(this.getName(), this.table, this.probabilities, this.det_parents);

        this.indices = new ArrayList<>(this.probabilities.size());
        for(String value : values) {
            this.indices.add(this.uniqueValues.indexOf(value));
        }

        this.n_combinations = this.indices.size();
        this.initAllIndices();

        this.mt = mt;
    }

    private void initAllIndices() {
        // adds all indices to answer
        this.all_indices = new HashSet<>();
        for(int i = 0; i < this.n_combinations; i++) {
            this.all_indices.add(i);
        }
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
     * Samples a new value for this variable, based on conditions.
     *
     * @param lastStart Last values from Dependency Network.
     * @return A new value for this variable.
     */
    public String conditionalSampling(HashMap<String, String> lastStart) throws CombinationNotPresentException {
        // tries to sample a value given probabilistic parent values.
        // if probabilistic parents are absents from current sample, samples an unconditional value

        // killer move: checks first for deterministic parents. if any is null, then automatically returns null
        for(String detParent : this.getDeterministicParents()) {
            if(String.valueOf(lastStart.get(detParent)).equals("null")) {
                return null;
            }
        }

        // otherwise, samples value from probabilistic variables
        int[] indices = this.getArrayOfIndices(
                this.table, lastStart,null,false, this.getAllParents()
        );

        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            localProbs[i] = probabilities.get(indices[i]);
        }
        // samples values based on probabilities
        try {
            EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
            String value = this.uniqueValues.get(this.indices.get(localDist.sample()));
            if(!String.valueOf(value).equals("null")) {
                return value;
            }
        } catch(MathArithmeticException | NotANumberException mae) {
            System.err.println("Variable: " + this.name + " value: " + lastStart.get(this.name));
            System.err.println("probabilistic parents: ");
            for(String parent : this.prob_parents) {
                System.err.println("\t" + parent + " value: " + lastStart.get(parent));
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
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            HashMap<String, String> conditions, String variableValue, boolean useVariableValue,
            HashSet<String> allParents) throws CombinationNotPresentException {

        HashSet<Integer> intersection = this.getSetOfIndices(table, conditions, variableValue, useVariableValue, allParents);

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
            HashMap<String, String> conditions,
            String variableValue, boolean locOnVariable,
            HashSet<String> allParents
    ) throws CombinationNotPresentException {

        HashSet<Integer> intersection = new HashSet<>(this.all_indices);

        ArrayList<Integer> localIndices;
        for (Iterator<String> it = allParents.iterator(); it.hasNext(); ) {
            String parentName = it.next();
            String parentVal = conditions.get(parentName);
            localIndices = table.get(parentName).get(String.valueOf(parentVal));

            intersection.retainAll(localIndices);
        }
        // whether to find indices that match exactly this variable's value.
        // if true, returns a single index; otherwise, returns an array of indices
        if(locOnVariable) {
            localIndices = table.get(this.name).get(String.valueOf(variableValue));
            intersection.retainAll(localIndices);
        }
        if(intersection.isEmpty()) {
            throw new CombinationNotPresentException(
                    "combination of values not present in probability table: " + AbstractVariable.conditionsToString(conditions)
            );
        }

        return intersection;
    }

    /**
     * Clears the parent_names, values, probabilities and table properties.
     * Updates the parent set of this variable, based on the fittest individuals from a generation.
     *
     * @param all_parents All parents (both deterministic, that never change, and new probabilistic) of this variable.
     */
    public void updateStructure(AbstractVariable[] all_parents) {
        this.all_parents.removeAll(this.prob_parents);
        this.prob_parents.clear();

        for(AbstractVariable par : all_parents) {
            if(!this.det_parents.contains(par.getName())) {
                this.prob_parents.add(par.getName());
            }
        }
        this.all_parents.addAll(this.prob_parents);

        this.probabilities.clear();
        this.table.clear();
        this.indices.clear();
        this.n_combinations = 0;

        HashMap<String, ArrayList<String>> varUniqueValues = Combinator.getUniqueValuesFromVariables(all_parents, this);

        Object[] allVariableNames = varUniqueValues.keySet().toArray();

        this.n_combinations = 1;
        int[] n_unique = new int [varUniqueValues.size()];
        int[] repeat_every = new int [varUniqueValues.size()];

        this.table = new HashMap<>(varUniqueValues.size());
        for(int i = varUniqueValues.size() - 1; i >= 0; i--) {
            HashMap<String, ArrayList<Integer>> variableDict = new HashMap<>(varUniqueValues.get((String)allVariableNames[i]).size());

            for(Iterator<String> it = varUniqueValues.get((String)allVariableNames[i]).iterator(); it.hasNext(); ) {
                String val = it.next();
                variableDict.put(val, new ArrayList<>());
            }
            table.put((String)allVariableNames[i], variableDict);

            // computes the period of appearance of variable's dictUnique
            n_unique[i] = variableDict.size();
            this.n_combinations *= n_unique[i];
            if(i == (varUniqueValues.size() - 1)) {
                repeat_every[i] = 1;
            } else {
                repeat_every[i] = repeat_every[i + 1] * n_unique[i + 1];
            }
        }

        this.indices = new ArrayList<>(this.n_combinations);
        for(int i = 0; i < this.n_combinations; i++) {
            this.indices.add(-1);
        }


        for(int i = 0; i < allVariableNames.length; i++) {
            HashMap<String, ArrayList<Integer>> local = table.get((String)allVariableNames[i]);
            Object[] subKeys = local.keySet().toArray();

            // updates
            int counter = 0;
            while(counter < this.n_combinations) {
                for(int j = 0; j < subKeys.length; j++) {
                    for(int k = 0; k < repeat_every[i]; k++) {
                        local.get((String)subKeys[j]).add(counter);
                        counter += 1;
                    }
                }
            }
            // re-inserts
            table.put((String)allVariableNames[i], local);

            // re-creates array of indices
            if(allVariableNames[i].equals(this.getName())) {
                for(String value : local.keySet()) {
                    for(int index : local.get(value)) {
                        this.indices.set(index, this.uniqueValues.indexOf(value));
                    }
                }
            }
        }
        this.initAllIndices();
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
     * Updates probabilities of this variable, based on the relationship it has with its parents (both deterministic and
     * probabilistic ones). Will not update conditional probabilities on deterministic parents.
     *
     * @param currFittestValues A HashMap where each key is a variable name and each value the array of values from the
     *                          current sub-population of fittest individuals.
     * @param lastFittestValues A HashMap where each key is a variable name and each value the array of values from the
     *                          previous sub-population of fittest individuals.
     * @throws Exception If any exceptions occurs.
     */
    public void updateProbabilities(
            HashMap<String, ArrayList<String>> currFittestValues,
            HashMap<String, ArrayList<String>> lastFittestValues,
            double learningRate) throws Exception {

        // collects bivariate statistics between child variable and all of its parents from the current fittest population
        HashMap<String, HashMap<Combination, Double>> newBivariateStatistics = this.bs.getBivariateStatisticsFromPopulation(
                currFittestValues, this.prob_parents, this.det_parents, this.table
        );

        // updates bivariate statistics, as well as univariate statistics, with learning rate
        // but only if learning rate is different from 1
        if(Math.abs(1 - learningRate) > 0.01) {
            newBivariateStatistics = this.bs.updateStatisticsWithLearningRate(
                    this.prob_parents, learningRate,
                    this.uniqueValues, lastFittestValues,
                    newBivariateStatistics, this.table
            );
        }

        this.probabilities = new ArrayList<>(this.n_combinations);
        for(int i = 0; i < this.n_combinations; i++) {
            this.probabilities.add(-1.0);
        }

        if((this.prob_parents.size() == 0) && (this.det_parents.size() == 0)) {
            HashMap<Combination, Double> local = newBivariateStatistics.get(this.getName());

            for(Combination comb : local.keySet()) {
                String childValue = comb.getPairs().get(this.getName());
                ArrayList<Integer> indices = this.table.get(this.getName()).get(childValue);
                if(indices.size() > 1) {
                    throw new Exception("unexpected behavior!");
                }
                this.probabilities.set(indices.get(0), local.get(comb));
            }
        } else {  // there are new parents for this variable
            // probabilities is set to -1 to catch unexpected behaviors, but for updating
            // independent probabilities a 1 value is required. so we initialize a new arraylist with all
            // ones
            ArrayList<Double> shadowProb = new ArrayList<>();
            for(int i = 0; i < this.n_combinations; i++) {
                shadowProb.add(1.0);
            }

            // updates combination probabilities
            HashMap<HashMap<String, String>, HashSet<Integer>> parentCombinations = this.groupBy(this.all_parents);
            for(HashMap<String, String> combination : parentCombinations.keySet()) {
                HashMap<String, Integer> childValueIndex = new HashMap<>();

                for(String childVal : this.uniqueValues) {
                    HashSet<Integer> indices = new HashSet<>(parentCombinations.get(combination));
                    indices.retainAll(this.table.get(this.getName()).get(childVal));
                    if(indices.size() != 1) {
                        throw new Exception("unexpected behavior!");
                    }
                    int index = (int)indices.toArray()[0];
                    childValueIndex.put(childVal, index);

                    for(String parent : combination.keySet()) {
                        HashMap<String, String> pair = new HashMap<>();
                        pair.put(this.getName(), childVal);
                        pair.put(parent, combination.get(parent));

                        double bivariateProb = newBivariateStatistics.get(parent).get(new Combination(pair));

                        if(Double.isNaN(bivariateProb)) {
                            bivariateProb = 0.0;  // will only be used if laplace correction is set to false
                        }

                        shadowProb.set(index, shadowProb.get(index) * bivariateProb);
                    }
                }
                double sum = 0.0;
                for(int index : childValueIndex.values()) {
                    sum += shadowProb.get(index);
                }
                for(int index : childValueIndex.values()) {
                    this.probabilities.set(index, shadowProb.get(index) / sum);
                }
            }
        }
        this.bs.setOldBivariateStatistics(newBivariateStatistics);

        // TODO remove me later!
        if(this.probabilities.indexOf(-1.0) != -1) {
            throw new Exception("should not have -1 values!");
        }
    }

    /**
     * Groups indices in probability table by variables.
     *
     * Generates a dictionary where the key is a HashMap with a value assignment for each variable,
     * and the value is an ArrayList to where those values appear in the probabilistic table.
     *
     * If no variable is provided, then returns all indices.
     *
     * @return Returns a dictionary where the key is the combination of parent values (excluded child, i.e this variable
     * values) and the dictionary value for that key is the set of indices where that combination of parent values happen.
     */
    private HashMap<HashMap<String, String>, HashSet<Integer>> groupBy(HashSet<String> variables) {
        HashMap<HashMap<String, String>, HashSet<Integer>> results = new HashMap<>();
        if(variables.size() == 0) {
            HashSet<Integer> allIndices = new HashSet<>();
            for(int i = 0; i < this.uniqueValues.size(); i++) {
                allIndices.add(i);
            }
            results.put(new HashMap<>(), allIndices);
            return results;
        }

        HashMap<String, Object[]> arrayOfValues = new HashMap<>();
        int[] sizes = new int [variables.size() + 1];
        int[] carousel = new int [variables.size() + 1];

        Object[] variables_array = variables.toArray();
        int n_combinations = 1;
        for(int i = 0; i < variables_array.length; i++) {
            Object[] uniqueValues = this.table.get((String)variables_array[i]).keySet().toArray();
            arrayOfValues.put(
                    (String)variables_array[i],
                    uniqueValues
            );
            sizes[i] = uniqueValues.length;
            n_combinations *= sizes[i];
            carousel[i] = 0;
        }

        int counter = 0;
        while(counter < n_combinations) {
            HashMap<String, String> cond = new HashMap<>();

            // adds a new combination of values for parents
            for (int i = 0; i < variables_array.length; i++) {
                String varName = (String) variables_array[i];
                cond.put(
                        varName,
                        (String)arrayOfValues.get(varName)[carousel[i]]
                );
            }
            carousel = advanceCarousel(carousel, sizes);
            counter += 1;

            try {
                String variableValue = null;
                boolean locOnVariable = false;
                if(variables.contains(this.getName())) {
                    variableValue = cond.get(this.getName());
                    locOnVariable = true;
                }

                HashSet<Integer> indices = this.getSetOfIndices(
                        this.table, cond, variableValue, locOnVariable, variables
                );
                results.put(cond, indices);
            } catch(CombinationNotPresentException cnp) {
                System.err.println("unexpected behavior!");
            }

        }
        return results;
    }


    public HashMap<String, Double> getTablePrettyPrint() {
        ArrayList<String> lines = new ArrayList<>(this.indices.size());

        for(int i = 0; i < this.indices.size(); i++) {
            lines.add("");
        }

        Object[][] toProcess = {det_parents.toArray(), prob_parents.toArray(), new String[]{name}};

        for(Object[] current : toProcess) {  // iterates over groups of variables
            for(Object variableName : current) {  // iterates over variables of that group
                HashMap<String, ArrayList<Integer>> variableValues = this.table.get((String)variableName);
                for(String variableVal : variableValues.keySet()) {  // iterate over variable values
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

    public String getName() {
        return this.name;
    }

    /**
     * Counts both probabilistic and deterministic parents.
     * @return
     */
    public int getParentCount() {
        return this.prob_parents.size() + this.det_parents.size();
    }

    /**
     * Returns all variables that must be sampled before this variable, either fixed or mutable.
     */
    public HashSet<String> getDeterministicParents() {
        return this.det_parents;
    }

    public HashSet<String> getProbabilisticParents() {
        return this.prob_parents;
    }

    public static AbstractVariable fromPath(String path, MersenneTwister mt) throws Exception {

//        BufferedReader csvReader = new BufferedReader(new FileReader(path));
        BufferedReader csvReader = new BufferedReader(
                new InputStreamReader(
                        AbstractVariable.class.getClassLoader().getResourceAsStream(path),
                        StandardCharsets.UTF_8
                )
        );

        String variableName = path.substring(
//                path.lastIndexOf(File.separator) + 1,
                path.lastIndexOf("/") + 1,
                path.lastIndexOf(".")
        );

        String row = csvReader.readLine();
        String[] header = row.split(",(?![^(]*\\))");

        int n_variables_table = 1;
//        HashSet<String> parents_names = new HashSet<>();

        HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(header.length);
        table.put(variableName, new HashMap<>());

        // if there are more entries than simply the name of this variable plus column "probability"
        if(header.length > 2) {
            n_variables_table = n_variables_table + header.length - 2;
            for(int k = 0; k < header.length - 2; k++) {
//                parents_names.add(header[k]);
                table.put(header[k], new HashMap<>());
            }
        }

        ArrayList<Double> probabilities = new ArrayList<>();
        ArrayList<String> values = new ArrayList<>();

        int index = 0;

        while ((row = csvReader.readLine()) != null) {
            String[] this_data = row.split(",(?![^(]*\\))");

            probabilities.add(Double.valueOf(this_data[this_data.length - 1]));
            values.add(this_data[this_data.length - 2]);

            for(int k = 0; k < this_data.length - 1; k++) {
                // if this variable does not have this value
                if(!table.get(header[k]).containsKey(this_data[k])) {
                    table.get(header[k]).put(this_data[k], new ArrayList<>());
                }
                table.get(header[k]).get(this_data[k]).add(index);
            }
            index++;
        }
        csvReader.close();  // finishes reading this file

        return new AbstractVariable(variableName, table, values, probabilities, mt);

    }

    public ArrayList<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public HashSet<String> getAllParents() {
        return this.all_parents;
    }

}



