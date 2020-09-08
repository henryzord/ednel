package ednel.network.variables;

import ednel.utils.Combination;

import java.util.*;

public class BivariateStatistics {

    /** Bivariate conditional probabilities */
    protected HashMap<String, HashMap<Combination, Double>> oldBivariateStatistics;

    protected String variable_name;
    
    public BivariateStatistics(
            String variable_name,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<Double> probabilities, HashSet<String> det_parents
    ) throws Exception {

        this.variable_name = variable_name;
        this.oldBivariateStatistics = new HashMap<>();

        // adds univariate statistics
        this.oldBivariateStatistics.put(this.variable_name,
                addBivariateStatisticsFromTable(null, table, probabilities)
        );
        if(det_parents.size() > 0) {
            for(String detParent : det_parents) {
                this.oldBivariateStatistics.put(
                        detParent,
                        addBivariateStatisticsFromTable(detParent, table, probabilities)
                );
            }
        }
    }

    private HashMap<Combination, Double> addBivariateStatisticsFromTable(
            String parent, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<Double> probabilities
    ) throws Exception {
        
        HashMap<Combination, Double> built = new HashMap<>();

        // adds univariate statistics, independent of the current number of parents

        // add bivariate statistics if any
        if(parent != null) {
            for(String parentVal : table.get(parent).keySet()) {
                for(String childVal : table.get(this.variable_name).keySet()) {

                    HashSet<Integer> parentIndices = new HashSet<>();
                    parentIndices.addAll(table.get(parent).get(parentVal));
                    HashSet<Integer> childIndices = new HashSet<>();
                    childIndices.addAll(table.get(this.variable_name).get(childVal));
                    parentIndices.retainAll(childIndices);

                    if(parentIndices.size() != 1) {
                        throw new Exception("unexpected behavior!");
                    }

                    HashMap<String, String> valuePairs = new HashMap<>();
                    valuePairs.put(parent, parentVal);
                    valuePairs.put(this.variable_name, childVal);

                    built.put(new Combination(valuePairs), probabilities.get((int)parentIndices.toArray()[0]));
                }
            }
        } else {
            double overSum = 0;

            for(String childVal : table.get(this.variable_name).keySet()) {
                ArrayList<Integer> childIndices = table.get(this.variable_name).get(childVal);

                double localSum = 0;

                // does not take into account probability to be null; irrelevant
                if(!childVal.equals("null")) {
                    for(int index : childIndices) {
                        localSum += probabilities.get(index);
                    }
                }
                overSum += localSum;

                HashMap<String, String> valuePairs = new HashMap<>();
                valuePairs.put(this.variable_name, childVal);
                Combination comb = new Combination(valuePairs);
                built.put(comb, localSum);
            }
            for(Combination comb : built.keySet()) {
                built.put(comb, built.get(comb) / overSum);
            }
        }
        return built;
    }

    /**
     * Also collects univariate statistics of child variable.
     *
     * @param fittestValues A dictionary where the key is a variable name, and the value an ArrayList of values assumed
     *                      by that variable in the current generation fittest population.
     * @return
     */
    public HashMap<String, HashMap<Combination, Double>> getBivariateStatisticsFromPopulation(
            HashMap<String, ArrayList<String>> fittestValues, HashSet<String> all_parents, HashSet<String> det_parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table) {

        HashMap<String, HashMap<Combination, Double>> newBivariateStatistics = new HashMap<>();

        // adds univariate statistics
        HashMap<Combination, Double> uni_local = this.initializeCombinationsOfValues(null, null, table, null);
        newBivariateStatistics.put(this.variable_name, uni_local);

        // adds bivariate statistics: creates dictionaries of combinations and counts
        for(String parentName : all_parents) {
            HashSet<String> parentValues = new HashSet<>();
            parentValues.addAll(table.get(parentName).keySet());
            HashMap<Combination, Double> local = this.initializeCombinationsOfValues(parentName, parentValues, table, det_parents.contains(parentName));
            newBivariateStatistics.put(parentName, local);
        }
        HashSet<String> prob_parents = new HashSet<>();
        prob_parents.addAll(all_parents);
        prob_parents.removeAll(det_parents);

        newBivariateStatistics = this.collectFittestIndividualCounts(fittestValues, all_parents, prob_parents, newBivariateStatistics, true);
        newBivariateStatistics = this.normalizeProbabilities(all_parents, prob_parents, newBivariateStatistics, true);

        return newBivariateStatistics;
    }

    /**
     * Initializes bivariate (or univariate) statistics.
     *
     * @param parentName Name of parent variable. May be null: in this case, collects univariate statistics of child variable
     *                   (i.e. the object that is calling this method).
     * @param parentValues HashSet of values that parent variable can assume. May be null: in this case, collects
     *                     univariate statistics of child variable (i.e. the object that is calling this method).
     * @return A HashMap where the key is a Combination object containing (parent, child) tuples, and the value the
     * number of occurrences of that combination of values. Note: uses laplace correction, what means it starts this
     * structure with 1 in each combination of values.
     */
    public HashMap<Combination, Double> initializeCombinationsOfValues(
            String parentName, HashSet<String> parentValues, HashMap<String, HashMap<String, ArrayList<Integer>>> table, Boolean isAdeterministicParent) {
        HashMap<Combination, Double> local = new HashMap<>();

        HashSet<String> childValues = new HashSet<>();
        childValues.addAll(table.get(this.variable_name).keySet());

        if(parentName != null) {
            // TODO testing this!!!
            double addValue = isAdeterministicParent? 0 : 1;

            for(String parentVal : parentValues) {
                for(String childVal : childValues) {
                    HashMap<String, String> pairs = new HashMap<>();
                    pairs.put(this.variable_name, childVal);
                    pairs.put(parentName, parentVal);
                    local.put(new Combination(pairs), addValue);  // TODO using laplace correction!
                }
            }
        } else {
            for(String childVal : childValues) {
                HashMap<String, String> pairs = new HashMap<>();
                pairs.put(this.variable_name, childVal);
                local.put(new Combination(pairs), 1.0);  // TODO using laplace correction!
            }
        }
        return local;
    }

    /**
     * Based on the fittest population, collects bivariate statistics, as well as univariate statistics for current variable.
     *
     * @param fittestValues A HashMap where the key is the variable name and the value an array of values assumed by
     *                      the fittest population.
     * @param parentVariables Parent variables of this variable (i.e. object calling this method).
     * @param bivariateStatistics A HashMap where the key is the name of a variable and the value another HashMap, this
     *                            time where the key is the combination of pairwise values, and the value the count
     *                            (note: not probability) of individuals with those pairwise values.
     * @param computeUnivariate Whether to also collect univariate statistics
     * @return bivariateStatistics updated based on the fittest population. Contains counts of individuals, not probabilities.
     */
    public HashMap<String, HashMap<Combination, Double>> collectFittestIndividualCounts(
            HashMap<String, ArrayList<String>> fittestValues, HashSet<String> parentVariables, HashSet<String> prob_parents,
            HashMap<String, HashMap<Combination, Double>> bivariateStatistics, boolean computeUnivariate) {
        // adds counter to each fittest individual, for each bivariate distribution
        int n_fittest = fittestValues.get((String)fittestValues.keySet().toArray()[0]).size();
        for(int i = 0; i < n_fittest; i++) {
            String childValue = fittestValues.get(this.variable_name).get(i);

            // TODO testing new code!
            if(String.valueOf(childValue).equals("null")) {
                continue;
            }

            // collects bivariate statistics
            for(String parent : parentVariables) {
                String parentValue = fittestValues.get(parent).get(i);

                HashMap<String, String> valuePairs = new HashMap<>();
                valuePairs.put(this.variable_name, childValue);
                valuePairs.put(parent, parentValue);

                HashMap<Combination, Double> thisParentDict = bivariateStatistics.get(parent);

                Combination localComb = new Combination(valuePairs);
                thisParentDict.put(localComb, thisParentDict.get(localComb) + 1);

                bivariateStatistics.put(parent, thisParentDict);
            }
            // collects univariate statistics
            if(computeUnivariate) {
                HashMap<String, String> valuePairs = new HashMap<>();
                valuePairs.put(this.variable_name, childValue);

                HashMap<Combination, Double> thisDict = bivariateStatistics.get(this.variable_name);
                Combination localComb = new Combination(valuePairs);
                thisDict.put(localComb, thisDict.get(localComb) + 1);

                bivariateStatistics.put(this.variable_name, thisDict);
            }
        }
        return bivariateStatistics;
    }

    /**
     * Normalizes probabilities of bivariate statistics (also univariate statistics for current variable).
     *
     * @param all_parents A list of parent variables.
     * @param bivariateStatistics A HashMap where the key is the name of a variable and the value another HashMap, this
     *                            time where the key is the combination of pairwise values, and the value the count
     *                            (note: not probability) of individuals with those pairwise values.
     * @param computeUnivariate Whether to also normalize univariate counts.
     * @return bivariateStatistics, but this time are probabilities instead of counts. Probabilities are also normalized
     *         by parent values.
     */
    public HashMap<String, HashMap<Combination, Double>> normalizeProbabilities(
            HashSet<String> all_parents, HashSet<String> prob_parents, HashMap<String, HashMap<Combination, Double>> bivariateStatistics,
            boolean computeUnivariate) {
        // normalizes probabilities
        // normalizes bivariate probabilities
        for(String parent : all_parents) {
            HashMap<Combination, Double> local = bivariateStatistics.get(parent);

            HashMap<String, Double> byParent = new HashMap<>();
            for(Combination pairs : local.keySet()) {
                byParent.put(
                        pairs.getPairs().get(parent),
                        byParent.getOrDefault(pairs.getPairs().get(parent), 0.0) + local.get(pairs)
                );
            }

            for(Combination pairs : local.keySet()) {
                local.put(pairs, local.get(pairs) / byParent.get(pairs.getPairs().get(parent)));
            }

            bivariateStatistics.put(parent, local);
        }
        // normalizes univariate probabilities
        if(computeUnivariate) {
            HashMap<Combination, Double> local = bivariateStatistics.get(this.variable_name);
            double sum = 0.0;
            for(Double val : local.values()) {
                sum += val;
            }
            for(Combination key : local.keySet()) {
                local.put(key, local.get(key) / sum);
            }
            bivariateStatistics.put(this.variable_name, local);
        }
        return bivariateStatistics;
    }

    public Map<String, HashMap<Combination, Double>> getOldBivariateStatistics() {
        return this.oldBivariateStatistics;
    }

    public void setOldBivariateStatistics(HashMap<String, HashMap<Combination, Double>> bivariateStatistics) {
        this.oldBivariateStatistics = bivariateStatistics;
    }

    /**
     * Updates bivariate statistics with learning rate.
     *
     * @param newParents
     * @param learningRate
     * @param childUniqueValues
     * @param newBivariateStatistics
     * @param table
     * @return
     * @throws Exception
     */
    public HashMap<String, HashMap<Combination, Double>> updateWithLearningRate(
            HashSet<String> det_parents,
            HashSet<String> newParents,
            double learningRate,
            ArrayList<String> childUniqueValues,
            HashMap<String, ArrayList<String>> lastFittestValues,
            HashMap<String, HashMap<Combination, Double>> newBivariateStatistics,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table
    ) throws Exception {

        HashMap<Combination, Double> oldLocal = this.oldBivariateStatistics.get(this.variable_name);
        HashMap<Combination, Double> newLocal = newBivariateStatistics.get(this.variable_name);

        HashSet<String> prob_parents = new HashSet<>();
        prob_parents.addAll(newParents);
        prob_parents.removeAll(det_parents);

        // updates univariate statistics
        newBivariateStatistics.put(
                this.variable_name,
                this.updateUnivariateWithLearningRate(childUniqueValues, learningRate, oldLocal, newLocal)
        );

        if(newParents.size() > 0) {
            // updates bivariate statistics using learning rate
            HashMap<String, HashMap<Combination, Double>> updatedBivariateStatistics = new HashMap<>();
            for(String parent : newParents) {
                // if there are no bivariate statistics from last generation
                // this happens when a new probabilistic parent is introduced
                // in the current generation
                if(!this.oldBivariateStatistics.containsKey(parent)) {
                    HashSet<String> soloParent = new HashSet<String>() {{
                        add(parent);
                    }};
                    HashSet<String> temp = new HashSet<>();
                    temp.addAll(table.get(parent).keySet());
                    this.oldBivariateStatistics.put(parent, this.initializeCombinationsOfValues(parent, temp, table, det_parents.contains(parent)));
                    this.collectFittestIndividualCounts(lastFittestValues, soloParent, prob_parents, this.oldBivariateStatistics, false);
                    this.normalizeProbabilities(soloParent, prob_parents, this.oldBivariateStatistics, false);
                }
                HashMap<Combination, Double> local = new HashMap<>();
                HashMap<String, Double> sumByParentVal = new HashMap<>();

                if(this.oldBivariateStatistics.get(parent).keySet().size() != newBivariateStatistics.get(parent).keySet().size()) {
                    throw new Exception("Different number of combinations between last and current generation!");
                }

                for(Combination pair : this.oldBivariateStatistics.get(parent).keySet()) {
                    double oldProb = this.oldBivariateStatistics.get(parent).get(pair);
                    double newProb = newBivariateStatistics.get(parent).get(pair);

                    double newValue = Double.isNaN(newProb)? oldProb : (1 - learningRate) * oldProb + learningRate * newProb;

                    sumByParentVal.put(pair.getPairs().get(parent), sumByParentVal.getOrDefault(pair.getPairs().get(parent), 0.0) + newValue);

                    local.put(pair, newValue);
                }

                // normalizes
                for(Combination pair : this.oldBivariateStatistics.get(parent).keySet()) {
                    local.put(pair, local.get(pair) / sumByParentVal.get(pair.getPairs().get(parent)));
                }
                updatedBivariateStatistics.put(parent, local);
            }
            updatedBivariateStatistics.put(this.variable_name, newBivariateStatistics.get(this.variable_name));
            newBivariateStatistics = updatedBivariateStatistics;
        }
        return newBivariateStatistics;
    }

    private HashMap<Combination, Double> updateUnivariateWithLearningRate(
            ArrayList<String> uniqueValues, double learningRate,
            HashMap<Combination, Double> oldLocal,
            HashMap<Combination, Double> newLocal
    ) {
        // uses learning rate on univariate statistics
        double sum = 0.0;

        HashMap<Combination, Double> notNormalized = new HashMap<>();

        for(String childValue : uniqueValues) {
            HashMap<String, String> pair = new HashMap<>();
            pair.put(this.variable_name, childValue);
            Combination comb = new Combination(pair);
            double provProb = (1 - learningRate) * oldLocal.get(comb) + learningRate * newLocal.get(comb);
            sum += provProb;

            notNormalized.put(comb, provProb);
        }
        for(Combination comb : notNormalized.keySet()) {
            notNormalized.put(comb, notNormalized.get(comb) / sum);
        } // not normalized is normalized now. funny huh
        return notNormalized;
    }
}
