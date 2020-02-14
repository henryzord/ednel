package dn.variables;

import eda.Individual;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.classifiers.rules.DecisionTableHashKey;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutionException;

public class ContinuousVariable extends AbstractVariable {

    protected ArrayList<HashMap<String, Float>> normalProperties;
    protected ArrayList<NormalDistribution> normalDistributions;


    public ContinuousVariable(String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
                              ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt,
                              float learningRate, int n_generations) throws Exception {
        super(name, parents, table, values, probabilities, mt, learningRate, n_generations);

        normalProperties = new ArrayList<>(values.size());
        normalDistributions = new ArrayList<>(values.size());

        for(int i = 0; i < values.size(); i++) {
            if(values.get(i) != null){
                String[] property = values.get(i).replaceAll("[\\(\\)\"]", "").split(",");

                HashMap<String, Float> thisProperty = new HashMap<>(property.length);

                for(int j = 0; j < property.length; j++) {
                    String[] pair = property[j].split("=");
                    thisProperty.put(pair[0], Float.valueOf(pair[1]));
                }
                normalProperties.add(thisProperty);
                normalDistributions.add(
                        new NormalDistribution(this.mt, thisProperty.get("loc"), thisProperty.get("scale"))
                );

            } else {
                normalProperties.add(null);
                normalDistributions.add(null);
            }
        }
    }

    @Override
    public String[] unconditionalSampling(int sample_size) throws Exception {
        throw new Exception("not implemented yet!");
    }

    @Override
    public String conditionalSampling(HashMap<String, String> lastStart) throws Exception {
        int idx = this.conditionalSamplingIndex(lastStart);

        if(this.values.get(idx) != null) {
            HashMap<String, Float> thisNormal = normalProperties.get(idx);
            NormalDistribution nd = normalDistributions.get(idx);
            double sampled = Math.max(
                    thisNormal.get("a_min"),
                    Math.min(thisNormal.get("a_max"), nd.sample())
            );
            return String.valueOf(sampled);
        }
        return null;
    }

    public void updateProbabilities(Individual[] population, Integer[] sortedIndices, float selectionShare) throws Exception {
        ArrayList<Float> occurs = new ArrayList<>();
        for(int i = 0; i < probabilities.size(); i++) {
            occurs.add((float)0.0);
        }

        int to_select = Math.round(selectionShare * sortedIndices.length);

        // gets the count of occurrences
        for(int i = 0; i < to_select; i++) {
            HashSet<Integer> parentIndices = this.getSetOfIndices(
                    population[sortedIndices[i]].getCharacteristics(),
                    null,
                    false
            );
            HashSet<Integer> nullIndices = this.getSetOfIndices(
                    population[sortedIndices[i]].getCharacteristics(),
                    null,
                    true
            );

            parentIndices.retainAll(nullIndices);

            for(Object index : parentIndices.toArray()) {
                occurs.set((int)index, occurs.get((int)index) + 1);
            }
        }

//        throw new Exception("Check probabilities before updating normal distribution!");

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>();
        for(Object value : this.getUniqueValues().toArray()) {
            HashMap<String, String> local = new HashMap<>();
            local.put(this.name, null);
            combinations.add(local);
        }
        for(String parent : this.parents) {
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

        for(int i = 0; i < combinations.size(); i++) {
            int[] indices = this.getIndices(combinations.get(i), null, false);
            float sum = 0;
            for(int j = 0; j < indices.length; j++) {
                sum += occurs.get(indices[j]);
            }

            // updates probabilities using learning rate and relative frequencies
            for(int j = 0; j < indices.length; j++) {
                this.probabilities.set(
                        indices[j],
                        (float) ((1.0 - this.learningRate) * this.probabilities.get(indices[j]) +
                                this.learningRate * occurs.get(indices[j]) / sum)

                );
            }
        }

        for(int i = 0; i < probabilities.size(); i++) {
            if(Double.isNaN(probabilities.get(i))) {
                probabilities.set(i, (float)0);
//                throw new Exception("found NaN value!");  // TODO throw away this code later
            }
        }

        this.updateNormalDistributions(population, sortedIndices, selectionShare);
    }

    private void updateNormalDistributions(Individual[] population, Integer[] sortedIndices, float selectionShare) throws Exception {
        int to_select = Math.round(selectionShare * sortedIndices.length);

        HashSet<Integer> nullSet = new HashSet<>(this.values.size());
        nullSet.addAll(this.table.get(this.name).get(null));

        HashMap<Integer, DescriptiveStatistics> sampledValues = new HashMap<>(this.values.size());

        for(int i = 0; i < to_select; i++) {
            HashMap<String, String> localCars = population[sortedIndices[i]].getCharacteristics();
            // this variable is not set for this individual; continue
            if(localCars.get(this.name) == null) {
                continue;
            }

            HashSet<Integer> nullSetLocal = (HashSet<Integer>) nullSet.clone();
            // gets all indices where variable is null
            HashSet<Integer> parentIndices = getSetOfIndices(
                    population[sortedIndices[i]].getCharacteristics(), null, false
            );
            parentIndices.removeAll(nullSetLocal);  // now parentIndices has all the indices which should be updated
            if(parentIndices.size() != 1) {
                throw new Exception("selection is wrong!");
            }
            int onlyIndex = (int) parentIndices.toArray()[0];

            if(!sampledValues.containsKey(onlyIndex)) {
                DescriptiveStatistics stats = new DescriptiveStatistics();
                stats.addValue((Float.valueOf((String)localCars.get(this.name))));
                sampledValues.put(onlyIndex, stats);
            } else {
                sampledValues.get(onlyIndex).addValue((Float.valueOf((String)localCars.get(this.name))));
            }
        }

        for(Object key: sampledValues.keySet().toArray()) {
            float loc = (float)this.normalDistributions.get((Integer)key).getMean();

            float diff = loc - (float)sampledValues.get(key).getMean();
            loc = loc + (diff * this.learningRate);
            float scale = Math.max(
                    0,
                    (float)this.normalDistributions.get((Integer)key).getStandardDeviation() -
                            (Float.valueOf(this.normalProperties.get((Integer)key).get("scale_init")) / this.n_generations)
            );
            this.normalProperties.get((Integer)key).put("loc", loc);
            this.normalProperties.get((Integer)key).put("scale", scale);

            this.normalDistributions.set((Integer)key, new NormalDistribution(this.mt, loc, scale));
        }
    }

}
