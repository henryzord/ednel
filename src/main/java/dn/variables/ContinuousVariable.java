package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

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

    @Override
    public void updateProbabilities(Individual[] fittest) throws Exception {
        super.updateProbabilities(fittest);
        this.updateNormalDistributions(fittest);
    }

    private void updateNormalDistributions(Individual[] fittest) throws Exception {
        HashSet<Integer> nullSet = new HashSet<>(this.values.size());
        nullSet.addAll(this.table.get(this.name).get(null));

        HashMap<Integer, DescriptiveStatistics> sampledValues = new HashMap<>(this.values.size());

        for(int i = 0; i < fittest.length; i++) {
            // local characteristics
            HashMap<String, String> localCars = fittest[i].getCharacteristics();
            // this variable is not set for this individual; continue
            // if this individual does not use this variable, then do not do anything
            if(localCars.get(this.name) == null) {
                continue;
            }

            HashSet<Integer> nullSetLocal = (HashSet<Integer>) nullSet.clone();
            // gets all indices where variable is null
            HashSet<Integer> parentIndices = getSetOfIndices(
                    fittest[i].getCharacteristics(), null, false
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
