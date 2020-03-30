package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;

/**
 * This class encodes a continuous variable, more precisely a normal distribution.
 */
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

    @Override
    public void updateStructure(VariableStructure[] parents, Individual[] fittest) throws Exception {
        throw new Exception("not implemented yet!");

//        super.updateStructure(parents, fittest);
//
//        int countDiscrete = countDiscreteParents(parents);
//
//        if(countDiscrete == parents.length) {
//            // all parents are discrete variables
//             throw new Exception("not implemented yet!");
//        } else if (countDiscrete == 0) {
//            // multivariate normal distribution
//
//            String[] queryNames = new String [parents.length + 1];
//            for(int i = 0; i < parents.length; i++) {
//                queryNames[i] = parents[i].getName();
//            }
//            queryNames[parents.length] = this.getName();
//            int min_size = fittest.length + 1;
//
//            for(int i = 0; i < queryNames.length; i++) {
//                int cur_size = 0;
//                for(int j = 0; j < fittest.length; j++) {
//                    HashMap<String, String> characteristics = fittest[j].getCharacteristics();
//                    String val = String.valueOf(characteristics.get(queryNames[i]));
//
//                    cur_size += (!val.toLowerCase().equals("null"))? 1 : 0;
//                }
//                min_size = Math.min(min_size, cur_size);
//            }
//            if(min_size == 0) {
//                throw new Exception("remove other variable from parents.");
//            }
//
//            double[][] values = new double[queryNames.length][];
//            for(int i = 0; i < queryNames.length; i++) {
//                values[i] = new double [min_size];
//
//                int counter = 0;
//                for(int j = 0; j < fittest.length; j++) {
//                    HashMap<String, String> characteristics = fittest[j].getCharacteristics();
//                    String val = String.valueOf(characteristics.get(queryNames[i]));
//
//                    if(!val.toLowerCase().equals("null")) {
//                        values[i][counter] = Double.parseDouble(val);
//                        counter += 1;
//                    }
//
//                }
//            }
//
//            Covariance cov = new Covariance();
//            double covariance = cov.covariance(values[0], values[1]);
//            RealMatrix covarianceMatrix = cov.getCovarianceMatrix();
//
//            int z = 0;
//
//
//            // TODo implement!
//            throw new Exception("not implemented yet!");
//        } else {
//            throw new Exception("not implemented yet!");
//        }
    }

    public static void main(String[] args) {
        double[] x = {0, 1, 2, 3, 4, 5};
        double[] y = {6, 7, 8, 9, 10, 11};

        Covariance cov = new Covariance();
        double covariance = cov.covariance(x, y);
        RealMatrix covarianceMatrix = cov.getCovarianceMatrix();



        int z = 0;
    }

}
