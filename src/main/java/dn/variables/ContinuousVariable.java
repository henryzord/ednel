package dn.variables;

import eda.individual.Individual;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.*;

/**
 * This class encodes a continuous variable, more precisely a normal distribution.
 */
public class ContinuousVariable extends AbstractVariable {

    protected ArrayList<HashMap<String, Float>> normalProperties;
    protected ArrayList<NormalDistribution> normalDistributions;
    protected MultivariateNormalDistribution mvNormalDistribution;
    protected double a_min, a_max, scale_init;


    private HashMap<String, Float> fromStringToProperty(String str) {
        String[] property = str.replaceAll("[\\(\\)\"]", "").split(",");

        HashMap<String, Float> thisProperty = new HashMap<>(property.length);

        for(int j = 0; j < property.length; j++) {
            String[] pair = property[j].split("=");
            thisProperty.put(pair[0], Float.valueOf(pair[1]));
        }
        return thisProperty;
    }

    public ContinuousVariable(String name, ArrayList<String> parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
                              ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt,
                              float learningRate, int n_generations) throws Exception {
        super(name, parents, table, values, probabilities, mt, learningRate, n_generations);

        normalProperties = new ArrayList<>(values.size());
        normalDistributions = new ArrayList<>(values.size());
        mvNormalDistribution = null;

        for(int i = 0; i < values.size(); i++) {
            if(values.get(i) != null){
                HashMap<String, Float> thisProperty = this.fromStringToProperty(values.get(i));

                a_min = thisProperty.get("a_min");
                a_max = thisProperty.get("a_max");
                scale_init = thisProperty.get("scale_init");

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
        if(this.mvNormalDistribution == null) {
            this.updateNormalDistributions(fittest);
        } else {
            updateMultivariateNormalDistribution(fittest);
        }
    }

    private void updateMultivariateNormalDistribution(Individual[] fittest) throws Exception {
        throw new Exception("not implemented yet!");
    }

    /**
     * Updates both the probability of the table entry, and the Gaussian of each entry (if any).
     * @param fittest Fittest individuals in the population, used to update the probabilities.
     * @throws Exception
     */
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

    protected HashMap<String, double[]> getContinuousVariablesValues(VariableStructure[] parents, Individual[] fittest) {
        HashMap<String, double[]> values = new HashMap<>(parents.length + 1);

        ArrayList<String> queryNames = new ArrayList<String>(parents.length + 1);
        for(int i = 0; i < parents.length; i++) {
            if(parents[i] instanceof ContinuousVariable) {
                queryNames.add(parents[i].getName());
            }
        }
        queryNames.add(this.getName());

        for(int i = 0; i < queryNames.size(); i++) {
            ArrayList<Double> tempValues = new ArrayList<>(fittest.length);
            for(int j = 0; j < fittest.length; j++) {
                String val = String.valueOf(fittest[j].getCharacteristics().get(queryNames.get(i))).toLowerCase();
                if(!val.equals("null")) {
                    tempValues.add(Double.parseDouble(val));
                }
            }
            Double[] rawValues = new Double[tempValues.size()];
            tempValues.toArray(rawValues);

            values.put(queryNames.get(i), ArrayUtils.toPrimitive(rawValues));
        }
        return values;
    }

    protected HashMap<String, String> getContinuousVariablesUniqueValues(HashMap<String, double[]> doubleValues) {

        Object[] keys = doubleValues.keySet().toArray();
        HashMap<String, String> descriptiveString = new HashMap<>(keys.length);

        for(int i = 0; i < keys.length; i++) {
            double[] fullValues = doubleValues.get((String)keys[i]);
            DescriptiveStatistics fS = new DescriptiveStatistics(fullValues);
            descriptiveString.put(
                (String)keys[i],
                String.format(
                    Locale.US,
                    "(loc=%01.6f,scale=%01.6f,a_min=%01.6f,a_max=%01.6f,scale_init=%01.6f)",
                    fS.getMean(), fS.getStandardDeviation(), this.a_min, this.a_max, this.scale_init
                )
            );
        }
        return descriptiveString;
    }

    private MultivariateNormalDistribution generateMvNormalDistribution(HashMap<String, double[]> doubleValues) {
        int min_size = Integer.MAX_VALUE;
        for(String key : doubleValues.keySet()) {
            double[] local_values = doubleValues.get(key);
            min_size = Math.min(local_values.length, min_size);
        }

        Object[] keys = doubleValues.keySet().toArray();
        double[] reducedMeans = new double [keys.length];
        double[][] data = new double[keys.length][min_size];

        for(int i = 0; i < keys.length; i++) {
            double[] fullValues = doubleValues.get((String)keys[i]);
            data[i] = Arrays.copyOfRange(fullValues, 0, min_size);
            DescriptiveStatistics rS = new DescriptiveStatistics(data[i]);
            reducedMeans[i] = rS.getMean();
        }
        double[][] covMatrix = new double[keys.length][keys.length];
        Covariance cov = new Covariance();
        for(int i = 0; i < keys.length; i++) {
            for(int j = 0; j < keys.length; j++) {
                covMatrix[i][j] = cov.covariance(data[i], data[j]);
            }
        }
        MultivariateNormalDistribution mv = new MultivariateNormalDistribution(mt, reducedMeans, covMatrix);
        return mv;
    }

    @Override
    public void updateStructure(VariableStructure[] parents, Individual[] fittest) throws Exception {
        super.updateStructure(parents, fittest);

        int countDiscrete = countDiscreteParents(parents);

        // all parents are discrete variables
        if(countDiscrete == parents.length) {
             throw new Exception("not implemented yet!");
        // multivariate normal distribution
        } else if ((countDiscrete == 0) && (parents.length == 1)) {
            HashMap<String, double[]> doubleValues = this.getContinuousVariablesValues(parents, fittest);
            HashMap<String, String> descriptiveString = this.getContinuousVariablesUniqueValues(doubleValues);

            HashMap<String, HashSet<String>> combRawOver = new HashMap<>(parents.length + 1);
            for(int i = 0; i < parents.length; i++) {
                HashSet<String> rawUniqueValues = new HashSet<>();
                rawUniqueValues.add(null);
                rawUniqueValues.add(descriptiveString.get(parents[i].getName()));
                combRawOver.put(parents[i].getName(), rawUniqueValues);
            }
            HashSet<String> rawUniqueValues = new HashSet<>();
            rawUniqueValues.add(null);
            rawUniqueValues.add(descriptiveString.get(this.getName()));
            combRawOver.put(this.getName(), rawUniqueValues);

            this.uniqueValues = new HashSet<>();
            this.uniqueValues.addAll(rawUniqueValues);

            this.updateTableEntries(combRawOver);
            
            // now place normal distribution in correct position
            this.mvNormalDistribution = this.generateMvNormalDistribution(doubleValues);

            // null, not null must be 1-st row of table (starting at zero)
            this.normalDistributions.clear();
            this.normalProperties.clear();

            HashMap<String, String> lastStart = new HashMap<>();
            for(int i = 0; i < parents.length; i++) {
                lastStart.put(parents[i].getName(), null);
            }
            HashSet<Integer> idx = this.getSetOfIndices(lastStart, descriptiveString.get(this.getName()),true);
            int index = (Integer)idx.toArray()[0];

            for(int i = 0; i < this.values.size(); i++) {
                if(i == index) {
                    DescriptiveStatistics fS = new DescriptiveStatistics(doubleValues.get(this.getName()));
                    this.normalDistributions.add(new NormalDistribution(fS.getMean(), fS.getStandardDeviation()));
                    this.normalProperties.add(this.fromStringToProperty(descriptiveString.get(this.getName())));
                } else {
                    this.normalDistributions.add(null);
                    this.normalProperties.add(null);
                }
            }
        } else {
            throw new Exception("not implemented yet!");
        }
    }
}
