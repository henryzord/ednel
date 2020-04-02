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
    protected double a_min;
    protected double a_max;
    protected double scale_init;

    public ContinuousVariable(String name, ArrayList<String> parents_names, HashMap<String, Boolean> isParentContinuous,
                              HashMap<String, HashMap<String, ArrayList<Integer>>> table,
                              ArrayList<String> values, ArrayList<Double> probabilities,
                              MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        super(name, parents_names, isParentContinuous, table,
            null, null, null, probabilities, mt, learningRate, n_generations
        );

        this.values = new ArrayList<>(values.size());
        this.uniqueValues = new HashSet<>();
        this.uniqueShadowvalues = new HashSet<>();

        for(int i = 0; i < values.size(); i++) {
            if(values.get(i) != null) {
                HashMap<String, Double> properties = this.fromStringToProperty(values.get(i));
                this.a_max = properties.get("a_max");
                this.a_min = properties.get("a_min");
                this.scale_init = properties.get("scale_init");

                Shadowvalue sv;
                if(properties.containsKey("covariance_matrix")) {
                    throw new Exception("not implemented yet!");
//                    sv = new ShadowMultivariateNormalDistribution(null, null);  // TODO implement
                } else {
                    sv = new ShadowNormalDistribution(this.mt, properties);
                }
                this.values.add(sv);
                this.uniqueValues.add(sv.toString());
                this.uniqueShadowvalues.add(sv);
            } else {
                this.values.add(null);
                this.uniqueValues.add(null);
                this.uniqueShadowvalues.add(null);
            }
        }
    }

    /**
     * Converts a string that denotes a normal distribution (or a multivariate normal distribution)
     * into a dictionary.
     * @param str The string.
     * @return A dictionary.
     */
    private HashMap<String, Double> fromStringToProperty(String str) {
        String[] property = str.replaceAll("[\\(\\)\"]", "").split(",");

        HashMap<String, Double> thisProperty = new HashMap<>(property.length);

        for(int j = 0; j < property.length; j++) {
            String[] pair = property[j].split("=");
            thisProperty.put(pair[0], Double.valueOf(pair[1]));
        }
        return thisProperty;
    }

    @Override
    public void updateProbabilities(Individual[] fittest) throws Exception {
        super.updateProbabilities(fittest);
        throw new Exception("now update shadow unique values and values of normal distributions!");
        // TODO as well as multivariate normal distribution

//            HashMap<String, double[]> doubleValues = this.getContinuousVariablesValues(parents, fittest);
//            HashMap<String, String> descriptiveString = this.getContinuousVariablesUniqueValues(doubleValues);
//
//            HashMap<String, HashSet<String>> combRawOver = new HashMap<>(parents.length + 1);
//            for(int i = 0; i < parents.length; i++) {
//                HashSet<String> rawUniqueValues = new HashSet<>();
//                rawUniqueValues.add(null);
//                rawUniqueValues.add(descriptiveString.get(parents[i].getName()));
//                combRawOver.put(parents[i].getName(), rawUniqueValues);
//            }
//            HashSet<String> rawUniqueValues = new HashSet<>();
//            rawUniqueValues.add(null);
//            rawUniqueValues.add(descriptiveString.get(this.getName()));
//            combRawOver.put(this.getName(), rawUniqueValues);
//
//            this.uniqueValues = new HashSet<>();
//            this.uniqueValues.addAll(rawUniqueValues);
//
//            this.updateTableEntries(combRawOver);
//
//            // now place normal distribution in correct position
//            this.mvNormalDistribution = this.generateMvNormalDistribution(doubleValues);
//
//            // null, not null must be 1-st row of table (starting at zero)
//            this.normalDistributions.clear();
//            this.normalProperties.clear();
//
//            HashMap<String, String> lastStart = new HashMap<>();
//            for(int i = 0; i < parents.length; i++) {
//                lastStart.put(parents[i].getName(), null);
//            }
//            HashSet<Integer> idx = this.getSetOfIndices(lastStart, descriptiveString.get(this.getName()),true);
//            int index = (Integer)idx.toArray()[0];
//
//            for(int i = 0; i < this.values.size(); i++) {
//                if(i == index) {
//                    DescriptiveStatistics fS = new DescriptiveStatistics(doubleValues.get(this.getName()));
//                    this.normalDistributions.add(new NormalDistribution(fS.getMean(), fS.getStandardDeviation()));
//                    this.normalProperties.add(this.fromStringToProperty(descriptiveString.get(this.getName())));
//                } else {
//                    this.normalDistributions.add(null);
//                    this.normalProperties.add(null);
//                }
//            }
//        } else {
//            throw new Exception("not implemented yet!");
//        }
    }

//    /**
//     * Updates both the probability of the table entry, and the Gaussian of each entry (if any).
//     * @param fittest Fittest individuals in the population, used to update the probabilities.
//     * @throws Exception
//     */
//    private void updateNormalDistributions(Individual[] fittest) throws Exception {
//        HashSet<Integer> nullSet = new HashSet<>(this.values.size());
//        nullSet.addAll(this.table.get(this.name).get(null));
//
//        HashMap<Integer, DescriptiveStatistics> sampledValues = new HashMap<>(this.values.size());
//
//        for(int i = 0; i < fittest.length; i++) {
//            // local characteristics
//            HashMap<String, String> localCars = fittest[i].getCharacteristics();
//            // this variable is not set for this individual; continue
//            // if this individual does not use this variable, then do not do anything
//            if(localCars.get(this.name) == null) {
//                continue;
//            }
//
//            HashSet<Integer> nullSetLocal = (HashSet<Integer>) nullSet.clone();
//            // gets all indices where variable is null
//            HashSet<Integer> parentIndices = getSetOfIndices(
//                    fittest[i].getCharacteristics(), null, false
//            );
//            parentIndices.removeAll(nullSetLocal);  // now parentIndices has all the indices which should be updated
//            if(parentIndices.size() != 1) {
//                throw new Exception("selection is wrong!");
//            }
//            int onlyIndex = (int) parentIndices.toArray()[0];
//
//            if(!sampledValues.containsKey(onlyIndex)) {
//                DescriptiveStatistics stats = new DescriptiveStatistics();
//                stats.addValue((Float.valueOf((String)localCars.get(this.name))));
//                sampledValues.put(onlyIndex, stats);
//            } else {
//                sampledValues.get(onlyIndex).addValue((Float.valueOf((String)localCars.get(this.name))));
//            }
//        }
//
//        for(Object key: sampledValues.keySet().toArray()) {
//            float loc = (float)this.normalDistributions.get((Integer)key).getMean();
//
//            float diff = loc - (float)sampledValues.get(key).getMean();
//            loc = loc + (diff * this.learningRate);
//            float scale = Math.max(
//                    0,
//                    (float)this.normalDistributions.get((Integer)key).getStandardDeviation() -
//                            (Float.valueOf(this.normalProperties.get((Integer)key).get("scale_init")) / this.n_generations)
//            );
//            this.normalProperties.get((Integer)key).put("loc", loc);
//            this.normalProperties.get((Integer)key).put("scale", scale);
//
//            this.normalDistributions.set((Integer)key, new NormalDistribution(this.mt, loc, scale));
//        }
//    }

//    protected HashMap<String, double[]> getContinuousVariablesValues(VariableStructure[] parents, Individual[] fittest) {
//        HashMap<String, double[]> values = new HashMap<>(parents.length + 1);
//
//        ArrayList<String> queryNames = new ArrayList<String>(parents.length + 1);
//        for(int i = 0; i < parents.length; i++) {
//            if(parents[i] instanceof ContinuousVariable) {
//                queryNames.add(parents[i].getName());
//            }
//        }
//        queryNames.add(this.getName());
//
//        for(int i = 0; i < queryNames.size(); i++) {
//            ArrayList<Double> tempValues = new ArrayList<>(fittest.length);
//            for(int j = 0; j < fittest.length; j++) {
//                String val = String.valueOf(fittest[j].getCharacteristics().get(queryNames.get(i))).toLowerCase();
//                if(!val.equals("null")) {
//                    tempValues.add(Double.parseDouble(val));
//                }
//            }
//            Double[] rawValues = new Double[tempValues.size()];
//            tempValues.toArray(rawValues);
//
//            values.put(queryNames.get(i), ArrayUtils.toPrimitive(rawValues));
//        }
//        return values;
//    }

//    protected HashMap<String, String> getContinuousVariablesUniqueValues(HashMap<String, double[]> doubleValues) {
//
//        Object[] keys = doubleValues.keySet().toArray();
//        HashMap<String, String> descriptiveString = new HashMap<>(keys.length);
//
//        for(int i = 0; i < keys.length; i++) {
//            double[] fullValues = doubleValues.get((String)keys[i]);
//            DescriptiveStatistics fS = new DescriptiveStatistics(fullValues);
//            descriptiveString.put(
//                (String)keys[i],
//                String.format(
//                    Locale.US,
//                    "(loc=%01.6f,scale=%01.6f,a_min=%01.6f,a_max=%01.6f,scale_init=%01.6f)",
//                    fS.getMean(), fS.getStandardDeviation(), this.a_min, this.a_max, this.scale_init
//                )
//            );
//        }
//        return descriptiveString;
//    }

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
    public void updateStructure(AbstractVariable[] parents, Individual[] fittest) throws Exception {
        super.updateStructure(parents, fittest);

        for(AbstractVariable par : parents) {
            this.parents_names.add(par.getName());
            this.isParentContinuous.put(par.getName(), par instanceof ContinuousVariable);
        }

        HashMap<String, HashSet<String>> eoUniqueValues = Combinator.getUniqueValuesFromVariables(parents);
        // adds unique values of this variable
        eoUniqueValues.put(this.getName(), this.getUniqueValues());

        this.updateTable(eoUniqueValues);
    }

    public void updateUniqueValues(Individual[] fittest) {
        int counter = 0;
        for(Individual fit : fittest) {
            if(fit.getCharacteristics().get(this.getName()) != null) {
                counter += 1;
            }
        }
        double[] values = new double [counter];
        counter = 0;
        for(Individual fit : fittest) {
            if(fit.getCharacteristics().get(this.getName()) != null) {
                values[counter] = Double.parseDouble(fit.getCharacteristics().get(this.getName()));
                counter += 1;
            }
        }
        DescriptiveStatistics ds = new DescriptiveStatistics(values);

        String descriptiveString = String.format(
            Locale.US,
            "(loc=%01.6f,scale=%01.6f,a_min=%01.6f,a_max=%01.6f,scale_init=%01.6f)",
            ds.getMean(), ds.getStandardDeviation(), this.a_min, this.a_max, this.scale_init
        );

        this.uniqueValues.clear();
        this.uniqueValues.add(null);
        this.uniqueValues.add(descriptiveString);
    }
}
