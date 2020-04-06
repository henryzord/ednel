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

        ArrayList<String> uniqueArray = new ArrayList<>();
        for(Shadowvalue val : this.values) {
            uniqueArray.add(val.toString());
        }
        this.uniqueValues = new HashSet<>(uniqueArray);
        this.uniqueShadowvalues = new HashSet<>();
        for(Shadowvalue val : this.values) {
            if(this.uniqueValues.contains(val.toString())) {
                this.uniqueShadowvalues.add(val);
            }
        }
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

    public double getMinValue() {
        return a_min;
    }

    public double getMaxValue() {
        return a_max;
    }

    public double getInitialStandardDeviation() {
        return scale_init;
    }
}
