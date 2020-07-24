package ednel.network.variables;

import ednel.eda.individual.Individual;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.util.*;

/**
 * This class encodes a continuous variable, more precisely a normal distribution.
 */
public class ContinuousVariable extends AbstractVariable {
    protected double a_min;
    protected double a_max;
    protected double loc_init;
    protected double scale;
    protected double scale_init;

    public ContinuousVariable(String name, HashSet<String> parents_names, HashMap<String, Boolean> isParentContinuous,
                              HashMap<String, HashMap<String, ArrayList<Integer>>> table,
                              ArrayList<String> values, ArrayList<Double> probabilities,
                              MersenneTwister mt, double learningRate, int n_generations, int max_parents) throws Exception {

        super(name, parents_names, isParentContinuous, table,
            null, null, null, probabilities, mt, learningRate, n_generations, max_parents
        );

        this.values = new ArrayList<>(values.size());
        this.uniqueValues = new HashSet<>();
        this.uniqueShadowvalues = new HashSet<>();

        for(int i = 0; i < values.size(); i++) {
            if(!values.get(i).equals("null")) {
                HashMap<String, Double> properties = this.fromStringToProperty(values.get(i));
                this.a_max = properties.get("a_max");
                this.a_min = properties.get("a_min");
                this.scale = properties.get("scale");
                this.loc_init = properties.get("loc");
                this.scale_init = properties.get("scale_init");

                Shadowvalue sv;
                if(properties.containsKey("means")) {
                    throw new Exception("not implemented yet!");  // TODO implement
                } else {
                    sv = new ShadowNormalDistribution(this.mt, properties);
                }
                this.values.add(sv);
                this.uniqueValues.add(sv.toString());
                this.uniqueShadowvalues.add(sv);
            } else {
                Shadowvalue sv = new Shadowvalue(
                    String.class.getMethod("toString"),
                    "null"
                );

                this.values.add(sv);
                this.uniqueValues.add(sv.toString());
                this.uniqueShadowvalues.add(sv);
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
    public void updateProbabilities(HashMap<String, ArrayList<String>> fittestValues, Individual[] fittest) throws Exception {
        super.updateProbabilities(fittestValues, fittest);
        // updates shadow values
        this.uniqueShadowvalues = new HashSet<>(this.values);
        this.uniqueValues = new HashSet<>();
        for(Shadowvalue val : this.uniqueShadowvalues) {
            this.uniqueValues.add(val.toString());
        }
    }

    /**
     * Updates normal distributions for positions that do have a normal distribution.
     * Null positions do not need to be updated.
     * Method attached to updateProbabilities.
     */
    protected void updateNormalDistributions(
            HashSet<Integer> parentIndices, double[][][] dda, int[][] ddc, HashMap<String, Integer> ddd) throws Exception {
        HashSet<Integer> notNullSet = new HashSet<>(this.notNullLoc(this.getName()));

        // now nnSet has the indices that match parent values & this variable is not null
        notNullSet.retainAll(parentIndices);
        if(notNullSet.size() > 1) {
            throw new Exception("unexpected behavior!");
        }
        int idx = (int)notNullSet.toArray()[0];

        // number of other variables that are also continuous
        int multivariate = 0;
        Object[] variableNames = ddd.keySet().toArray();
        boolean[] insert = new boolean[ddd.size()];
        int min_size = Integer.MAX_VALUE;
        for(int i = 0; i < variableNames.length; i++) {
            if(ddc[ddd.get((String)variableNames[i])][idx] > 0) {
                multivariate += 1;
                insert[i] = true;

                min_size = Math.min(
                        min_size,
                        ddc[ddd.get((String)variableNames[i])][idx]
                );
            } else {
                insert[i] = false;
            }
        }
        if((multivariate > 1) && (min_size > 1)) {
            try {
                this.addMultivariateDistribution(idx, multivariate, min_size, ddd, dda, insert, variableNames);
            } catch (SingularMatrixException sme) {
                this.addUnivariateDistribution(idx, ddd, dda, ddc);
            }
        } else {
            this.addUnivariateDistribution(idx, ddd, dda, ddc);
        }
    }

    protected void addUnivariateDistribution(int idx, HashMap<String, Integer> ddd, double[][][] dda, int[][] ddc) throws Exception {
        double[] data = Arrays.copyOfRange(
                dda[ddd.get(this.getName())][idx],
                0,
                ddc[ddd.get(this.getName())][idx]
        );
        DescriptiveStatistics ds = new DescriptiveStatistics(data);
        double loc = ds.getMean();
        double scale = (this.scale - (this.scale_init / this.n_generations));
        // this is an extreme case, where the normal distribution is reset to its initial value
        if(Double.isNaN(loc)) {
            loc = this.loc_init;
        }

        this.values.set(
                idx,
                new ShadowNormalDistribution(this.mt, loc, scale, this.a_min, this.a_max, this.scale_init)
        );
    }

    protected void addMultivariateDistribution(int idx, int n_variables, int n_data, HashMap<String, Integer> ddd, double[][][] dda, boolean[] insert, Object[] keys) throws org.apache.commons.math3.linear.SingularMatrixException, NoSuchMethodException {
        double[][] toProcess = new double[n_variables][];

        int adder = 0;
        for(int i = 0; i < ddd.size(); i++) {
            if(insert[i]) {
                toProcess[adder] = Arrays.copyOfRange(
                        dda[ddd.get((String)keys[i])][idx],
                        0,
                        n_data
                );
                adder += 1;
            }

        }
        this.values.set(
            idx,
            new ShadowMultivariateNormalDistribution(
                    this.mt,
                    toProcess,
                    this.a_min,
                    this.a_max,
                    this.scale_init
            )
        );
    }

    public void updateUniqueValues(HashMap<String, ArrayList<String>> fittest) {
        int counter = 0;
        for(String fit : fittest.get(this.getName())) {
            if(fit != null) {
                counter += 1;
            }
        }
        double[] values = new double [counter];
        counter = 0;
        for(String fit : fittest.get(this.getName())) {
            if(fit != null) {
                values[counter] = Double.parseDouble(fit);
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
        this.uniqueValues.add("null");
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
