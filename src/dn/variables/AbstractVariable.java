package dn.variables;

import eda.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.omg.CORBA.INTERNAL;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public abstract  class AbstractVariable {
    protected String name;
    protected String[] parents;
    protected HashMap<String, HashMap<String, ArrayList<Integer>>> table;
    protected ArrayList<Float> probabilities;
    protected MersenneTwister mt;
    protected ArrayList<String> values;
    protected HashSet<String> uniqueValues;

    public AbstractVariable(
            String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt) throws Exception {
        this.name = name;
        this.parents = parents;
        this.table = table;
        this.values = values;
        this.probabilities = probabilities;
        this.mt = mt;

        for(int i = 0; i < this.values.size(); i++) {
            if(this.values.get(i).toLowerCase().equals("null")) {
                this.values.set(i, null);
            }
        }
        this.uniqueValues = new HashSet<>(this.values);
    }

    public abstract String[] unconditionalSampling(int sample_size) throws Exception;

    /**
     * Get indices in the table object that correspond to a given assignment of values to the set
     * of parents for this variable.
     * @return
     */
    protected int[] getIndices(HashMap<String, String> conditions) {
        Set<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        ArrayList<Integer> localIndices;
        for(int i = 0; i < this.parents.length; i++) {
            localIndices = this.table.get(this.parents[i]).get(conditions.get(this.parents[i]));
            if(localIndices != null) {
                intersection.retainAll(new HashSet<>(localIndices));
            }
        }
        if(conditions.get(this.name) != null) {
            localIndices = this.table.get(this.name).get(conditions.get(this.name));
            intersection.retainAll(new HashSet<>(localIndices));
        }

        Object[] indices = intersection.toArray();
        int[] intIndices = new int [indices.length];
        for(int i = 0; i < indices.length; i++) {
            intIndices[i] = (int)indices[i];
        }

        return intIndices;
    }


    public String conditionalSampling(HashMap<String, String> lastStart) throws Exception {
        lastStart.put(this.name, null);
        int[] indices = this.getIndices(lastStart);

        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            localProbs[i] = probabilities.get((Integer)indices[i]);
        }

        EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(indices, localProbs);
        int idx = localDist.sample();
        return values.get(idx);
    }

    public String[] getParents() {
        return parents;
    }

    public static void main(String[] args) {
//        MersenneTwister mt = new MersenneTwister();
//
//        Variable v = new Variable(mt, new String[]{"a", "b", "c"}, new float[]{(float)0.5, (float)0.4, (float)0.1});
//        String[] samples = v.conditionalSampling(100);
//        for(int i = 0; i < samples.length; i++) {
//            System.out.print(samples[i]);
//        }

    }

    public HashSet<String> getUniqueValues() {
        return this.uniqueValues;
    }

    public HashMap<String, HashMap<String, ArrayList<Integer>>> getTable() {
        return this.table;
    }

    /**
     * Sets all probabilities to zero. Use carefully!
     */
    public void clearProbabilities() {
        for(int i = 0; i < this.probabilities.size(); i++) {
            this.probabilities.set(i, (float)0.0);
        }
    }

    public abstract void updateProbabilities(Individual[] population, Integer[] sortedIndices, int to_select) throws Exception;
}
