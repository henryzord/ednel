package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

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

    public AbstractVariable(
            String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt) throws Exception {
        this.name = name;
        this.parents = parents;
        this.table = table;
        this.values = values;
        this.probabilities = probabilities;
        this.mt = mt;
    }

    public abstract String[] unconditionalSampling(int sample_size) throws Exception;

    public String conditionalSampling(String[] parentNames, String[] parentValues) throws Exception {
        Set<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        for(int i = 0; i < parentNames.length; i++) {
            ArrayList<Integer> localIndices = this.table.get(parentNames[i]).getOrDefault(parentValues[i], null);
            if(localIndices != null) {
                intersection.retainAll(new HashSet<>(localIndices));
            }
        }

        Object[] indices = intersection.toArray();

        float probSum = 0;
        for(int i = 0; i < indices.length; i++) {
            probSum = probSum + this.probabilities.get((Integer) indices[i]);
        }

        float sum = 0;
        float spread = 1000;
        float num = mt.nextInt((int)spread) / spread;  // spread is used to guarantee that numbers up to third decimal will be sampled

        String sampled = null;
        for(int i = 0; i < indices.length; i++) {
            if((sum < num) && (num <= (sum + (this.probabilities.get((Integer)indices[i])/probSum)))) {
                sampled = values.get(i);
                break;
            } else {
                sum += (this.probabilities.get((Integer)indices[i])/probSum);
            }
        }

        return sampled;
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

}
