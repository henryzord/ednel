package dn.variables;

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

    public String conditionalSampling(HashMap<String, String> lastStart) throws Exception {
        Set<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        for(int i = 0; i < this.parents.length; i++) {
            ArrayList<Integer> localIndices = this.table.get(this.parents[i]).get(lastStart.get(this.parents[i]));
            if(localIndices != null) {
                intersection.retainAll(new HashSet<>(localIndices));
            }
        }

        Object[] indices = intersection.toArray();

        int[] intIndices = new int [indices.length];
        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            intIndices[i] = (Integer)indices[i];
            localProbs[i] = probabilities.get((Integer)indices[i]);
        }
        try {
            EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(intIndices, localProbs);
            int idx = localDist.sample();
            return values.get(idx);
        } catch(org.apache.commons.math3.exception.MathArithmeticException e) {
            return null;
        }
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
