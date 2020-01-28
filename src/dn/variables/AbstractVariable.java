package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;

public abstract  class AbstractVariable {
    protected String name;
    protected String[] parents;
    protected Hashtable<String, Hashtable<String, ArrayList<Integer>>> table;
    protected ArrayList<Float> probabilities;
    protected MersenneTwister number_generator;

    public AbstractVariable(
            String name, String[] parents, Hashtable<String, Hashtable<String, ArrayList<Integer>>> table,
            ArrayList<Float> probabilities, MersenneTwister number_generator) throws Exception {
        this.name = name;
        this.parents = parents;
        this.table = table;
        this.probabilities = probabilities;
        this.number_generator = number_generator;
    }

    public abstract String[] unconditionalSampling(int sample_size);
    public abstract String[] conditionalSampling(Hashtable<String, String> evidence, int sample_size);

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
