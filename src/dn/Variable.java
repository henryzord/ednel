package dn;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.Hashtable;

public class Variable {
    private String name;
    private String[] parents;
    private Hashtable<String, Hashtable<String, ArrayList<Integer>>> table;
    private ArrayList<Float> probabilities;
    private MersenneTwister number_generator;

    public Variable(
            String name, String[] parents, Hashtable<String, Hashtable<String, ArrayList<Integer>>> table,
            ArrayList<Float> probabilities, MersenneTwister number_generator) {
        this.name = name;
        this.parents = parents;
        this.table = table;
        this.probabilities = probabilities;
        this.number_generator = number_generator;
    }

    public String[] unconditionalSampling(int sample_size) {
        return null;
    }

    public String[] conditionalSampling(Hashtable<String, String> evidence, int sample_size) {

//        float sum, num, spread = 1000;  // spread is used to guarantee that numbers up to third decimal will be sampled
//
//        String[] sampled = new String [sample_size];
//
//        for(int i = 0; i < sample_size; i++) {
//            num = number_generator.nextInt((int)spread) / spread;
//            sum = 0;
//
//            this.table.get(evidence)
//
//            for(int k = 0; k < values.length; k++) {
//                if((sum < num) && (num <= (sum + probabilities[k]))) {
//                    sampled[i] = values[k];
//                    break;
//                } else {
//                    sum += probabilities[k];
//                }
//            }
//        }
//        return sampled;
        return null;
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
