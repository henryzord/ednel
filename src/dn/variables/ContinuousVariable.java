package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.Hashtable;

public class ContinuousVariable extends AbstractVariable {


    public ContinuousVariable(String name, String[] parents, Hashtable<String, Hashtable<String, ArrayList<Integer>>> table, ArrayList<Float> probabilities, MersenneTwister number_generator) throws Exception {
        super(name, parents, table, probabilities, number_generator);
    }

    @Override
    public String[] unconditionalSampling(int sample_size) {
        return new String[0];
    }

    @Override
    public String[] conditionalSampling(Hashtable<String, String> evidence, int sample_size) {
        return new String[0];
    }
}
