package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;

public class ContinuousVariable extends AbstractVariable {

    public ContinuousVariable(String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table, ArrayList<Float> probabilities, MersenneTwister number_generator) throws Exception {
        super(name, parents, table, probabilities, number_generator);
    }

    @Override
    public String[] unconditionalSampling(int sample_size) {
        return new String[0];
    }

    @Override
    public String conditionalSampling(HashMap<String, String> evidence) {
        return null;
    }

    @Override
    public String conditionalSampling(String[] parentNames, String[] parentValues) {
        return null;
    }
}
