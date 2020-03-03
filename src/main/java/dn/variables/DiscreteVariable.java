package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(
            String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt,
            float learningRate, int n_generations) throws Exception {
        super(name, parents, table, values, probabilities, mt, learningRate, n_generations);
    }

    @Override
    public String[] unconditionalSampling(int sample_size) {
        float sum, num, spread = 1000;  // spread is used to guarantee that numbers up to third decimal will be sampled

        String[] sampled = new String [sample_size];
        float uniformProbability = (float)(1. / this.values.size());
        for(int i = 0; i < sample_size; i++) {
            num = mt.nextInt((int)spread) / spread;
            sum = 0;

            for(int k = 0; k < values.size(); k++) {
                if((sum < num) && (num <= (sum + uniformProbability))) {
                    sampled[i] = values.get(k);
                    break;
                } else {
                    sum += uniformProbability;
                }
            }
        }
        return sampled;
    }
}
