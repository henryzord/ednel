package dn.variables;

import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Set;

public class DiscreteVariable extends AbstractVariable {

    protected Object[] values;

    public DiscreteVariable(String name, String[] parents, Hashtable<String, Hashtable<String, ArrayList<Integer>>> table, ArrayList<Float> probabilities, MersenneTwister number_generator) throws Exception {
        super(name, parents, table, probabilities, number_generator);
        this.values = table.get(name).keySet().toArray();
    }

    @Override
    public String[] unconditionalSampling(int sample_size) {
        float sum, num, spread = 1000;  // spread is used to guarantee that numbers up to third decimal will be sampled

        String[] sampled = new String [sample_size];
        float uniformProbability = (float)(1. / this.values.length);
        for(int i = 0; i < sample_size; i++) {
            num = mt.nextInt((int)spread) / spread;
            sum = 0;

            for(int k = 0; k < values.length; k++) {
                if((sum < num) && (num <= (sum + uniformProbability))) {
                    sampled[i] = (String)values[k];
                    break;
                } else {
                    sum += uniformProbability;
                }
            }
        }
        return sampled;
    }

    @Override
    public String conditionalSampling(Hashtable<String, String> evidence) {
        return null;
    }

    public String conditionalSampling(String[] parentNames, String[] parentValues) {

        Set<Integer> intersection = new HashSet<>();
        for(int i = 0; i < this.probabilities.size(); i++) {
            intersection.add(i);
        }

        for(int i = 0; i < parentNames.length; i++) {
            Set<Integer> thisSet = new HashSet<>(this.table.get(parentNames[i]).get(parentValues[i]));
            intersection.retainAll(thisSet);
        }

        Object[] indices = intersection.toArray();
        float probSum = 0;
        for(int i = 0; i < indices.length; i++) {
            probSum += this.probabilities.get((int)indices[i]);
        }

        float sum = 0;
        float spread = 1000;
        float num = mt.nextInt((int)spread) / spread;  // spread is used to guarantee that numbers up to third decimal will be sampled

        String sampled = null;
        for(int i = 0; i < indices.length; i++) {
            if((sum < num) && (num <= (sum + (this.probabilities.get((Integer)indices[i])/probSum)))) {
                sampled = (String)values[i];
                break;
            } else {
                sum += (this.probabilities.get((Integer)indices[i])/probSum);
            }
        }

        return sampled;
    }

}
