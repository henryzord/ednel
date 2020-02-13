package dn.variables;

import eda.Individual;
import org.apache.commons.math3.random.MersenneTwister;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Set;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(String name, String[] parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table, ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt) throws Exception {
        super(name, parents, table, values, probabilities, mt);
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

    public void updateProbabilities(Individual[] population, Integer[] sortedIndices, int to_select) throws Exception {
        this.clearProbabilities();  // sets all to zero
        for(int i = 0; i < population.length; i++) {
            int[] indices = this.getIndices(population[i].getCharacteristics(), population[i].getCharacteristics().get(this.name));
            for(int index : indices) {
                this.probabilities.set(index, this.probabilities.get(index) + 1);
            }
        }

        ArrayList<HashMap<String, String>> combinations = new ArrayList<>();
        for(Object value : this.getUniqueValues().toArray()) {
            HashMap<String, String> local = new HashMap<>();
            local.put(this.name, null);
            combinations.add(local);
        }
        for(String parent : this.parents) {
            ArrayList<HashMap<String, String>> new_combinations = new ArrayList<>();
            for(int i = 0; i < combinations.size(); i++) {
                Object[] parentUniqueVals = this.table.get(parent).keySet().toArray();
                for(int j = 0; j < parentUniqueVals.length; j++) {
                    HashMap<String, String> local = (HashMap<String, String>)combinations.get(i).clone();
                    local.put(parent, (String)parentUniqueVals[j]);
                    new_combinations.add(local);
                }
            }
            combinations = new_combinations;
        }
        for(int i = 0; i < combinations.size(); i++) {
            int[] indices = this.getIndices(combinations.get(i), null);
            float sum = 0;
            for(int j = 0; j < indices.length; j++) {
                sum += this.probabilities.get(indices[j]);
            }
            for(int j = 0; j < indices.length; j++) {
                this.probabilities.set(indices[j], this.probabilities.get(indices[j]) / sum);
            }
        }
        for(int i = 0; i < probabilities.size(); i++) {
            if(Double.isNaN(probabilities.get(i))) {
                probabilities.set(i, (float)0);
            }
        }
    }
}
