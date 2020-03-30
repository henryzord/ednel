package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;

import java.lang.reflect.Array;
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

    @Override
    public void updateStructure(VariableStructure[] parents, Individual[] fittest) throws Exception {
        super.updateStructure(parents, fittest);

        int countDiscreteParents = this.countDiscreteParents(parents);
        VariableStructure[] discreteParents = null;
        if(countDiscreteParents != parents.length) {
            discreteParents = new AbstractVariable [countDiscreteParents];
            int counter = 0;
            for(int i = 0; i < parents.length; i++) {
                if(parents[i] instanceof DiscreteVariable) {
                    discreteParents[counter] = parents[i];
                    counter += 1;
                }
            }
        } else {
            discreteParents = parents;
        }
        this.setParents(discreteParents);

        ArrayList<ArrayList<String>> combinations = this.generateCombinations(discreteParents, true);
        int n_combinations = combinations.size();

        this.values = new ArrayList<>(n_combinations);
        this.probabilities = new ArrayList<>(n_combinations);

        ArrayList<String> indices = new ArrayList<>(discreteParents.length + 1);

        this.table = new HashMap<>(n_combinations);
        for(int i = 0; i < discreteParents.length; i++) {
            String name = discreteParents[i].getName();
            Object[] uniqueValues = discreteParents[i].getUniqueValues().toArray();
            indices.add(name);

            this.table.put(name, new HashMap<>(uniqueValues.length));
            for(Object value : uniqueValues) {
                this.table.get(name).put((String)value, new ArrayList<>());
            }
        }
        this.table.put(this.name, new HashMap<>(this.uniqueValues.size()));
        for(Object value : this.uniqueValues.toArray()) {
            this.table.get(this.name).put((String)value, new ArrayList<>());
        }
        indices.add(this.name);

        for(int i = 0; i < combinations.size(); i++) {
            ArrayList<String> values = combinations.get(i);

            for(int j = 0; j < values.size(); j++) {
                table.get(indices.get(j)).get(values.get(j)).add(i);
            }
        }
    }
}
