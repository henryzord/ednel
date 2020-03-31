package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(
        String name, ArrayList<String> parents_names, ArrayList<Boolean> isParentDiscrete,
        HashMap<String, HashMap<String, ArrayList<Integer>>> table,
        ArrayList<String> values, ArrayList<Double> probabilities,
        MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        super(name, parents_names, isParentDiscrete, table,
            null, null, probabilities, mt, learningRate, n_generations
        );

        this.values = new ArrayList<>(values.size());
        this.uniqueValues = new HashSet<>();

        for(int i = 0; i < values.size(); i++) {
            ShadowValue sv = new ShadowValue(
                String.class.getMethod("toString"),
                values.get(i)
            );
            this.values.add(sv);
            this.uniqueValues.add(sv);
        }

    }

    @Override
    public String[] unconditionalSampling(int sample_size) {
        float sum, num, spread = 1000;  // spread is used to guarantee that numbers up to third decimal will be sampled

        String[] sampled = new String [sample_size];
        double uniformProbability = (1. / this.values.size());
        for(int i = 0; i < sample_size; i++) {
            num = mt.nextInt((int)spread) / spread;
            sum = 0;

            for(int k = 0; k < values.size(); k++) {
                if((sum < num) && (num <= (sum + uniformProbability))) {
                    sampled[i] = values.get(k).getValue();
                    break;
                } else {
                    sum += uniformProbability;
                }
            }
        }
        return sampled;
    }

    /**
     * Updates the structure of the table of indices.
     * @param parents Set of indices of this variable.
     * @param fittest Set of fittest individuals of current generation.
     * @throws Exception
     */
    @Override
    public void updateStructure(AbstractVariable[] parents, Individual[] fittest) throws Exception {
        super.updateStructure(parents, fittest);
        AbstractVariable[] discreteParents = this.removeContinuousParents(parents);

        HashMap<String, HashSet<String>> eoUniqueValues = this.getUniqueValuesFromVariables(discreteParents, true);
        this.updateTableEntries(eoUniqueValues);
    }
}
