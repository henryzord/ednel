package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public abstract class AbstractVariable extends VariableStructure {
    protected float learningRate;
    protected int n_generations;
    protected MersenneTwister mt;

    public AbstractVariable(
            String name, ArrayList<String> parents, HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Float> probabilities, MersenneTwister mt,
            float learningRate, int n_generations) throws Exception {

        super(name, parents, table, values, probabilities);

        this.mt = mt;
        this.learningRate = learningRate;
        this.n_generations = n_generations;
    }

    public abstract String[] unconditionalSampling(int sample_size) throws Exception;



    protected int conditionalSamplingIndex(HashMap<String, String> lastStart) {
        int[] indices = this.getIndices(lastStart, null, false);

        double[] localProbs = new double [indices.length];
        for(int i = 0; i < localProbs.length; i++) {
            localProbs[i] = probabilities.get(indices[i]);
        }

        EnumeratedIntegerDistribution localDist = new EnumeratedIntegerDistribution(mt, indices, localProbs);
        int idx = localDist.sample();
        return idx;
    }

    public HashMap<String, HashMap<String, ArrayList<Integer>>> getTable() {
        return this.table;
    }

    /**
     * Update probabilities of this Variable based on the fittest population of a generation.
     * @param fittest
     * @throws Exception
     */
    public void updateProbabilities(Individual[] fittest) throws Exception {
        ArrayList<Float> occurs = new ArrayList<>();
        for(int i = 0; i < probabilities.size(); i++) {
            occurs.add((float)0.0);
        }

        if(this.getClass().equals(ContinuousVariable.class)) {
            // gets the count of occurrences
            for(int i = 0; i < fittest.length; i++) {
                HashSet<Integer> parentIndices = this.getSetOfIndices(
                        fittest[i].getCharacteristics(),
                        null,
                        false
                );
                HashSet<Integer> nullIndices = this.getSetOfIndices(
                        fittest[i].getCharacteristics(),
                        null,
                        true
                );

                parentIndices.retainAll(nullIndices);

                for(Object index : parentIndices.toArray()) {
                    occurs.set((int)index, occurs.get((int)index) + 1);
                }
            }
        } else {
            // gets the count of occurrences
            for(int i = 0; i < fittest.length; i++) {
                int[] indices = this.getIndices(
                        fittest[i].getCharacteristics(),
                        fittest[i].getCharacteristics().get(this.name),
                        true
                );
                for(int index : indices) {
                    occurs.set(index, occurs.get(index) + 1);
                }
            }
        }

        // generates combinations of values
        ArrayList<HashMap<String, String>> combinations = new ArrayList<>();
        for(Object value : this.getUniqueValues().toArray()) {
            HashMap<String, String> local = new HashMap<>();
            local.put(this.name, null);
            combinations.add(local);
        }
        for(String parent : this.parents_names) {
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

        // calculates the sum
        for(int i = 0; i < combinations.size(); i++) {
            int[] indices = this.getIndices(combinations.get(i), null, false);
            float sum = 0, newSum = 0, newValue, rest;
            for(int j = 0; j < indices.length; j++) {
                sum += occurs.get(indices[j]);
            }

            if(sum > 0) {
                // updates probabilities using learning rate and relative frequencies
                for(int j = 0; j < indices.length; j++) {
                    newValue = (float) ((1.0 - this.learningRate) * this.probabilities.get(indices[j]) +
                            this.learningRate * occurs.get(indices[j]) / sum);
                    newSum += newValue;
                    this.probabilities.set(
                            indices[j],
                            newValue

                    );
                }
                rest = 1 - newSum;
                newSum = 0;
                for(int j = 0; j < indices.length; j++) {
                    this.probabilities.set(indices[j], this.probabilities.get(indices[j]) + rest / indices.length);
                    newSum += this.probabilities.get(indices[j]);
                }

                if(Math.abs(1 - newSum) > 0.01) {
                    throw new Exception("does not sum up to 1!");
                }
            }
        }

        // clears NaN values
        // TODO throw away this code later
        for(int i = 0; i < probabilities.size(); i++) {
            if(Double.isNaN(probabilities.get(i)) || probabilities.get(i) < 0) {
                probabilities.set(i, (float)0);
            }
        }
    }

    /**
     * Samples a new value for this variable, based on conditions.
     * @param lastStart Last values from the Dependency Network.
     * @return A new value for this variable.
     * @throws Exception
     */
    public String conditionalSampling(HashMap<String, String> lastStart) throws Exception {
        return values.get(this.conditionalSamplingIndex(lastStart));
    }

}
