package dn.variables;

import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(
        String name, ArrayList<String> parents_names, HashMap<String, Boolean> isParentContinuous,
        HashMap<String, HashMap<String, ArrayList<Integer>>> table,
        ArrayList<String> values, ArrayList<Double> probabilities,
        MersenneTwister mt, double learningRate, int n_generations) throws Exception {

        super(name, parents_names, isParentContinuous, table,
            null, null, null, probabilities, mt, learningRate, n_generations
        );

        this.values = new ArrayList<>(values.size());
        this.uniqueValues = new HashSet<>();
        this.uniqueShadowvalues = new HashSet<>();

        for(int i = 0; i < values.size(); i++) {
            Shadowvalue sv = new Shadowvalue(
                String.class.getMethod("toString"),
                values.get(i)
            );
            this.values.add(sv);
            this.uniqueValues.add(sv.toString());
            this.uniqueShadowvalues.add(sv);
        }

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
        // removes continuous parents from the set of parents of this variable.
        AbstractVariable[] discreteParents = AbstractVariable.getOnlyDiscrete(parents);
        for(AbstractVariable par : discreteParents) {
            this.parents_names.add(par.getName());
            this.isParentContinuous.put(par.getName(), false);
        }

        HashMap<String, HashSet<String>> eoUniqueValues = Combinator.getUniqueValuesFromVariables(discreteParents);
        // adds unique values of this variable
        eoUniqueValues.put(this.getName(), this.getUniqueValues());

        this.updateTable(eoUniqueValues);
    }

    public void updateUniqueValues(Individual[] fittest) {
        // nothing happens
    }
}
