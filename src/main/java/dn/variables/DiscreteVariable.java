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

        for (String value : values) {
            Shadowvalue sv = new Shadowvalue(
                String.class.getMethod("toString"),
                value
            );
            this.values.add(sv);
            this.uniqueValues.add(sv.toString());
            this.uniqueShadowvalues.add(sv);
        }
    }

    /**
     * Updates the structure of the table of indices.
     * @param mutableParents Parents to be added to mutable parents of this variable.
     * @param fittest Set of fittest individuals of current generation.
     * @throws Exception
     */
    @Override
    public void updateStructure(AbstractVariable[] mutableParents, AbstractVariable[] fixedParents,
                                HashMap<String, ArrayList<String>> fittest) throws Exception {
        super.updateStructure(mutableParents, fixedParents, fittest);
        // removes continuous parents from the set of parents of this variable.
        AbstractVariable[] discreteParents = AbstractVariable.getOnlyDiscrete(mutableParents);
        for(AbstractVariable par : discreteParents) {
            if(this.fixed_parents.indexOf(par.getName()) == -1) {
                this.mutable_parents.add(par.getName());
                this.isParentContinuous.put(par.getName(), false);
            }
        }

        HashMap<String, HashSet<String>> mutableUniqueValues = Combinator.getUniqueValuesFromVariables(discreteParents);

        // adds unique values of fixed variables
        mutableUniqueValues.putAll(Combinator.getUniqueValuesFromVariables(fixedParents));
        // adds unique values of this variable
        mutableUniqueValues.put(this.getName(), this.getUniqueValues());

        this.updateTable(mutableUniqueValues);
    }

    public void updateUniqueValues(HashMap<String, ArrayList<String>> fittest) {
        // nothing happens
    }
}
