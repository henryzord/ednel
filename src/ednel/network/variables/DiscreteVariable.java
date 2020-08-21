package ednel.network.variables;

import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(
            String name, JSONObject fixedBlocking, HashMap<String, Boolean> isParentContinuous,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> initial_values, ArrayList<Double> probabilities,
            MersenneTwister mt, int max_parents) throws Exception {

        super(name, fixedBlocking, isParentContinuous, table, initial_values, probabilities, mt, max_parents);
    }

    /**
     * Updates the structure of the table of indices.
     * @param prob_parents Parents to be added to mutable parents of this variable.
     * @param fittest Set of fittest individuals of current generation.
     * @throws Exception
     */
    @Override
    public void updateStructure(AbstractVariable[] prob_parents, HashMap<String, ArrayList<String>> fittest) throws Exception {
        prob_parents = AbstractVariable.getOnlyDiscrete(prob_parents);

        super.updateStructure(prob_parents, fittest);
    }

    public void setValues(HashMap<String, ArrayList<String>> fittest) throws Exception {
        super.setValues(fittest);
    }
}
