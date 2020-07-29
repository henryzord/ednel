package ednel.network.variables;

import netscape.javascript.JSObject;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class DiscreteVariable extends AbstractVariable {

    public DiscreteVariable(
            String name, JSONObject fixedBlocking, HashMap<String, Boolean> isParentContinuous,
            HashMap<String, HashMap<String, ArrayList<Integer>>> table,
            ArrayList<String> values, ArrayList<Double> probabilities,
            MersenneTwister mt, int max_parents) throws Exception {

        super(name, fixedBlocking, isParentContinuous, table,
            null, null, null, probabilities, mt, max_parents
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
    public void updateStructure(AbstractVariable[] mutableParents, HashMap<String, ArrayList<String>> fittest) throws Exception {
        mutableParents = AbstractVariable.getOnlyDiscrete(mutableParents);

        super.updateStructure(mutableParents, fittest);
    }

    public void updateUniqueValues(HashMap<String, ArrayList<String>> fittest) {
        // nothing happens
    }
}
