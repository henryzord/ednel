package ednel.network.variables;

import java.util.HashMap;
import java.util.HashSet;

public class Combinator {

    public static HashMap<String, HashSet<String>> getUniqueValuesFromVariables(AbstractVariable[] parents) {
        HashMap<String, HashSet<String>> uniqueValues = new HashMap<>(parents.length + 1);

        for (AbstractVariable parent : parents) {
            uniqueValues.put(parent.getName(), parent.getUniqueValues());
        }
        return uniqueValues;
    }

}
