package ednel.network.variables;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

public class Combinator {

    public static HashMap<String, ArrayList<String>> getUniqueValuesFromVariables(AbstractVariable[] all_parents, AbstractVariable child) {
        HashMap<String, ArrayList<String>> uniqueValues = new HashMap<>(all_parents.length + 1);

        for (AbstractVariable parent : all_parents) {
            uniqueValues.put(parent.getName(), parent.getUniqueValues());
        }
        if(child != null) {
            uniqueValues.put(child.getName(), child.getUniqueValues());
        }

        return uniqueValues;
    }

}
