package dn.variables;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;

public class Combinator {

    /**
     * Generates a list of combinations of values between variables.
     * Values of this variable (if selfInclude = true) are placed in the end of the list.
     *
     * @param parents Parents of this variable.
     * @param selfInclude Whether to include values of this variable. The values will be placed
     *                    at the end of the list.
     * @return Combinations of values between the (discrete) variables.
     */
//    public static ArrayList<ArrayList<String>> generateCombinations(AbstractVariable[] parents, boolean selfInclude) throws Exception {
//        ArrayList<ArrayList<String>> allValues = new ArrayList<>(parents.length);
//
//        Method getUniqueValues = AbstractVariable.class.getMethod("getUniqueValues");
//        ArrayList<HashSet<String>> localValues = new ArrayList<>(parents.length + 1);
//        for(int i = 0; i < parents.length; i++) {
//            localValues.add((HashSet<String>)getUniqueValues.invoke(parents[i], null));
//        }
//        if(selfInclude) {
//            localValues.add((HashSet<String>) getUniqueValues.invoke(this, null));
//        }
//
//        for(int i = 0; i < localValues.size(); i++) {
//            ArrayList<String> local = new ArrayList<>();
//            String[] dummy = new String[localValues.get(i).size()];
//            throw new Exception("fix this");
//            dummy = localValues.get(i).toArray(dummy);
//            local.addAll(localValues.get(i));
//            allValues.add(local);
//        }
//        return generateCombinations(allValues);
//    }

    public static HashMap<String, HashSet<String>> getUniqueValuesFromVariables(AbstractVariable[] parents) {
        HashMap<String, HashSet<String>> uniqueValues = new HashMap<>(parents.length + 1);

        for (AbstractVariable parent : parents) {
            uniqueValues.put(parent.getName(), parent.getUniqueValues());
        }
        return uniqueValues;
    }

}
