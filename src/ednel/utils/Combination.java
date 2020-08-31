package ednel.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class Combination {
    HashMap<String, String> pairs;

    ArrayList<String> sortedKeys;

    public Combination(HashMap<String, String> pairs) {
        this.pairs = pairs;

        sortedKeys = new ArrayList<>();
        sortedKeys.addAll(this.pairs.keySet());
        Collections.sort(sortedKeys);
    }

    public HashMap<String, String> getPairs() {
        return pairs;
    }

    @Override
    public int hashCode() {
        StringBuilder hash = new StringBuilder(String.format("%s=%s", sortedKeys.get(0), pairs.get(sortedKeys.get(0))));
        for(int i = 1; i < sortedKeys.size(); i++) {
            hash.append(String.format(",%s=%s", sortedKeys.get(i), pairs.get(sortedKeys.get(i))));
        }
        return hash.toString().hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        return (obj instanceof Combination) && ((Combination)obj).hashCode() == this.hashCode();
    }
}
