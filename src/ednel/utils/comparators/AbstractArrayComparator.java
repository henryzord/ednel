package ednel.utils.comparators;

import java.util.Comparator;

public abstract class AbstractArrayComparator implements Comparator<Integer> {
    protected Double[] array;

    public AbstractArrayComparator(Double[] array) {
        this.array = array;
    }

    public Integer[] createIndexArray() {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indexes[i] = i;
        }
        return indexes;
    }
}
