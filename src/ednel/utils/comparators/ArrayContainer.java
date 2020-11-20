package ednel.utils.comparators;

import java.util.Comparator;

/**
 * A class that holds an array of objects, for later comparison.
 * Generates an index for each item in this array. When sorting,
 * the indices of the objects in the array will also be sorted.
 */
public abstract class ArrayContainer implements Comparator<Integer> {
    protected Comparable[] array;

    protected Integer[] indices;

    public ArrayContainer(Comparable[] array) {
        this.array = array;

        this.indices = new Integer[this.array.length];
        for (int i = 0; i < this.array.length; i++) {
            this.indices[i] = i;
        }
    }

    public Integer[] getIndices() {
        return indices;
    }
}
