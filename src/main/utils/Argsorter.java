package utils;

import java.util.Arrays;

public class Argsorter {

    private static Integer[] argsort(Double[] collection, AbstractArrayComparator comparator) {
        Integer[] sortedIndices = comparator.createIndexArray();
        Arrays.sort(sortedIndices, comparator);

        return sortedIndices;
    }

    public static Integer[] decrescent_argsort(Double[] collection) {
        DecrescentArrayComparator comparator = new DecrescentArrayComparator(collection);
        return Argsorter.argsort(collection, comparator);
    }

    public static Integer[] crescent_argsort(Double[] collection) {
        CrescentArrayComparator comparator = new CrescentArrayComparator(collection);
        return Argsorter.argsort(collection, comparator);
    }
}
