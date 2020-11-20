package ednel.utils.sorters;

import java.util.Arrays;
import ednel.utils.comparators.ArrayContainer;
import ednel.utils.comparators.CrescentArrayComparator;
import ednel.utils.comparators.DecrescentArrayComparator;

public class Argsorter {

    private static Integer[] argsort(ArrayContainer container) {
        Integer[] indices = container.getIndices();
        Arrays.sort(indices, container);
        return indices;
    }

    public static Integer[] decrescent_argsort(Comparable[] collection) {
        DecrescentArrayComparator comparator = new DecrescentArrayComparator(collection);
        return Argsorter.argsort(comparator);
    }

    public static Integer[] crescent_argsort(Comparable[] collection) {
        CrescentArrayComparator comparator = new CrescentArrayComparator(collection);
        return Argsorter.argsort(comparator);
    }
}
