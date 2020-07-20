package ednel.utils.comparators;
import java.util.Comparator;

public class DecrescentArrayComparator extends AbstractArrayComparator {

    public DecrescentArrayComparator(Double[] array) {
        super(array);
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return array[index2].compareTo(array[index1]);
    }
}