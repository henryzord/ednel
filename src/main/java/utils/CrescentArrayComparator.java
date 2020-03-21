package utils;

public class CrescentArrayComparator extends AbstractArrayComparator {

    public CrescentArrayComparator(Double[] array) {
        super(array);
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return array[index1].compareTo(array[index2]);
    }
}
