package ednel.utils.comparators;

public class CrescentArrayComparator extends ArrayContainer {

    public CrescentArrayComparator(Comparable<Object>[] array) {
        super(array);
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return array[index1].compareTo(array[index2]);
    }
}
