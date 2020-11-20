package ednel.utils.comparators;

public class DecrescentArrayComparator extends ArrayContainer {

    public DecrescentArrayComparator(Comparable[] array) {
        super(array);
    }

    @Override
    public int compare(Integer index1, Integer index2) {
        return array[index2].compareTo(array[index1]);
    }
}