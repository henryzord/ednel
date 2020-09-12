package ednel.utils.operators;

public class EqualsAny extends AbstractOperator {
    @Override
    public boolean operate(double a, double b) {
        return true;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new EqualTo().operate(3, 3));
        System.out.println(new EqualTo().operate(1, 2));
    }
}
