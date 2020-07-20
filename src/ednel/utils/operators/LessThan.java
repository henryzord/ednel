package ednel.utils.operators;

public class LessThan extends AbstractOperator {
    public boolean operate(double a, double b) {
        return a < b;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new LessThan().operate(3, 3));
        System.out.println(new LessThan().operate(1, 2));
        System.out.println(new LessThan().operate(2, 1));
    }
}
