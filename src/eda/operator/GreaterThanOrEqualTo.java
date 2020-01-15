package eda.operator;

public class GreaterThanOrEqualTo extends AbstractOperator {
    public boolean operate(double a, double b) {
        return a >= b;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new GreaterThanOrEqualTo().operate(3, 3));
        System.out.println(new GreaterThanOrEqualTo().operate(1, 2));
        System.out.println(new GreaterThanOrEqualTo().operate(2, 1));
    }
}
