package eda.operator;

public class EqualTo extends AbstractOperator {
    @Override
    public boolean operate(double a, double b) {
        return a == b;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new EqualTo().operate(3, 3));
        System.out.println(new EqualTo().operate(1, 2));
    }
}
