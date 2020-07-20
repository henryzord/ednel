package eda.operator;

public class GreaterThan extends AbstractOperator  {
    public boolean operate(double a, double b) {
        return a > b;
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new GreaterThan().operate(3, 3));
        System.out.println(new GreaterThan().operate(1.0, 2.0));
    }
}
