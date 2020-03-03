package eda.operator;

import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;

public abstract class AbstractOperator {
    public boolean operate(double a, double b) {
        return false;
    }
    static public AbstractOperator valueOf(String operation) {
        operation = operation.trim();
        if(operation.equals(">=")) {
            return new GreaterThanOrEqualTo();
        } else if(operation.equals("<=")) {
            return new LessThanOrEqualTo();
        } else  if(operation.equals("<")){
            return new LessThan();
        } else if(operation.equals(">")) {
            return new GreaterThan();
        } else if(operation.equals("=") || operation.equals("==")) {
            return new EqualTo();
        } else if(operation.equals("!=")) {
            return new NotEqualTo();
        } else {
            throw new ValueException("Unrecognized operator: " + operation);
        }
    }
}
