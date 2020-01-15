package eda.rules;

import weka.classifiers.rules.Rule;
import weka.core.Instance;
import weka.core.Instances;
import eda.operator.AbstractOperator;

public class RealRule extends Rule {
    private double classIndex;

    private int[] attrIndex;
    private double[] thresholds;
    private AbstractOperator[] operators;

    private int numClasses;
    private String string;

    public RealRule(String line) {
        if(line.contains("(")) {
            line = line.substring(0, line.indexOf("(")).trim();
        }
        this.string = line;
    }

    @Override
    public void grow(Instances data) throws Exception {
        String[] parts = this.string.split(":");
        this.classIndex = data.classAttribute().indexOfValue(parts[1].trim());
        this.numClasses = data.classAttribute().numValues();

        String[] conditions = parts[0].split(" and ");

        this.attrIndex = new int [conditions.length];
        this.thresholds = new double [conditions.length];
        this.operators = new AbstractOperator [conditions.length];

        for(int i = 0; i < conditions.length; i++) {
            String[] parted = conditions[i].split(" ");
            int attr_index = data.attribute(parted[0].trim()).index();
            boolean isNominal = data.attribute(attr_index).isNominal();

            this.attrIndex[i] = attr_index;
            this.operators[i] = AbstractOperator.valueOf(parted[1]);
            this.thresholds[i] = isNominal? data.attribute(attr_index).indexOfValue(parted[2].trim()) : Double.valueOf(parted[2]);
        }
    }

    @Override
    public boolean covers(Instance datum) {
        boolean pass = true;
        int i = 0;
        while(pass && (i < this.attrIndex.length )) {
            AbstractOperator operator = this.operators[i];
            pass = operator.operate(datum.value(this.attrIndex[i]), this.thresholds[i]);
            i += 1;
        }
        // return pass? this.classIndex : -1;
        return pass;
    }

    @Override
    public boolean hasAntds() {
        return (this.attrIndex.length > 0);
    }

    @Override
    public double getConsequent() {
        return this.classIndex;
    }

    @Override
    public double size() {
        return this.attrIndex.length;
    }

    @Override
    public String toString() {
        return this.string;
    }

    @Override
    public String getRevision() {
        return null;
    }
}
