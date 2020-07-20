package eda.rules;

import eda.operator.AbstractOperator;
import weka.classifiers.rules.Rule;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Iterator;

public class RealRule extends Rule {
    private double classIndex;

    private int[] attrIndex;
    private double[] thresholds;
    private AbstractOperator[] operators;
    /**
     * Negative preconditions are rules that should NOT be triggered
     * before triggering this rule.
     */
    private ArrayList<RealRule> negativePreconditions;

    private int numClasses;
    private String string;

    /**
     * How many instances does this rule covers?
     */
    private Double coverage;
    /**
     * From the amount of instances that this rule covers, how many are wrong predictions?
     */
    private Double errors;

    public RealRule(String line, Instances train_data, ArrayList<RealRule> negativePreconditions) throws Exception {
        this.negativePreconditions = negativePreconditions == null? new ArrayList<>() : (ArrayList<RealRule>)negativePreconditions.clone();

        if(line.contains("(")) {
            String[] metadata_splited = line.substring(line.indexOf("(") + 1, line.indexOf(")")).split("/");
            this.coverage = Double.parseDouble(metadata_splited[0]);
            try {
                this.errors = Double.parseDouble(metadata_splited[1]);
            } catch(IndexOutOfBoundsException e) {  // rule has no errors
                this.errors = 0.0;
            }
            line = line.substring(0, line.indexOf("(")).trim();
        } else {
            this.coverage = Double.NaN;
            this.errors = Double.NaN;
        }
        this.string = line;
        this.grow(train_data);
    }

    @Override
    public void grow(Instances data) throws Exception {
        String[] parts = this.string.split(":");
        this.classIndex = data.classAttribute().indexOfValue(parts[1].trim());
        this.numClasses = data.classAttribute().numValues();

        // if there are pre-conditions to this specific rule; otherwise, it is the default rule
        if(parts[0].length() > 0) {
            String[] conditions = parts[0].split(" and ");

            this.attrIndex = new int [conditions.length];
            this.thresholds = new double [conditions.length];
            this.operators = new AbstractOperator [conditions.length];

            for(int i = 0; i < conditions.length; i++) {
                String[] parted = conditions[i].trim().split(" ");
                int attr_index = data.attribute(parted[0].trim()).index();
                boolean isNominal = data.attribute(attr_index).isNominal();

                this.attrIndex[i] = attr_index;
                this.operators[i] = AbstractOperator.valueOf(parted[1].trim());
                this.thresholds[i] = isNominal? data.attribute(attr_index).indexOfValue(parted[2].trim()) : Double.valueOf(parted[2].trim());
            }
        } else {
            this.attrIndex = new int[0];
            this.thresholds = new double[0];
            this.operators = new AbstractOperator[0];
        }
    }

    @Override
    public boolean covers(Instance datum) {
        // checks if negative pre-conditions cover data. if so, returns false
        for(int i = 0; i < this.negativePreconditions.size(); i++) {
            if(this.negativePreconditions.get(i).covers(datum)) {
                return false;
            }
        }
        boolean pass = true;
        int i = 0;
        while(pass && (i < this.attrIndex.length)) {
            AbstractOperator operator = this.operators[i];
            pass = operator.operate(datum.value(this.attrIndex[i]), this.thresholds[i]);
            i += 1;
        }
        return pass;
    }

    @Override
    public boolean hasAntds() {
        return (this.attrIndex.length > 0) || this.negativePreconditions.size() > 0;
    }

    @Override
    public double getConsequent() {
        return this.classIndex;
    }

    @Override
    public double size() {
        int negative_count = 0;
        for (Iterator<RealRule> iterator = this.negativePreconditions.iterator(); iterator.hasNext(); ) {
            RealRule antcd = iterator.next();
            negative_count += antcd.size();
        }

        return negative_count + this.attrIndex.length;
    }

    @Override
    public String toString() {
        String answer = this.string;

        // TODO to completely fix this, it will be needed an algorithm to remove redundancies

//        if(this.negativePreconditions.size() > 0) {
//            String negatives_joined = this.negativePreconditions.get(0).toString();
//            for(int i = 1; i < this.negativePreconditions.size(); i++) {
//                negatives_joined += " and " + this.negativePreconditions.get(i).toString();
//            }
//            answer = String.format("not (%s) and %s", negatives_joined, answer);
//        }
        return answer;
    }

    @Override
    public String getRevision() {
        return null;
    }
}
