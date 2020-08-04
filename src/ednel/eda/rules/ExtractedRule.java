package ednel.eda.rules;

import com.sun.org.apache.xpath.internal.operations.Bool;
import ednel.utils.operators.AbstractOperator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.Rule;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Iterator;

public class ExtractedRule extends Rule {
    private double classIndex;

    private int[] attrIndex;
    private double[] thresholds;
    private AbstractOperator[] operators;
    /**
     * Negative preconditions are rules that should NOT be triggered
     * before triggering this rule.
     */
    private ArrayList<ExtractedRule> negativePreconditions;

    /**
     * Number of classes.
     */
    private int numClasses;

    /**
     * Description of this rule.
     */
    private String string;

    /**
     * How many instances were originally covered by this rule, when it was extracted from its context.
     */
    private Double originalCoverage;
    /**
     * From the (original) number of instances that this rule covered, the amount that were incorrectly classified
     */
    private Double originalErrors;

    public ExtractedRule(String line, Instances train_data, ArrayList<ExtractedRule> negativePreconditions) throws Exception {
        this.negativePreconditions = negativePreconditions == null? new ArrayList<>() : (ArrayList<ExtractedRule>)negativePreconditions.clone();

        if(line.contains("(")) {
            String[] metadata_splited = line.substring(line.indexOf("(") + 1, line.indexOf(")")).split("/");
            this.originalCoverage = Double.parseDouble(metadata_splited[0]);
            try {
                this.originalErrors = Double.parseDouble(metadata_splited[1]);
            } catch(IndexOutOfBoundsException e) {  // rule has no errors
                this.originalErrors = 0.0;
            }
            line = line.substring(0, line.indexOf("(")).trim();
        } else {
            this.originalCoverage = Double.NaN;
            this.originalErrors = Double.NaN;
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

    public boolean[] covers(Instances datum) {
        boolean[] covered = new boolean[datum.size()];
        for(int i = 0; i < datum.size(); i++) {
            covered[i] = this.covers(datum.get(i));
        }
        return covered;
    }

    public boolean[] covers(Instances datum, boolean[] activated) {
        boolean covered[] = new boolean[datum.size()];
        for(int i = 0; i < datum.size(); i++) {
            covered[i] = activated[i] && this.covers(datum.get(i));
        }
        return covered;
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
        for (Iterator<ExtractedRule> iterator = this.negativePreconditions.iterator(); iterator.hasNext(); ) {
            ExtractedRule antcd = iterator.next();
            negative_count += antcd.size();
        }

        return negative_count + this.attrIndex.length;
    }


    /**
     * The quality of a rule is given by its precision * recall over a set of instances for a given class.
     *
     * The rule covers only one class; so it is either positive or negative on that set of instances.
     *
     * @param data training data
     * @param activated instances that must be analyzed by this rule
     * @return precison * recall of this rule (on the class that it predicts)
     */
    public double quality(Instances data, boolean[] activated) {
        int[][] confusionMatrix = new int[][]{{0, 0}, {0, 0}};  // only has positive and negative classes

        // confusion matrix:
        //                 predicted true predicted false
        // actual true     [true positive, false negative]
        // actual false    [false positive, true negative]
        for(int i = 0; i < activated.length; i++) {
            if(activated[i]) {
                boolean covers = this.covers(data.get(i));
                boolean getsRight = this.getConsequent() == data.get(i).classValue();

                int line = getsRight? (covers? 0 : 1) : (covers? 1 : 0);
                int column = covers? 0 : 1;
                confusionMatrix[line][column] += 1;
            }
        }
        double precision = (double)confusionMatrix[0][0] / (double)(confusionMatrix[0][0] + confusionMatrix[1][0]);
        double recall = (double)confusionMatrix[0][0] / (double)(confusionMatrix[0][0] + confusionMatrix[0][1]);
        return precision * recall;
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
