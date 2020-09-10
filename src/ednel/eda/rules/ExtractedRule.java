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

    /** Number of classes. */
    private int numClasses;

    /** Description of this rule. */
    private String string;

    /** How many instances were originally covered by this rule, when it was extracted from its context. */
    private Double originalCoverage;
    /** From the (original) number of instances that this rule covered, the amount that were incorrectly classified */
    private Double originalErrors;

    private static final String[] str_operators = new String[]{">=", "<=", ">", "<", "!=", "="};

    /** Most common values found in whole training dataset, for inputing in
     * instances in case that instance does not have that attribute value */
    private double[] mostCommonValueIndices;

    public ExtractedRule(String line, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        this.mostCommonValueIndices = mostCommonValueIndices;

        int last_parenthesis_index = line.indexOf("(", line.lastIndexOf(":"));

        if(last_parenthesis_index != -1) {
            String[] metadata_splited = line.substring(last_parenthesis_index + 1, line.indexOf(")", last_parenthesis_index)).split("/");
            this.originalCoverage = Double.parseDouble(metadata_splited[0]);
            try {
                this.originalErrors = Double.parseDouble(metadata_splited[1]);
            } catch(IndexOutOfBoundsException e) {  // rule has no errors
                this.originalErrors = 0.0;
            }
            line = line.substring(0, last_parenthesis_index).trim();
        } else {
            this.originalCoverage = Double.NaN;
            this.originalErrors = Double.NaN;
        }
        this.string = line;
        this.grow(train_data);
    }

    @Override
    public void grow(Instances data) throws Exception {
//        String[] parts = this.string.split(":");
        String antecedents = this.string.substring(0, this.string.lastIndexOf(":")).trim();
        String consequent = this.string.substring(this.string.lastIndexOf(":") + 1, this.string.length()).trim();

        this.classIndex = data.classAttribute().indexOfValue(consequent);
        this.numClasses = data.classAttribute().numValues();

        // if there are pre-conditions to this specific rule; otherwise, it is the default rule
        if(antecedents.length() > 0) {
            String[] conditions = antecedents.split(" and ");

            this.attrIndex = new int [conditions.length];
            this.thresholds = new double [conditions.length];
            this.operators = new AbstractOperator [conditions.length];

            for(int i = 0; i < conditions.length; i++) {
                String[] parted = new String[]{conditions[i].trim()};
                String op = null;
                for(int j = 0; (j < ExtractedRule.str_operators.length) && (parted.length < 2); j++) {
                    op = ExtractedRule.str_operators[j];
                    parted = conditions[i].split(op);
                }
                if(parted.length < 2) {
                    throw new Exception("could not split rule fragment!");
                }

                int attr_index = data.attribute(parted[0].trim()).index();
                boolean isNominal = data.attribute(attr_index).isNominal();

                this.attrIndex[i] = attr_index;
                this.operators[i] = AbstractOperator.valueOf(op);
                try {
                    this.thresholds[i] = isNominal? data.attribute(attr_index).indexOfValue(parted[1].trim()) : Double.valueOf(parted[1].trim());
                } catch(NumberFormatException nfe) {
                    // sometimes decision tables will place a '?' instead of a number for numeric attributes. this exception treats it
                    this.thresholds[i] = Double.NaN;
                }
            }
        } else {
            this.attrIndex = new int[0];
            this.thresholds = new double[0];
            this.operators = new AbstractOperator[0];
        }
    }

    @Override
    public boolean covers(Instance datum) {
        boolean pass = true;
        int i = 0;
        while(pass && (i < this.attrIndex.length)) {
            double inst_val;
            if(datum.isMissing(this.attrIndex[i])) {
                if(Double.isNaN(this.thresholds[i])) {
                    pass = true;
                    i += 1;  // passes
                    continue;
                } else {
                    inst_val = this.mostCommonValueIndices[this.attrIndex[i]];
                }
            } else {
                inst_val = datum.value(this.attrIndex[i]);
            }

            AbstractOperator operator = this.operators[i];
            pass = operator.operate(inst_val, this.thresholds[i]);
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
        return this.attrIndex.length > 0;
    }

    @Override
    public double getConsequent() {
        return this.classIndex;
    }

    @Override
    public double size() {
        return this.attrIndex.length;
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
        return this.string;
    }

    @Override
    public String getRevision() {
        return null;
    }

}
