package ednel.eda.aggregators.rules;

import ednel.eda.aggregators.RuleExtractorAggregator;
import ednel.eda.rules.ExtractedRule;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;

/**
 * A classifier for checking if extracting rules are performing the same way as the classifier that extracted them.
 */
public class SimpleRuleClassifier extends AbstractClassifier {
    private final boolean isOrdered;
    private final AbstractClassifier clf;
    private ExtractedRule[] rules = null;
    private int n_classes;

    public SimpleRuleClassifier(AbstractClassifier clf) {
        this.isOrdered = clf instanceof JRip || clf instanceof PART;
        this.clf = clf;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        //        this.clf.buildClassifier(data);
        this.rules = RuleExtractor.fromClassifierToRules(this.clf, data);
        this.n_classes = data.numClasses();
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] votes = this.distributionForInstance(instance);

        double index = -1, max = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < this.n_classes; i++) {
            if(votes[i] > max) {
                index = i;
                max = votes[i];
            }
        }
        return index;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] votes = new double [this.n_classes];
        for(int i = 0; i < this.n_classes; i++) {
            votes[i] = 0;
        }

        double sum = 0;
        for(int i = 0; i < this.rules.length; i++) {
            if(this.rules[i].covers(instance)) {
                if((this.clf instanceof DecisionTable) && (i == this.rules.length - 1) && (sum > 0)) {
                    break;
                }

                votes[(int)this.rules[i].getConsequent()] += 1;
                sum += 1;
                if(this.isOrdered) {
                    break;
                }
            }
        }
        if(sum <= 0) {
//            double pred = clf.classifyInstance(instance);
            throw new Exception("no rule covers instance!");
        }

        for(int i = 0; i < this.n_classes; i++) {
            votes[i] /= sum;
        }
        return votes;
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        double[][] dists = new double[batch.numInstances()][];
        for(int i = 0; i < batch.numInstances(); i++) {
            dists[i] = this.distributionForInstance(batch.get(i));
        }
        return dists;
    }
}
