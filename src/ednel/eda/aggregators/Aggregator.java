package ednel.eda.aggregators;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Aggregator {
    protected double[] competences;

    public abstract double[][] aggregateProba(AbstractClassifier[] clfs, Instances batch) throws Exception;
    public abstract double[] aggregateProba(AbstractClassifier[] clfs, Instance instance) throws Exception;
    public abstract void setCompetences(AbstractClassifier[] clfs, Instances train_data) throws Exception;

    protected int getActiveClassifiersCount(AbstractClassifier[] clfs) {
        int n_active_classifiers = 0;
        for(AbstractClassifier clf : clfs) {
            if(clf != null) {
                n_active_classifiers += 1;
            }
        }
        return n_active_classifiers;
    }

    public abstract String[] getOptions();
}
