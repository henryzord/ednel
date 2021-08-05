package ednel.eda.individual;

import weka.core.Instances;

public class EvaluateValidationSetThread extends Thread {
    private Instances learn_data;
    private Instances val_data;
    private Individual ind;
    Exception myExcept;
    double valQuality;
    private FitnessCalculator.EvaluationMetric metric;

    public EvaluateValidationSetThread(
            Instances learn_data, Instances val_data,
            Individual ind, Integer timeout_individual,
            FitnessCalculator.EvaluationMetric metric
    ) throws EmptyEnsembleException, NoAggregationPolicyException {
        this.myExcept = null;
        this.valQuality = 0;
        this.learn_data = learn_data;
        this.val_data = val_data;
        this.metric = metric;
        this.ind = new Individual(ind, timeout_individual);
    }

    public void run() {
        try {
            if(this.val_data != null) {
                this.ind.buildClassifier(this.learn_data);
                switch(this.metric) {
                    case UNWEIGHTED_AUC:
                        this.valQuality = FitnessCalculator.getUnweightedAreaUnderROC(this.learn_data, this.val_data, this.ind);
                        break;
                    case BALANCED_ACCURACY:
                        this.valQuality = FitnessCalculator.getBalancedAccuracy(this.learn_data, this.val_data, this.ind);
                        break;
                    default:
                        throw new Exception("Unrecognized metric.");
                }
            } else {
                this.valQuality = 0;
            }
        } catch(Exception e) {
            this.myExcept = e;
        }
    }

    public double getValQuality() {
        return this.valQuality;
    }
}
