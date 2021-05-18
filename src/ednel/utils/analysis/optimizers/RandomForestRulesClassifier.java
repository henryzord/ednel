package ednel.utils.analysis.optimizers;

import ednel.eda.aggregators.RuleExtractorAggregator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class RandomForestRulesClassifier extends RandomForest {

    RuleExtractorAggregator rea;

    public RandomForestRulesClassifier() {
        super();
        this.rea = new RuleExtractorAggregator();
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);


        this.rea.setCompetences(new AbstractClassifier[]{this}, data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = this.distributionForInstance(instance);

        double max_value = -1;
        int max_index = -1;
        for(int i = 0; i < dist.length; i++) {
            if(dist[i] > max_value) {
                max_value = dist[i];
                max_index = i;
            }
        }
        return max_index;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.rea.aggregateProba(null, instance);
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return this.rea.aggregateProba(null, batch);
    }
}
