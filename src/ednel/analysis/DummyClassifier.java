package ednel.analysis;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;

public class DummyClassifier extends AbstractClassifier {

    HashMap<Integer, double[]> compiledPredictions;

    public DummyClassifier(HashMap<Integer, double[]> predictions) {
        compiledPredictions = new HashMap<>();
        for(Integer key : predictions.keySet()) {
            compiledPredictions.put(key, predictions.get(key));
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = this.compiledPredictions.get((int)instance.value(0));
        double max = Double.NEGATIVE_INFINITY;
        int argmax = -1;
        for(int i = 0; i < probs.length; i++) {
            if(probs[i] > max) {
                max = probs[i];
                argmax = i;
            }
        }
        return argmax;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.compiledPredictions.get((int)instance.value(0)).clone();
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        double[][] predictions = new double[batch.size()][];
        for(int i = 0; i < batch.size(); i++) {
            predictions[i] = this.distributionForInstance(batch.get(i));
        }
        return predictions;
    }
}
