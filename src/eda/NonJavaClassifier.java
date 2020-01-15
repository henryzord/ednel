package eda;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

public class NonJavaClassifier extends AbstractClassifier {

    private double[][] train_predictions;
    private double[][] val_predictions;
    private Instances train_data;
    private Instances val_data;

    public NonJavaClassifier(double[][] train_predictions, double [][] val_predictions, Instances train_data, Instances val_data) {
        this.train_predictions = train_predictions;
        this.val_predictions = val_predictions;
        this.train_data = train_data;
        this.val_data = val_data;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        double max_prob = -1, max_prob_index = -1;
        for(int i = 0; i < probs.length; i++) {
            if(probs[i] > max_prob) {
                max_prob_index = i;
                max_prob = probs[i];
            }
        }

        return max_prob_index;
    }

    private static int checkIfContains(Instances set, Instance instance) {
        for(int i = 0; i < set.numInstances(); i++) {
            boolean equal = true;
            for(int j = 0; j < instance.numAttributes() - 1; j++) {
                if(set.instance(i).isMissing(j) && instance.isMissing(j) || set.instance(i).value(j) == instance.value(j)) {

                } else {
                    equal = false;
                    break;
                }
            }
            if(equal) {
                return i;
            }
        }
        return -1;
    }
    
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int index_instance = checkIfContains(train_data, instance);
        if(index_instance == -1) {
            index_instance = checkIfContains(val_data, instance);
            if(index_instance == -1) {
                throw new Exception("Instance is not in training set, nor in validation set.");
            }
            return val_predictions[index_instance];
        }
        return train_predictions[index_instance];
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        double[][] probs = new double [batch.numInstances()][val_predictions[0].length];

        for(int i = 0; i < batch.numInstances(); i++) {
            probs[i] = distributionForInstance(batch.get(i));
        }
        return probs;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    public static void main(String[] args) throws Exception {
        Instances train_data = new Instances(
                new BufferedReader(
                        new FileReader(
                                "/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff"
                        )
                )
        );
        train_data.setClassIndex(train_data.numAttributes() - 1);
        Instances test_data = new Instances(
                new BufferedReader(
                        new FileReader(
                                "/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tst.arff"
                        )
                )
        );
        test_data.setClassIndex(test_data.numAttributes() - 1);

        J48 j48 = new J48();
        j48.buildClassifier(train_data);

        NonJavaClassifier non = new NonJavaClassifier(
                j48.distributionsForInstances(train_data), j48.distributionsForInstances(test_data),
                train_data, test_data
        );

        Evaluation ev = new Evaluation(train_data);
        ev.evaluateModel(non, train_data);

        // System.out.println("Index of instance: " + train_data.indexOf(copy));

    }
}
