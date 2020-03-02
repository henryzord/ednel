package eda.individual;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

/**
 * A little demo java program for using WEKA.<br/>
 * Check out the Evaluation class for more details.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 * @see Evaluation
 */
public class FitnessCalculator {

    private Instances train_data, test_data;
    private int n_folds;

    // initializes the demo
    public FitnessCalculator(int n_folds, Instances train_data, Instances test_data) throws Exception {
        this.train_data = train_data;
        this.test_data = test_data;
        this.n_folds = n_folds;
    }

    public static double getUnweightedAreaUnderROC(Instances train_data, Instances test_data, AbstractClassifier clf) throws Exception {
        Evaluation evaluation = new Evaluation(train_data);
        evaluation.evaluateModel(clf, test_data);
        return getUnweightedAreaUnderROC(evaluation);
    }

    public static double getUnweightedAreaUnderROC(Evaluation evaluation) {
        int n_classes = evaluation.confusionMatrix().length;
        double unweighted = 0;
        for(int i = 0; i < n_classes; i++) {
            if(Utils.isMissingValue(evaluation.areaUnderROC(i))) {
                unweighted += 0;
            } else {
              unweighted += evaluation.areaUnderROC(i);
            }
        }

        return unweighted / n_classes;
    }

    public Double[][] evaluateEnsembles(int seed, Individual[] population) throws Exception {
        int n_individuals = population.length;

        Double[] trainEvaluations = new Double [n_individuals];
        Double[] testEvaluations = new Double [n_individuals];

        for(int k = 0; k < n_individuals; k++) {
            trainEvaluations[k] = 0.0;
        }

        Random random = new Random(seed);

        Evaluation trainEval = new Evaluation(train_data);

        // do the folds
        for (int i = 0; i < n_folds; i++) {
            Instances local_train = train_data.trainCV(n_folds, i, random), local_val = train_data.testCV(n_folds, i);

            for(int j = 0; j < n_individuals; j++) {
                population[j].buildClassifier(local_train);
                trainEval.evaluateModel(population[j], local_val);
                trainEvaluations[j] += getUnweightedAreaUnderROC(trainEval);

                if((i == n_folds - 1) && (test_data != null)) {
                    population[j].buildClassifier(train_data);

                    Evaluation testEval = new Evaluation(test_data);
                    testEval.evaluateModel(population[j], test_data);
                    testEvaluations[j] = getUnweightedAreaUnderROC(testEval);
                } else {
                    testEvaluations[j] = null;
                }
            }
        }
        for(int k = 0; k < n_individuals; k++) {
            trainEvaluations[k] /= n_folds;
        }

        return new Double[][]{trainEvaluations, testEvaluations};
    }

//    public Double[][] evaluateEnsembles(int seed,
//                                        String[][] j48Parameters, String[][] simpleCartParameters,
//                                        String[][] partParameters, String[][] jripParameters, String[][] decisionTableParameters,
//                                        String[][] aggregatorParameters) throws Exception {
//
//        int n_individuals = j48Parameters.length;
//
//        Individual[] population = new Individual [n_individuals];
//
//        for(int j = 0; j < n_individuals; j++) {
//            Individual individual = new Individual();
//            individual.setOptions(new String[]{
//                    "-J48", String.join(" ", j48Parameters[j]),
//                    "-SimpleCart", String.join(" ", simpleCartParameters[j]),
//                    "-PART", String.join(" ", partParameters[j]),
//                    "-JRip", String.join(" ", jripParameters[j]),
//                    "-DecisionTable", String.join(" ", decisionTableParameters[j]),
//                    "-Aggregator", String.join(" ", aggregatorParameters[j])
//            });
//            population[j] = individual;
//        }
//        return this.evaluateEnsembles(seed, population);
//    }
}