package eda;

import weka.classifiers.Evaluation;
import weka.core.Instances;

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

    public static double getUnweightedAreaUnderROC(Evaluation evaluation) {
        double [][] confusionMatrix = evaluation.confusionMatrix();
        int n_classes = confusionMatrix.length;
        double unweighted = 0;
        for(int i = 0; i < n_classes; i++) {
            unweighted += evaluation.areaUnderROC(i);
        }
        return unweighted / n_classes;
    }

    public double[][] evaluateEnsembles(int seed,
                                        String[][] j48Parameters, String[][] simpleCartParameters, String[][] reptreeParameters,
                                        String[][] partParameters, String[][] jripParameters, String[][] decisionTableParameters,
                                        String[][] aggregatorParameters) throws Exception {

        int n_individuals = j48Parameters.length;

        double[] trainEvaluations = new double [n_individuals];
        double[] testEvaluations = new double [n_individuals];

        for(int k = 0; k < n_individuals; k++) {
            trainEvaluations[k] = 0;
        }

        Random random = new Random(seed);

//        Instances data = new Instances(train_data);
//        data.randomize(random);
//        data.stratify(n_folds);
        Evaluation trainEval = new Evaluation(train_data);

        // do the folds
        for (int i = 0; i < n_folds; i++) {
            Instances local_train = train_data.trainCV(n_folds, i, random), local_val = train_data.testCV(n_folds, i);

            for(int j = 0; j < n_individuals; j++) {
                Individual individual = new Individual();
                individual.setOptions(new String[]{
                        "-J48", String.join(" ", j48Parameters[j]),
                        "-SimpleCart", String.join(" ", simpleCartParameters[j]),
                        "-REPTree", String.join(" ", reptreeParameters[j]),
                        "-PART", String.join(" ", partParameters[j]),
                        "-JRip", String.join(" ", jripParameters[j]),
                        "-DecisionTable", String.join(" ", decisionTableParameters[j]),
                        "-Aggregator", String.join(" ", aggregatorParameters[j])
                });

                individual.buildClassifier(local_train);
                trainEval.evaluateModel(individual, local_val);
                trainEvaluations[j] += getUnweightedAreaUnderROC(trainEval);

                if((i == n_folds - 1) && (test_data != null)) {
                    individual.buildClassifier(train_data);

                    Evaluation testEval = new Evaluation(test_data);
                    testEval.evaluateModel(individual, test_data);
                    testEvaluations[j] = getUnweightedAreaUnderROC(testEval);
                } else {
                    testEvaluations[j] = -1;
                }
            }
        }
        for(int k = 0; k < n_individuals; k++) {
            trainEvaluations[k] /= n_folds;
        }

        return new double[][]{trainEvaluations, testEvaluations};
    }

    public double[][] broadcastEvaluationBack(int[] seeds,
            String[] j48Parameters, String[] simpleCartParameters, String[] reptreeParameters,
            String[] partParameters, String[] jripParameters, String[] decisionTableParameters,
            String[] aggregatorParameters) throws Exception {

        double[] trainEvaluations = new double [seeds.length];
        double[] testEvaluations = new double [seeds.length];

        Individual individual = new Individual();
            individual.setOptions(new String[]{
                "-J48", String.join(" ", j48Parameters),
                "-SimpleCart", String.join(" ", simpleCartParameters),
                "-REPTree", String.join(" ", reptreeParameters),
                "-PART", String.join(" ", partParameters),
                "-JRip", String.join(" ", jripParameters),
                "-DecisionTable", String.join(" ", decisionTableParameters),
                "-Aggregator", String.join(" ", aggregatorParameters)
            });

        Evaluation trainEval = new Evaluation(train_data);
        for(int i = 0; i < seeds.length; i++) {
            Random random = new Random(seeds[i]);
            trainEval.crossValidateModel(individual, train_data, n_folds, random);  // formerly filtered

            trainEvaluations[i] = getUnweightedAreaUnderROC(trainEval);

            if(test_data != null) {
                individual.buildClassifier(train_data);
                Evaluation testEval = new Evaluation(test_data);
                testEval.evaluateModel(individual, test_data);
                testEvaluations[i] = getUnweightedAreaUnderROC(testEval);
            } else {
                testEvaluations[i] = -1;
            }
        }

        return new double[][]{trainEvaluations, testEvaluations};
    }

    /**
     * runs the program, the command line looks like this:<br/>
     * eda.EDAEvaluator CLASSIFIER classname [options] FILTER classname [options] DATASET
     * filename <br/>
     * e.g., <br/>
     * java -classpath ".:weka.jar" eda.EDAEvaluator \<br/>
     * CLASSIFIER weka.classifiers.trees.J48 -U \<br/>
     * FILTER weka.filters.unsupervised.instance.Randomize \<br/>
     * DATASET iris.arff<br/>
     */
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        Instances train_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff")));
        Instances test_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tst.arff")));
        train_data.setClassIndex(train_data.numAttributes() - 1);
        test_data.setClassIndex(test_data.numAttributes() - 1);

        String[][] j48Parameters           = new String[][]{{"-S -C 0.5 -B -M 7 -A "}};
        String[][] simpleCartParameters    = new String[][]{{"-M 2 -N 5 -C 1 -S 1"}};
        String[][] reptreeParameters       = new String[][]{{""}};
        String[][] partParameters          = new String[][]{{"-U -B -M 9 -doNotMakeSplitPointActualValue -J -Q 1"}};
        String[][] jripParameters          = new String[][]{{"-F 3 -N 10.0 -O 2 -E -P -S 1"}};
        String[][] decisionTableParameters = new String[][]{{"-X 1 -E auc -I -R -S weka.attributeSelection.GreedyStepwise -B -T -1.7976931348623157E308 -N -1 -num-slots 1"}};
        String[][] aggregatorParameters    = new String[][]{{"MajorityVotingAggregator"}};

        // String[][] j48Parameters           = new String[][]{{"-S -C 0.5 -B -M 7 -A -Q 1"}};
        // String[][] simpleCartParameters    = new String[][]{{"-M 2 -N 5 -C 1 -S 1"}};
        // String[][] partParameters          = new String[][]{{"-U -B -M 9 -doNotMakeSplitPointActualValue -J -Q 1"}};
        // String[][] jripParameters          = new String[][]{{"-F 3 -N 10.0 -O 2 -E -P -S 1"}};
        // String[][] reptreeParameters       = new String[][]{{"-M 8 -V 0.001 -N 4 -S 1 -L 3 -I 0.0"}};
        // String[][] decisionTableParameters = new String[][]{{"-X 1 -E auc -I -R -S weka.attributeSelection.GreedyStepwise -B -T -1.7976931348623157E308 -N -1 -num-slots 1"}};



        FitnessCalculator evaluator = new FitnessCalculator(5, train_data, test_data);
        int seed = 0;

        double[][] aucs;

        aucs = evaluator.evaluateEnsembles(
                seed, j48Parameters, simpleCartParameters, reptreeParameters,
                partParameters, jripParameters, decisionTableParameters, aggregatorParameters
        );
        System.out.println("train auc: " + aucs[0][0]);
        System.out.println("test auc: " + aucs[1][0]);

        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        System.out.println("Elapsed time: " + duration/1000000 + " miliseconds");
    }
}