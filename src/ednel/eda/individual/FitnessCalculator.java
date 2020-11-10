package ednel.eda.individual;

import ednel.RandomSearch;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.nio.file.NotLinkException;
import java.security.InvalidParameterException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.TimeoutException;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * A little demo java program for using WEKA.<br/>
 * Check out the Evaluation class for more details.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 * @see Evaluation
 */
public class FitnessCalculator {

    private Instances train_data;
    private int n_folds;

    public FitnessCalculator(int n_folds, Instances train_data) {
        this.train_data = train_data;
        this.n_folds = n_folds;
        this.train_data.stratify(this.n_folds);
    }

    public static double getUnweightedAreaUnderROC(
            Instances train_data, Instances test_data, AbstractClassifier clf) throws Exception {
        Evaluation evaluation = new Evaluation(train_data);
        evaluation.evaluateModel(clf, test_data);
        return FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
    }

    public static double getUnweightedAreaUnderROC(Evaluation evaluation) throws Exception {
        int n_classes = evaluation.confusionMatrix().length;
        double unweighted = 0;
        for(int i = 0; i < n_classes; i++) {
            try {
                double auc = evaluation.areaUnderROC(i);
                if(Utils.isMissingValue(auc)) {
                    throw new Exception("un-stratified code!");
//                unweighted += 0;
                } else {
//              unweighted += evaluation.areaUnderROC(i);
                    unweighted += auc;
                }
            } catch(NullPointerException npe) {
                throw npe;
            }
        }
        return unweighted / n_classes;
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param seed
     * @param ind
     * @return
     * @throws Exception
     */
    public Double evaluateEnsemble(int seed, Individual ind) throws Exception {
        Random random = new Random(seed);
        return this.evaluateEnsemble(random, ind, null);
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param random
     * @param ind
     * @return
     * @throws Exception
     */
    public Double evaluateEnsemble(Random random, Individual ind) throws Exception {
        return this.evaluateEnsemble(random, ind, null);
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param seed
     * @param ind
     * @return
     * @throws Exception
     */
    public Double evaluateEnsemble(int seed, Individual ind, Integer timeout_individual) throws Exception {
        Random random = new Random(seed);
        return this.evaluateEnsemble(random, ind, timeout_individual);
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param random
     * @param ind
     * @return
     * @throws Exception
     */
    public Double evaluateEnsemble(Random random, Individual ind, Integer timeout_individual) throws Exception {
        double trainEvaluation = 0.0;

        Object[] trainEvaluations = IntStream.range(0, n_folds).parallel().mapToObj(
                i -> FitnessCalculator.parallelFoldEvaluation(ind, train_data, i, n_folds, random, timeout_individual)).toArray();

        for(Object val : trainEvaluations) {
            if(val instanceof Double) {
                trainEvaluation += (Double)val;
            } else {
                throw (Exception)val;
            }
        }
        trainEvaluation /= n_folds;

        // TODO traditional method - it works
//        for (int i = 0; i < n_folds; i++) {
//            Instances local_train = train_data.trainCV(n_folds, i, random);
//            Instances local_val = train_data.testCV(n_folds, i);
//
//            ind.buildClassifier(local_train);
//            trainEval.evaluateModel(ind, local_val);
//            trainEvaluation += getUnweightedAreaUnderROC(trainEval);
//
//        }
//        trainEvaluation /= n_folds;
        // TODO traditional method - it works

        return trainEvaluation;
    }

    public static Object parallelFoldEvaluation(
            Individual ind, Instances train_data,
            int n_fold, int n_folds, Random random, Integer timeout_individual) {
        try {
            LocalDateTime start = LocalDateTime.now();

            Individual copy = new Individual(ind, timeout_individual);

            Evaluation eval = new Evaluation(train_data);

            Instances local_train = train_data.trainCV(n_folds, n_fold, random);
            Instances local_val = train_data.testCV(n_folds, n_fold);

            copy.buildClassifier(local_train);

//            if((timeout_individual != null) &&
//                    ((int)start.until(LocalDateTime.now(), ChronoUnit.SECONDS) > timeout_individual)) {
//                throw new TimeoutException("Individual evaluation took longer than allowed time.");
//            }
            eval.evaluateModel(copy, local_val);

            // TODO trying to implement complexity
//            double max_complexity = Math.ceil(
//                    (Math.log(2 * train_data.size() - 1)/Math.log(2)) - 1
//            );
//            double this_solution_complexity = Math.min(Math.max(0, Math.ceil(
//                    (Math.log(2 * copy.getNumberOfRules() - 1)/Math.log(2)) - 1
//            )), max_complexity);

            if(copy.n_rules == 0) {
                return 0.0;
            } else {
                return getUnweightedAreaUnderROC(eval) - copy.n_rules / train_data.size();
            }
        } catch(Exception e) {
            return e;
        }
    }


    public static void main(String[] args) {
        try {
            J48 j48 = new J48();
            j48.setConfidenceFactor((float)0.4);

            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource("C:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv\\iris\\iris-10-1tra.arff");
            ConverterUtils.DataSource test_set = new ConverterUtils.DataSource("C:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv\\iris\\iris-10-1tst.arff");

            Instances train_data = train_set.getDataSet();
            Instances test_data = test_set.getDataSet();
            train_data.setClassIndex(train_data.numAttributes() - 1);
            test_data.setClassIndex(test_data.numAttributes() - 1);

            j48.buildClassifier(train_data);
            Evaluation ev = new Evaluation(train_data);
            ev.evaluateModel(j48, test_data);

            for(int i = 0; i < train_data.numClasses(); i++) {
                System.out.println(ev.areaUnderROC(i));
            }
        } catch(Exception e) {
            System.err.println(e.getMessage());
        }

    }
}