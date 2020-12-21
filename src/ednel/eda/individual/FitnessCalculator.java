package ednel.eda.individual;

import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import org.apache.commons.math3.random.MersenneTwister;
import org.omg.CORBA.portable.UnknownException;
import smile.neighbor.lsh.Hash;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.TimeoutException;
import java.util.stream.IntStream;

/**
 * A little demo java program for using WEKA.<br/>
 * Check out the Evaluation class for more details.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 * @see Evaluation
 */
public class FitnessCalculator {

    private Instances learn_data;
    private Instances val_data;
    private int n_folds;

    public FitnessCalculator(int n_folds, Instances learn_data) {
        this(n_folds, learn_data, null);
    }

    public FitnessCalculator(int n_folds, Instances learn_data, Instances val_data) {
        this.learn_data = learn_data;
        this.val_data = val_data;

        this.n_folds = n_folds;
    }

    public static double getUnweightedAreaUnderROC(
            Instances train_data, Instances val_data, AbstractClassifier clf) throws Exception {
        Evaluation evaluation = new Evaluation(train_data);
        evaluation.evaluateModel(clf, val_data);
        return FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
    }

    public static double getUnweightedAreaUnderROC(Evaluation evaluation) throws Exception {
        int n_classes = evaluation.confusionMatrix().length;
        double unweighted = 0;
        for(int i = 0; i < n_classes; i++) {
            double auc = evaluation.areaUnderROC(i);
            if(Utils.isMissingValue(auc)) {
                throw new Exception("un-stratified code!");
            } else {
                unweighted += auc;
            }
        }
        return unweighted / n_classes;
    }

    public static Instances betterStratifier(Instances data, int n_folds) {
        if (n_folds <= 1) {
            throw new IllegalArgumentException(
                    "Number of folds must be greater than 1");
        }
        if (data.classIndex() < 0) {
            throw new UnassignedClassException("Class index is negative (not set)!");
        }
        if (!data.attribute(data.classIndex()).isNominal()) {
            throw new ValueException("only stratifies classification datasets!");
        }

        HashMap<Double, ArrayList<Integer>> mapping = new HashMap<>();

        // collects indices of instances for classes
        for(int i = 0; i < data.size(); i++) {
            ArrayList<Integer> indices;
            double index = data.instance(i).classValue();
            if(mapping.getOrDefault(index, null) == null) {
                indices = new ArrayList<>();
            } else {
                indices = mapping.get(index);
            }
            indices.add(i);
            mapping.put(index, indices);
        }

        HashMap<Double, ArrayList<Integer>> copy = new HashMap<>();
        // how many instances from each class to put at a given time in each fold
        HashMap<Double, Integer> howManies = new HashMap<>();
        // shuffles arrays
        for(Double key : mapping.keySet()) {
            Collections.shuffle(mapping.get(key));
            copy.put(key, mapping.get(key));

            int howMany = Math.round(copy.get(key).size() / (float)n_folds);
            if(howMany <= 0) {
                throw new ValueException(String.format(
                    "class %s has %d instances, but number of folds is %d",
                    data.classAttribute().value(key.intValue()),
                    copy.get(key).size(),
                    n_folds
                ));
            }
            howManies.put(key, howMany);

        }

        // get attribute info
        ArrayList<Attribute> attrInfo = new ArrayList<>();
        for(int j = 0; j < data.numAttributes(); j++) {
            attrInfo.add(data.attribute(j));
        }
        Instances container = new Instances(data.relationName(), attrInfo, data.size());
        for(int f = 0; f < n_folds; f++) {
            for(Double key : howManies.keySet()) {
                for(int h = 0; h < howManies.get(key); h++) {
                    try {
                        int index = mapping.get(key).remove(0);
                        container.add(data.get(index));
                    } catch(IndexOutOfBoundsException iobe) {
                        // Math.round can generate a number of instances above the available quantity; just ignore
                        // and proceed, this exception happens at the last fold
                    }
                }
            }
        }
        container.setClassIndex(data.classIndex());
        return container;
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param seed
     * @param ind
     * @return
     * @throws Exception
     */
    public Fitness evaluateEnsemble(int seed, Individual ind) throws
            EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
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
    public Fitness evaluateEnsemble(Random random, Individual ind) throws
            EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
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
    public Fitness evaluateEnsemble(int seed, Individual ind, Integer timeout_individual) throws
            EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
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
    public Fitness evaluateEnsemble(Random random, Individual ind, Integer timeout_individual) throws
            EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        double learnQuality = 0.0;
        int size = 0;

        EvaluateValidationSetThread t = new EvaluateValidationSetThread(this.learn_data, this.val_data, ind, timeout_individual);
        t.start();

        Object[] trainEvaluations = IntStream.range(0, n_folds).parallel().mapToObj(
                i -> FitnessCalculator.parallelFoldEvaluation(ind, learn_data, i, n_folds, random, timeout_individual)).toArray();

        // waits for evaluation of validation set to finish
        t.join();

        for(Object val : trainEvaluations) {
            if(val instanceof Fitness) {
                learnQuality += ((Fitness)val).getLearnQuality();
                size += ((Fitness)val).getSize();
            } else if(val instanceof EmptyEnsembleException) {
                throw (EmptyEnsembleException)val;
            } else if(val instanceof NoAggregationPolicyException) {
                throw (NoAggregationPolicyException)val;
            } else if(val instanceof TimeoutException) {
                throw (TimeoutException)val;
            } else {
                throw new UnknownException(((Exception)val));
            }
        }
        learnQuality /= n_folds;
        size /= n_folds;

        return new Fitness(size, learnQuality, t.getValQuality());
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

//            double max_complexity = Math.ceil(
//                    (Math.log(2 * train_data.size() - 1)/Math.log(2)) - 1
//            );
//            double this_solution_complexity = Math.min(Math.max(0, Math.ceil(
//                    (Math.log(2 * copy.getNumberOfRules() - 1)/Math.log(2)) - 1
//            )), max_complexity);

            return new Fitness(copy.getNumberOfRules(), getUnweightedAreaUnderROC(eval));
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