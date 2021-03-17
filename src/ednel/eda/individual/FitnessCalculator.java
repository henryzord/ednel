package ednel.eda.individual;

import ednel.utils.PBILLogger;
import ednel.utils.analysis.CompilePredictions;
import ednel.utils.sorters.PopulationSorter;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import org.omg.CORBA.portable.UnknownException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.TimeoutException;
import java.util.stream.IntStream;


/**
 * Class for calculating fitness of individuals from EDA.
 */
public class FitnessCalculator {

    /** Data used to train classifiers. Might be split into subsets if performing an internal n-fold cross-validation */
    private Instances learn_data;
    /** Data to measure validation fitness */
    private Instances val_data;
    /** Number of internal folds */
    private int n_folds;

    public enum EvaluationMethod {
            HOLDOUT, LEAVEONEOUT, CROSSVALIDATION;
    };
    private EvaluationMethod evaluation_method;

    private Integer[] sortedIndices_learn;
    private Integer[] sortedIndices_val;

    public FitnessCalculator(int n_folds, Instances learn_data) throws Exception {
        this(n_folds, learn_data, null);
    }

    public FitnessCalculator(int n_folds, Instances learn_data, Instances val_data) throws Exception {
        this.learn_data = learn_data;
        this.val_data = val_data;

        this.n_folds = n_folds;

        this.sortedIndices_learn = new Integer[0];
        this.sortedIndices_val = new Integer[0];

        if(n_folds < 0) {
            throw new Exception("Number of folds cannot be less than zero!");
        }

        switch(n_folds) {
            case 0:
                this.evaluation_method = EvaluationMethod.HOLDOUT;
                break;
            case 1:
                this.evaluation_method = EvaluationMethod.LEAVEONEOUT;
                break;
            default:
                this.evaluation_method = EvaluationMethod.CROSSVALIDATION;
        }
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

    /**
     * Stratifies data for posterior use of a cross-validation procedure.
     *
     * @param data Data to be stratified
     * @param n_folds Number of folds that will be used in cross-validation
     * @return Same dataset as data, but now stratified
     * @throws IllegalArgumentException Invalid number of folds
     * @throws UnassignedClassException Class index is not set
     * @throws ValueException Is not a classification dataset
     */
    public static Instances betterStratifier(Instances data, int n_folds)
            throws IllegalArgumentException, UnassignedClassException, ValueException {
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
    public Fitness evaluateEnsemble(int seed, Individual ind, Integer timeout_individual, boolean get_validation_fitness) throws
            EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        Random random = new Random(seed);
        return this.evaluateEnsemble(random, ind, timeout_individual, get_validation_fitness);
    }

    /**
     * Evaluates ensemble -- that is, returns the fitness function for this individual.
     *
     * @param random
     * @param ind
     * @return
     * @throws Exception
     */
    public Fitness evaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual, boolean get_validation_fitness
    ) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {

        switch(this.evaluation_method) {
            case HOLDOUT:
                return holdoutEvaluateEnsemble(random, ind, timeout_individual);
            case LEAVEONEOUT:
                return leaveOneOutEvaluateEnsemble(random, ind, timeout_individual, get_validation_fitness);
            case CROSSVALIDATION:
                return crossValidationEvaluateEnsemble(random, ind, timeout_individual, get_validation_fitness);
            default:
                throw new UnknownException(new Exception("unknown evaluation methodology"));
        }
    }

    private Fitness holdoutEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual
    ) throws UnknownException {
        try {
            Individual copy = new Individual(ind, timeout_individual);
            Evaluation eval = new Evaluation(this.learn_data);

            copy.buildClassifier(this.learn_data);
            eval.evaluateModel(copy, this.val_data);

            double auc = getUnweightedAreaUnderROC(eval);
            return new Fitness(copy.getNumberOfRules(), auc, auc);
        } catch(Exception e) {
            return new Fitness(0, 0, 0);
        }
    }

    private Fitness leaveOneOutEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual, boolean get_validation_fitness
    ) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        throw new UnknownException(new Exception("not implemented yet!"));
    }

    private Fitness crossValidationEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual, boolean get_validation_fitness
    ) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        EvaluateValidationSetThread t = null;

        if(get_validation_fitness) {
            t = new EvaluateValidationSetThread(this.learn_data, this.val_data, ind, timeout_individual);
            t.start();
        }

        Object[] trainEvaluations = IntStream.range(0, this.n_folds).parallel().mapToObj(
                i -> FitnessCalculator.parallelFoldEvaluation(ind, learn_data, i, this.n_folds, random, timeout_individual)).toArray();

        // waits for evaluation of validation set to finish
        if(get_validation_fitness) {
            t.join();
        }

        ArrayList<String> all_lines = new ArrayList<>();
        int size = 0;

        for(Object val : trainEvaluations) {
            if(val instanceof PredictionsSizeContainer) {
                all_lines.addAll(((PredictionsSizeContainer)val).getLines());
                size += ((PredictionsSizeContainer)val).getNumberOfRules();
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
        size /= n_folds;
        double learnQuality;
        try {
            CompilePredictions fj = new CompilePredictions(all_lines);
            learnQuality = fj.getAUC("ensemble");
        } catch(Exception e) {
            learnQuality = 0;
        }

        if(get_validation_fitness) {
            return new Fitness(size, learnQuality, t.getValQuality());
        } else {
            return new Fitness(size, learnQuality);
        }
    }


    public static Object parallelFoldEvaluation(
            Individual ind, Instances train_data,
            int n_fold, int n_folds, Random random, Integer timeout_individual) {
        try {
            // LocalDateTime start = LocalDateTime.now();

            Individual copy = new Individual(ind, timeout_individual);
            // Evaluation eval = new Evaluation(train_data);

            Instances local_train = train_data.trainCV(n_folds, n_fold, random);
            Instances local_val = train_data.testCV(n_folds, n_fold);

            copy.buildClassifier(local_train);

            double[][] dists = copy.distributionsForInstances(local_val);
            ArrayList<String> lines = new ArrayList<>(dists.length);
            lines.add("classValue;ensemble\n");
            for(int i = 0; i < local_val.size(); i++) {
                lines.add(local_val.instance(i).classValue() + ";" + PBILLogger.writeDistributionOfProbabilities(dists[i]) + "\n");
            }
            return new PredictionsSizeContainer(copy.getNumberOfRules(), lines);
            // eval.evaluateModel(copy, local_val);
            // return new Fitness(copy.getNumberOfRules(), getUnweightedAreaUnderROC(eval));
        } catch(Exception e) {
            return e;
        }
    }

    public Fitness getEnsembleValidationFitness(Individual ind) throws EmptyEnsembleException, NoAggregationPolicyException {
        EvaluateValidationSetThread t = new EvaluateValidationSetThread(this.learn_data, this.val_data, ind, null);
        t.run();
        Fitness fit = new Fitness(ind.getFitness().getSize(), ind.getFitness().getLearnQuality(), t.getValQuality());
        return fit;
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

    public Integer[] getSortedIndices(Individual[] population) throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Double[] fitnesses = new Double[population.length];

        Method fitnessMethod;

        switch(this.evaluation_method) {
            case HOLDOUT:
                fitnessMethod = Fitness.class.getMethod("getValQuality");
                break;
            case LEAVEONEOUT:
            case CROSSVALIDATION:
                fitnessMethod = Fitness.class.getMethod("getLearnQuality");
                break;
            default:
                throw new IllegalStateException("Unexpected value: " + this.evaluation_method);
        }

        for(int i = 0; i < population.length; i++) {
            fitnesses[i] = (double)fitnessMethod.invoke(population[i].getFitness());
        }

        return PopulationSorter.lexicographicArgsort(fitnesses);
    }
}