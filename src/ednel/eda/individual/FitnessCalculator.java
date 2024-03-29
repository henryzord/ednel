package ednel.eda.individual;

import ednel.utils.PBILLogger;
import ednel.utils.analysis.CompilePredictions;
import ednel.utils.analysis.optimizers.AUTOCVEProcedure;
import ednel.utils.sorters.PopulationSorter;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import org.omg.CORBA.portable.UnknownException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.UnassignedClassException;
import weka.core.Utils;

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

    private EvaluationMetric metric;

    public enum EvaluationMethod {
            HOLDOUT, LEAVEONEOUT, CROSSVALIDATION
    };

    public enum EvaluationMetric {
        UNWEIGHTED_AUC, BALANCED_ACCURACY
    }

    private EvaluationMethod evaluation_method;

    private Integer[] sortedIndices_learn;
    private Integer[] sortedIndices_val;

    public FitnessCalculator(int n_folds, Instances learn_data, EvaluationMetric metric) throws Exception {
        this(n_folds, learn_data, null, metric);
    }

    public FitnessCalculator(int n_folds, Instances learn_data, Instances val_data, EvaluationMetric metric) throws Exception {
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

        this.metric = metric;
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

    public static double getBalancedAccuracy(
            Instances train_data, Instances val_data, AbstractClassifier clf) throws Exception {
        Evaluation evaluation = new Evaluation(train_data);
        evaluation.evaluateModel(clf, val_data);
        return FitnessCalculator.getBalancedAccuracy(evaluation);
    }

    public static double getBalancedAccuracy(Evaluation evaluation) throws Exception {
        double[][] confMatrix = evaluation.confusionMatrix();
        int n_classes = confMatrix.length;

        double sum = 0;

        for(int c = 0; c < n_classes; c++) {
            double n_instances_class = 0;
            for(int i = 0; i < n_classes; i++) {
                n_instances_class += confMatrix[c][i];
            }
            sum += confMatrix[c][c] / n_instances_class;
        }
        sum /= n_classes;
        return sum;
    }

    public Method getPreferredFitnessMethod() throws NoSuchMethodException {
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
        return fitnessMethod;
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

    private Fitness leaveOneOutEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual, boolean get_validation_fitness
    ) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        throw new UnknownException(new Exception("not implemented yet!"));
    }

    private Fitness holdoutEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual
    ) throws UnknownException {
        try {
            Individual copy = new Individual(ind, timeout_individual);
            Evaluation eval = new Evaluation(this.learn_data);

            copy.buildClassifier(this.learn_data);
            eval.evaluateModel(copy, this.val_data);

            double score;
            switch(this.metric) {
                case UNWEIGHTED_AUC:
                    score = getUnweightedAreaUnderROC(eval);
                    break;
                case BALANCED_ACCURACY:
                    score = getBalancedAccuracy(eval);
                    break;
                default:
                    throw new Exception("unrecognized metric.");
            }
            return new Fitness(copy.getNumberOfRules(), null, score);
        } catch(Exception e) {
            return new Fitness(null, null, null);
        }
    }

    private Fitness crossValidationEvaluateEnsemble(
            Random random, Individual ind, Integer timeout_individual, boolean get_validation_fitness
    ) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException, UnknownException, InterruptedException {
        EvaluateValidationSetThread t = null;

        if(get_validation_fitness) {
            t = new EvaluateValidationSetThread(this.learn_data, this.val_data, ind, timeout_individual, this.metric);
            t.start();
        }

        Object[] trainEvaluations = IntStream.range(0, this.n_folds).parallel().mapToObj(
                i -> FitnessCalculator.parallelFoldEvaluation(ind, learn_data, i, this.n_folds, random, timeout_individual)
        ).toArray();

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
            switch(this.metric) {
                case UNWEIGHTED_AUC:
                    learnQuality = fj.getUnweightedAUC("ensemble");
                    break;
                case BALANCED_ACCURACY:
                    learnQuality = fj.getBalancedAccuracy("ensemble");
                    break;
                default:
                    throw new Exception("unrecognized metric.");
            }
        } catch(Exception e) {
            learnQuality = 0;
        }

        if(get_validation_fitness) {
            return new Fitness(size, learnQuality, t.getValQuality());
        } else {
            return new Fitness(size, learnQuality, null);
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

    public Fitness getEnsembleValidationFitness(Individual ind)
            throws EmptyEnsembleException, NoAggregationPolicyException {

        if(ind.getFitness().getValQuality() == null) {
            EvaluateValidationSetThread t = new EvaluateValidationSetThread(
                    this.learn_data, this.val_data, ind, null, this.metric
            );
            t.run();
            return new Fitness(ind.getFitness().getSize(), ind.getFitness().getLearnQuality(), t.getValQuality());
        }
        return ind.getFitness();
    }

    public Integer[] getSortedIndices(Individual[] population)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {

        Double[] fitnesses = new Double[population.length];

        Method fitnessMethod = getPreferredFitnessMethod();

        for(int i = 0; i < population.length; i++) {
            fitnesses[i] = (Double)fitnessMethod.invoke(population[i].getFitness());
        }
        return PopulationSorter.simpleArgsort(population);
//        return PopulationSorter.lexicographicArgsort(fitnesses);
    }

    public static void main(String[] args) throws Exception {
        String[] datasets_names = {"15", "37", "307", "451", "458", "469", "1476", "1485", "23517", "40496", "40499", "40994"};


        for(String dataset : datasets_names) {
            for(int j = 1; j <= 10; j++) {
                HashMap<String, Instances> sets = AUTOCVEProcedure.loadHoldoutDataset(
                        "C:\\Users\\henry\\Projects\\autocve_experiments\\experiments_EVOSTAR21\\new\\partitions",
                        dataset, j);

                RandomForest rf = new RandomForest();
                rf.buildClassifier(sets.get("train_data"));

                double bacc = FitnessCalculator.getBalancedAccuracy(sets.get("train_data"), sets.get("test_data"), rf);
                System.out.printf("Dataset: %s, Balanced accuracy: %f\n", dataset, bacc);
            }
        }
    }

}