package ednel.eda.individual;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

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
            if(Utils.isMissingValue(evaluation.areaUnderROC(i))) {
                throw new Exception("un-stratified code!");
//                unweighted += 0;
            } else {
              unweighted += evaluation.areaUnderROC(i);
            }
        }
        return unweighted / n_classes;
    }

    public Double evaluateEnsemble(int seed, Individual ind) throws Exception {
        Random random = new Random(seed);
        Evaluation trainEval = new Evaluation(train_data);
        return this.evaluateEnsemble(seed, ind, random, trainEval);
    }

    public Double evaluateEnsemble(int seed, Individual ind, Random random, Evaluation trainEval) throws Exception {
        double trainEvaluation = 0.0;

        for (int i = 0; i < n_folds; i++) {
            Instances local_train = train_data.trainCV(n_folds, i, random);
            Instances local_val = train_data.testCV(n_folds, i);

            ind.buildClassifier(local_train);
            trainEval.evaluateModel(ind, local_val);
            trainEvaluation += getUnweightedAreaUnderROC(trainEval);

        }
        trainEvaluation /= n_folds;
        return trainEvaluation;
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