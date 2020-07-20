package ednel.eda.aggregators;

import ednel.eda.individual.FitnessCalculator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

public class MajorityVotingAggregator extends Aggregator implements Serializable {

    public MajorityVotingAggregator() {
    }


    /**
     * Given a list of classifiers and test data, this function will make each classifier cast predictions on the data,
     * and then aggregate the votes based on the competence of each classifier.
     *
     * For majority voting, all classifiers weight the same in predicting a new instance.
     *
     * The fusion aggregation performed is described in Equation 5.17 (p 150) of the Book
     *
     * Kuncheva, Ludmila I. Combining pattern eda.classifiers: methods and algorithms. John Wiley & Sons, 2004.
     *
     * The fusion function is:
     *
     * .. math:: \mu(\mathbf{x}) = argmax_{j \in C}( 1/B \sum_{i \in B} d_{i,j}(\mathbf{x}))
     *
     * where :math:`\mu(\mathbf{x})` is the prediction of instance :math:`\mathbf{x}`, :math:`B` the number of base
     * eda.classifiers, :math:`C` the number of classes, and :math:`d_{i,j}` the confidence of the :math:`j`-th classifier
     * that the instance belongs to the :math:`i`-th class.
     *
     * This basically translates to assigning the class with the largest average support among all base classifiers.
     */
    @Override
    public double[][] aggregateProba(AbstractClassifier[] clfs, Instances batch) throws Exception {
        int n_active_classifiers = this.getActiveClassifiersCount(clfs);

        double[][][] dists = new double[n_active_classifiers][][];
        int cc = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[cc] != null) {
                dists[counter] = clfs[cc].distributionsForInstances(batch);
                counter += 1;
            }
            cc += 1;
        }

        int n_instances = batch.size(),
                n_classes = dists[0][0].length;

        double[][] finalDistribution = new double [n_instances][n_classes];

        for(int j = 0; j < n_instances; j++) {
            double sum = 0;
            for(int k = 0; k < n_classes; k++) {
                finalDistribution[j][k] = 0;
                for(int i = 0; i < n_active_classifiers; i++) {
                    finalDistribution[j][k] += dists[i][j][k] * competences[i];
                }
                finalDistribution[j][k] /= n_active_classifiers;
                sum += finalDistribution[j][k];
            }
            // normalizes
            for(int k = 0; k < n_classes; k++) {
                finalDistribution[j][k] /= sum;
            }
        }
        return finalDistribution;
    }

    /**
     * Given a list of classifiers and test data, this function will make each classifier cast predictions on the data,
     * and then aggregate the votes based on the competence of each classifier.
     *
     * For majority voting, all classifiers weight the same in predicting a new instance.
     *
     * The fusion aggregation performed is described in Equation 5.17 (p 150) of the Book
     *
     * Kuncheva, Ludmila I. Combining pattern eda.classifiers: methods and algorithms. John Wiley & Sons, 2004.
     *
     * The fusion function is:
     *
     * .. math:: \mu(\mathbf{x}) = argmax_{j \in C}( 1/B \sum_{i \in B} d_{i,j}(\mathbf{x}))
     *
     * where :math:`\mu(\mathbf{x})` is the prediction of instance :math:`\mathbf{x}`, :math:`B` the number of base
     * eda.classifiers, :math:`C` the number of classes, and :math:`d_{i,j}` the confidence of the :math:`j`-th classifier
     * that the instance belongs to the :math:`i`-th class.
     *
     * This basically translates to assigning the class with the largest average support among all base classifiers.
     */
    @Override
    public double[] aggregateProba(AbstractClassifier[] clfs, Instance instance) throws Exception {
        int n_active_classifiers = this.getActiveClassifiersCount(clfs);

        double[][] dists = new double[n_active_classifiers][];
        int cc = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[cc] != null) {
                dists[counter] = clfs[cc].distributionForInstance(instance);
                counter += 1;
            }
            cc += 1;
        }

        int n_classes = dists[0].length;

        double[] finalDistribution = new double [n_classes];

        double sum = 0;
        for(int k = 0; k < n_classes; k++) {
            finalDistribution[k] = 0;
            for(int i = 0; i < n_active_classifiers; i++) {
                finalDistribution[k] += dists[i][k] * competences[i];
            }
            finalDistribution[k] /= n_active_classifiers;
            sum += finalDistribution[k];
        }
        // normalizes
        for(int k = 0; k < n_classes; k++) {
            finalDistribution[k] /= sum;
        }

        return finalDistribution;
    }

    /**
     * Assigns the same competence for each and every classifier in the ensemble.
     * @param clfs
     * @param train_data
     * @throws Exception
     */
    @Override
    public void setCompetences(AbstractClassifier[] clfs, Instances train_data) throws Exception {
        int n_active_classifiers = this.getActiveClassifiersCount(clfs);

        this.competences = new double[n_active_classifiers];
        int i = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[i] != null) {
                this.competences[counter] = 1.0;
                counter += 1;
            }
            i += 1;
        }
    }
}
