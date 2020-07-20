package ednel.eda.aggregators;

import java.io.Serializable;

public class MajorityVotingAggregator extends Aggregator implements Serializable {

    public MajorityVotingAggregator() {
    }

    /**
     * Given a set of predictions, this function will return a matrix where each row is an instance, and each column
     * is a class. Each class value denotes the support of the whole ensemble for that class.
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
     * This basically translates to assigning the class with the largest average support among all base eda.classifiers.
     *
     * @param distributions: A 3D matrix with one axis for eda.classifiers, another for instances, and the last for classes.
     * @return a 2D matrix where each row is an instance and each column a class.
     */
    public double[][] aggregateProba(double[][][] distributions) {
        int
            n_classifiers = distributions.length,
            n_instances = distributions[0].length,
            n_classes = distributions[0][0].length;

        double[][] finalDistribution = new double [n_instances][n_classes];

        for(int j = 0; j < n_instances; j++) {
            double sum = 0;
            for(int k = 0; k < n_classes; k++) {
                finalDistribution[j][k] = 0;
                for(int i = 0; i < n_classifiers; i++) {
                    finalDistribution[j][k] += distributions[i][j][k];
                }
                finalDistribution[j][k] /= n_classifiers;
                sum += finalDistribution[j][k];
            }
            // normalizes
            for(int k = 0; k < n_classes; k++) {
                finalDistribution[j][k] /= sum;
            }
        }
        return finalDistribution;
    }

    @Override
    protected void setOptions(Object... args) {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
