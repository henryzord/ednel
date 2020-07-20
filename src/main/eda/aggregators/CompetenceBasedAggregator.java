package eda.aggregators;

import java.io.Serializable;

/**
 * It is a weighted majority voting aggregator.
 */
public class CompetenceBasedAggregator extends Aggregator implements Serializable {

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
                    finalDistribution[j][k] += distributions[i][j][k] * competences[i];
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
