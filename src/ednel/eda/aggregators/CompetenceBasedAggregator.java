package ednel.eda.aggregators;

import ednel.eda.individual.FitnessCalculator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

/**
 * It is a weighted majority voting aggregator.
 */
public class CompetenceBasedAggregator extends MajorityVotingAggregator implements Serializable {

    /**
     * Assign competence do classifiers based on how well they perform on training data, based on AUC.
     * @param clfs List of classifiers
     * @param train_data Data to measure competences from
     * @throws Exception
     */
    @Override
    public void setCompetences(AbstractClassifier[] clfs, Instances train_data) throws Exception {
        int n_active_classifiers = this.getActiveClassifiersCount(clfs);

        this.competences = new double[n_active_classifiers];
        int i = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[i] != null) {
                Evaluation evaluation = new Evaluation(train_data);
                evaluation.evaluateModel(clfs[i], train_data);
                this.competences[counter] = FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
                counter += 1;

            }
            i += 1;
        }
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
