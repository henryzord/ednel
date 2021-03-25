package ednel.utils.analysis.optimizers;

import ednel.eda.EDNEL;
import ednel.eda.individual.EmptyEnsembleException;
import ednel.utils.analysis.CompilePredictions;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class NCVMatrixHandler {
    double[] actualClasses;

    double[][] lastPredictionMatrix;
    double[][] overallPredictionMatrix;
    double[][] predictionMatrix;

    Boolean bestUsersOverall;
    Double auc;

    int counter_instance;

    NestedCrossValidation.SupportedAlgorithms algorithmName;

    public NCVMatrixHandler(Instances data, NestedCrossValidation.SupportedAlgorithms algorithmName) {
        this.actualClasses = new double[data.size()];
        this.algorithmName = algorithmName;
        this.counter_instance = 0;
        this.bestUsersOverall = null;
        this.auc = null;

        switch(algorithmName) {
            case EDNEL:
            case PBIL:
                lastPredictionMatrix = new double[data.size()][];
                overallPredictionMatrix = new double[data.size()][];
                predictionMatrix = null;
                break;
            case RandomForest:
            case AUTOCVE:
                predictionMatrix = new double[data.size()][];
                lastPredictionMatrix = null;
                overallPredictionMatrix = null;
                break;
        }

    }

    public void handle(AbstractClassifier abstractClassifier, Instances data) throws Exception {
        switch(algorithmName) {
            case EDNEL:
                double[][] lastFoldPreds;
                double[][] overallFoldPreds;

                try {
                    lastFoldPreds = ((EDNEL)abstractClassifier).getCurrentGenBest().distributionsForInstances(data);
                    overallFoldPreds = ((EDNEL)abstractClassifier).getOverallBest().distributionsForInstances(data);
                } catch (EmptyEnsembleException eee) {
                    lastFoldPreds = new double[data.size()][];
                    overallFoldPreds = new double[data.size()][];

                    int n_classes = data.numClasses();

                    for(int j = 0; j < data.size(); j++) {
                        lastFoldPreds[j] = new double[n_classes];
                        overallFoldPreds[j] = new double[n_classes];
                        for(int k = 0; k < n_classes; k++) {
                            lastFoldPreds[j][k] = 0.0f;
                            overallFoldPreds[j][k] = 0.0f;
                        }
                    }
                }
                for(int j = 0; j < data.size(); j++) {
                    this.lastPredictionMatrix[this.counter_instance] = lastFoldPreds[j];
                    this.overallPredictionMatrix[this.counter_instance] = overallFoldPreds[j];
                    this.actualClasses[this.counter_instance] = data.instance(j).value(data.classIndex());
                    this.counter_instance += 1;
                }
                break;
            case RandomForest:
                double[][] preds = abstractClassifier.distributionsForInstances(data);

                for(int j = 0; j < data.size(); j++) {
                    this.predictionMatrix[this.counter_instance] = preds[j];
                    this.actualClasses[this.counter_instance] = data.instance(j).value(data.classIndex());
                    this.counter_instance += 1;
                }
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public void compile() throws Exception {
        switch(this.algorithmName) {
            case EDNEL:
                CompilePredictions lastFJ = new CompilePredictions(this.lastPredictionMatrix, this.actualClasses, "LastClassifier");
                CompilePredictions overallFJ = new CompilePredictions(this.overallPredictionMatrix, this.actualClasses, "OverallClassifier");

                double lastAUC = lastFJ.getAUC("LastClassifier");
                double overallAUC = overallFJ.getAUC("OverallClassifier");

                this.bestUsersOverall = overallAUC >= lastAUC;
                this.auc = Math.max(overallAUC, lastAUC);
                break;
            case RandomForest:
                String tempClfName = "classifier";
                CompilePredictions foldJoiner = new CompilePredictions(predictionMatrix, actualClasses, tempClfName);
                this.auc = foldJoiner.getAUC(tempClfName);
                break;

            default:
                throw new NotImplementedException();
        }
    }

    public Boolean getBestUsersOverall() {
        return bestUsersOverall;
    }

    public Double getAuc() {
        return auc;
    }
}
