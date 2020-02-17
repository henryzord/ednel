/**
 * An individual with default hyper-parameters from Weka for each one of its classifiers.
 */

package eda.individual;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class BaselineIndividual extends Individual {

    public BaselineIndividual(Instances train_data) throws Exception {
        super();

        String j48Parameters            = "-C 0.25 -M 2";
        String simpleCartParameters     = "-M 2 -N 5 -C 1 -S 1";
        String partParameters           = "-M 2 -C 0.25 -Q 1";
        String jripParameters           = "-F 3 -N 2.0 -O 2 -S 1";
        String decisionTableParameters  = "-R -X 1 -S weka.attributeSelection.BestFirst";
        String bestFirstParameters      = "-D 1 -N 5";
        String greedyStepwiseParameters = "";
        String aggregatorParameters     = "MajorityVotingAggregator";

        String[] options = {
                "-J48", j48Parameters, "-SimpleCart", simpleCartParameters, "-PART", partParameters,
                "-JRip", jripParameters, "-DecisionTable", decisionTableParameters,
                "-Aggregator", aggregatorParameters, "-GreedyStepwise", greedyStepwiseParameters,
                "-BestFirst", bestFirstParameters
        };

        this.setOptions(options);
        this.buildClassifier(train_data);

        this.characteristics.put("Aggregator", "MajorityVotingAggregator");
        this.characteristics.put("J48", "true");
        this.characteristics.put("J48_binarySplits", "false");
        this.characteristics.put("J48_useLaplace", "true");
        this.characteristics.put("J48_minNumObj", "2");
        this.characteristics.put("J48_useMDLcorrection", "false");
        this.characteristics.put("J48_collapseTree", "true");
        this.characteristics.put("J48_doNotMakeSplitPointActualValue", "false");
        this.characteristics.put("J48_pruning", "J48_confidenceFactor");
        this.characteristics.put("J48_numFolds", null);
        this.characteristics.put("J48_subtreeRaising", "true");
        this.characteristics.put("J48_seed", null);
        this.characteristics.put("J48_confidenceFactorValue", "0.25");
        this.characteristics.put("SimpleCart", "true");
        this.characteristics.put("SimpleCart_heuristic", "true");
        this.characteristics.put("SimpleCart_minNumObj", "2");
        this.characteristics.put("SimpleCart_usePrune", "true");
        this.characteristics.put("SimpleCart_seed", "1");
        this.characteristics.put("SimpleCart_numFoldsPruning", "5");
        this.characteristics.put("SimpleCart_useOneSE", "false");
        this.characteristics.put("SimpleCart_sizePer", "1");
        this.characteristics.put("JRip", "true");
        this.characteristics.put("JRip_checkErrorRate", "true");
        this.characteristics.put("JRip_minNo", "2");
        this.characteristics.put("JRip_seed", "1");
        this.characteristics.put("JRip_usePruning", "true");
        this.characteristics.put("JRip_optimizations", "2");
        this.characteristics.put("JRip_folds", "3");
        this.characteristics.put("PART", "true");
        this.characteristics.put("PART_doNotMakeSplitPointActualValue", "false");
        this.characteristics.put("PART_minNumObj", "2");
        this.characteristics.put("PART_binarySplits", "false");
        this.characteristics.put("PART_useMDLcorrection", "true");
        this.characteristics.put("PART_pruning", "PART_confidenceFactor");
        this.characteristics.put("PART_confidenceFactorValue", "0.25");
        this.characteristics.put("PART_seed", null);
        this.characteristics.put("PART_numFolds", null);
        this.characteristics.put("DecisionTable", "true");
        this.characteristics.put("DecisionTable_useIBk", "false");
        this.characteristics.put("DecisionTable_crossVal", "1");
        this.characteristics.put("DecisionTable_evaluationMeasure", "auc");
        this.characteristics.put("DecisionTable_search", "BestFirst");
        this.characteristics.put("BestFirst_direction", "1");
        this.characteristics.put("BestFirst_searchTermination", "5");
        this.characteristics.put("GreedyStepwise_conservativeForwardSelection", null);
        this.characteristics.put("GreedyStepwise_searchBackwards", null);
    }

    public static void main(String[] args) throws Exception {
        Instances train_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff")));
        train_data.setClassIndex(train_data.numAttributes() - 1);

        BaselineIndividual bi = new BaselineIndividual(train_data);
    }

}
