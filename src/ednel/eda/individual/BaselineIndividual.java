/**
 * An individual with default hyper-parameters from Weka for each one of its classifiers.
 */

package ednel.eda.individual;

import ednel.Main;
import ednel.network.DependencyNetwork;
import org.apache.commons.math3.random.MersenneTwister;
import weka.core.Instances;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.HashMap;

public class BaselineIndividual extends Individual {

    private static final HashMap<String, String> options;

    private static final HashMap<String, String> baselineCharacteristics;

    static {
        HashMap<String, String> localOptions = new HashMap<>();
        localOptions.put("J48", "-C 0.25 -M 2");
        localOptions.put("SimpleCart", "-M 2 -N 5 -C 1 -S 1");
        localOptions.put("PART", "-M 2 -C 0.25 -Q 1");
        localOptions.put("JRip", "-F 3 -N 2.0 -O 2 -S 1");
        localOptions.put("DecisionTable", "-R -X 1 -S weka.attributeSelection.BestFirst");
        localOptions.put("Aggregator", "MajorityVotingAggregator");
        localOptions.put("BestFirst", "-D 1 -N 5");

        options = localOptions;

        HashMap<String, String> localChars = new HashMap<>();
        localChars.put("Aggregator", "MajorityVotingAggregator");
        localChars.put("J48", "true");
        localChars.put("J48_binarySplits", "false");
        localChars.put("J48_useLaplace", "true");
        localChars.put("J48_minNumObj", "2");
        localChars.put("J48_useMDLcorrection", "false");
        localChars.put("J48_collapseTree", "true");
        localChars.put("J48_doNotMakeSplitPointActualValue", "false");
        localChars.put("J48_pruning", "confidenceFactor");
        localChars.put("J48_numFolds", null);
        localChars.put("J48_subtreeRaising", "true");
        localChars.put("J48_confidenceFactorValue", "0.25");
        localChars.put("SimpleCart", "true");
        localChars.put("SimpleCart_heuristic", "true");
        localChars.put("SimpleCart_minNumObj", "2");
        localChars.put("SimpleCart_usePrune", "true");
        localChars.put("SimpleCart_numFoldsPruning", "5");
        localChars.put("SimpleCart_useOneSE", "false");
        localChars.put("JRip", "true");
        localChars.put("JRip_checkErrorRate", "true");
        localChars.put("JRip_minNo", "2");
        localChars.put("JRip_usePruning", "true");
        localChars.put("JRip_optimizations", "2");
        localChars.put("JRip_folds", "3");
        localChars.put("PART", "true");
        localChars.put("PART_doNotMakeSplitPointActualValue", "false");
        localChars.put("PART_minNumObj", "2");
        localChars.put("PART_binarySplits", "false");
        localChars.put("PART_useMDLcorrection", "true");
        localChars.put("PART_pruning", "confidenceFactor");
        localChars.put("PART_confidenceFactorValue", "0.25");
        localChars.put("PART_numFolds", null);
        localChars.put("DecisionTable", "true");
        localChars.put("DecisionTable_useIBk", "false");
        localChars.put("DecisionTable_crossVal", "1");
        localChars.put("DecisionTable_evaluationMeasure", "auc");
        localChars.put("DecisionTable_search", "weka.attributeSelection.BestFirst");
        localChars.put("BestFirst_direction", "1");
        localChars.put("BestFirst_searchTermination", "5");
        localChars.put("GreedyStepwise_conservativeForwardSelection", null);
        localChars.put("GreedyStepwise_searchBackwards", null);
        baselineCharacteristics = localChars;
    }

    public BaselineIndividual() throws Exception {
        super(BaselineIndividual.options, BaselineIndividual.baselineCharacteristics);
    }

    public static void main(String[] args) throws Exception {

        LocalDateTime start = LocalDateTime.now();

        HashMap<String, Instances> sets = Main.loadDataset("D:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv", "artificialcharacters", 1);
        Instances train_data = sets.get("train_data");
        Instances test_data = sets.get("test_data");

        final int n_individuals = 100;

        DependencyNetwork dn = new DependencyNetwork(
                new MersenneTwister(), 100, 0,
                false, 0.5, 2, 1,
                60
        );

        BaselineIndividual bi = new BaselineIndividual();
        HashMap<String, String> lastStart = bi.getCharacteristics();

        FitnessCalculator fc = new FitnessCalculator(5, train_data);

        Individual[] res = dn.gibbsSample(lastStart, n_individuals, fc, 1);

        Fitness fitness = fc.evaluateEnsemble(1, bi);

        LocalDateTime end = LocalDateTime.now();

        System.out.println("Fitness: " + fitness);
        System.out.println("Elapsed time: " + (int)start.until(end, ChronoUnit.SECONDS) + " seconds");
    }
}
