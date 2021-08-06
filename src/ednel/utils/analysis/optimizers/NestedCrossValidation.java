package ednel.utils.analysis.optimizers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.Main;
import ednel.classifiers.trees.SimpleCart;
import ednel.eda.EDNEL;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Parameter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

public class NestedCrossValidation {
    public enum SupportedAlgorithms {
        EDNEL, RandomForest, J48, SimpleCart, JRip, PART, DecisionTable
    }

    public static CommandLine parseCommandLine(String[] args) throws ParseException {
        Options options = new Options();

        options.addOption(Option.builder()
                .required(true)
                .longOpt("datasets_path")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Must lead to a path that contains several subpaths, one for each dataset. " +
                        "Each subpath, in turn, must have the arff files.")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("dataset_name")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of the datasets to run experiments. Must be a list separated by a comma\n" +
                        "\tExample: iris,mushroom,adult")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("metadata_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where runs results will be stored.")
                .build());

        // TODO allow for any number of external folds to run!
//        options.addOption(Option.builder()
//                .longOpt("n_external_folds")
//                .required(true)
//                .type(String.class)
//                .hasArg()
//                .numberOfArgs(1)
//                .desc("Number of external folds to use in nested cross validation.")
//                .build());

        options.addOption(Option.builder()
                .longOpt("n_internal_folds")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of folds to use in nested cross validation.")
                .build());

        options.addOption(Option.builder()
                .longOpt("classifier")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of classifier to optimize with nested cross validation")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_jobs")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of threads to use for computing. Use a positive non-zero value. Defaults to 10.")
                .build());

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    public static void wekaClassifierParametersToFile(
            int n_external_fold, String dataset_experiment_path, String dataset_name,
            HashMap<String, Object> bestCombination
    ) throws IOException {

        HashMap<String, String> obj = new HashMap<>();

        obj.put("dataset_name", dataset_name);
        obj.put("dataset_experiment_path", dataset_experiment_path);
        for(String key : bestCombination.keySet()) {
            obj.put(key, bestCombination.get(key).toString());
        }

        FileWriter fw = new FileWriter(
                dataset_experiment_path + File.separator + String.format("test_sample-01_fold-%02d_parameters.json", n_external_fold)
        );

        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        fw.write(converter.toJson(obj));
        fw.flush();
        fw.close();
    }

    /**
     * Given a mathematical expression (as a String), which selects the number of features to be used as hyper-parameter
     * for RandomForest, and a dataset, returns the exact number of features that should be used.<br>
     *<br>
     * Example:<br>
     * Input: "-K {Math.sqrt(p)/2}" (and a dataset with 100 attributes)<br>
     * Output: "-K 5"<br>
     *
     * @param param The mtry hyper-parameter of RandomForest, expressed as "-K {mathematical_expression}" (see example above)
     * @param data Dataset
     * @return The mtry hyper-parameter of RandomForest, now with a number
     * @throws ScriptException If the mathematical expression cannot be evaluated
     */
    private static String getRandomForestNumFeatures(String param, Instances data) throws ScriptException {
        ScriptEngineManager mgr = new ScriptEngineManager();
        ScriptEngine engine = mgr.getEngineByName("JavaScript");
        String segment = param.substring(param.indexOf("{") + 1, param.indexOf("}"));
        segment = segment.replace("p", String.valueOf(data.numAttributes()));
        String new_param = param.substring(0, param.indexOf("{")) + Math.round(Double.valueOf(engine.eval(segment).toString()));
        return new_param;
    }

    private static String[] atomize(String[] params) {
        String[] new_string = new String[params.length * 2];
        int counter = 0;
        for(int i = 0; i < params.length; i++) {
            if(params[i].length() > 0) {
                String[] splitted = params[i].split(" ");
                new_string[counter] = splitted[0];
                try {
                    new_string[counter + 1] = splitted[1];
                } catch(ArrayIndexOutOfBoundsException aei) {
                    new_string[counter + 1] = "";
                }
            } else {
                new_string[counter] = "";
                new_string[counter + 1] = "";
            }
            counter += 2;
        }
        return new_string;
    }

    private static AbstractClassifier getInternalCrossValidationAbstractClassifier(
            SupportedAlgorithms algorithmName, Constructor constructor,
            HashMap<String, Object> comb, Instances data
    ) throws Exception {

        Parameter[] constructorParams = constructor.getParameters();
        AbstractClassifier abs = null;
        Object[] toUseParams;
        switch(algorithmName) {
            case EDNEL:
                toUseParams = new Object[constructorParams.length];

                for(int c_param = 0; c_param < constructorParams.length; c_param++) {
                    toUseParams[c_param] = comb.get(constructorParams[c_param].getName());
                }

                abs = (AbstractClassifier)constructor.newInstance(toUseParams);
                abs.buildClassifier(data);
                break;
            case RandomForest:
                toUseParams = new String[comb.size() * 2];
                int counter_param_rf = 0;
                for(String key : comb.keySet()) {
                    String paramName;
                    String paramValue;
                    String[] splitted = comb.get(key).toString().split(" ");
                    if(key.equals("numFeatures")) {
                        paramName = splitted[0];
                        paramValue = NestedCrossValidation.getRandomForestNumFeatures(
                                comb.get("numFeatures").toString(),
                                data
                        ).split(" ")[1];
                    } else if(splitted.length > 1) {
                        paramName = splitted[0];
                        paramValue = splitted[1];
                    } else {
                        paramName = "";
                        paramValue = "";
                    }
                    toUseParams[counter_param_rf] = paramName;
                    toUseParams[counter_param_rf + 1] = paramValue;
                    counter_param_rf += 2;
                }
                abs = new RandomForest();
                abs.setOptions((String[])toUseParams);
                abs.buildClassifier(data);
                break;
            case J48:
            case SimpleCart:
            case JRip:
            case PART:
            case DecisionTable:
                toUseParams = new String[comb.size() * 2];
                int counter_param_others = 0;
                for(String key : comb.keySet()) {
                    String paramName;
                    String paramValue;
                    String[] splitted = comb.get(key).toString().split(" ", 2);
                    if(splitted.length > 2) {
                        throw new Exception("unexpected behavior!");
                    }
                    if(splitted.length > 1) {
                        paramName = splitted[0];
                        paramValue = splitted[1];
                    } else {
                        paramName = "";
                        paramValue = splitted[0];
                    }
                    toUseParams[counter_param_others] = paramName;
                    toUseParams[counter_param_others + 1] = paramValue;
                    counter_param_others += 2;
                }
                abs = (AbstractClassifier)constructor.newInstance();
                abs.setOptions((String[])toUseParams);
                abs.buildClassifier(data);
                break;
            default:
                throw new NotImplementedException();
        }
        return abs;
    }

    private static Object runExternalCrossValidationFoldBareBones(
            SupportedAlgorithms algorithmName, ArrayList<HashMap<String, Object>> combinations, int n_external_fold, int n_internal_folds,
            String dataset_name, String datasets_path, String dataset_experiment_path
    ) {
        try {
            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances external_train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances external_test_data = datasets.get("test_data");  // 1/10 of external cv folds

            // prepares data for future internal cross validation
            external_train_data = FitnessCalculator.betterStratifier(external_train_data, n_internal_folds);

            ArrayList<NCVMatrixHandler> combinationsHandlers = new ArrayList<>();

            Constructor constructor;

            switch(algorithmName) {
                case EDNEL:
                    constructor = EDNEL.class.getConstructor(
                            float.class, float.class, int.class, int.class, int.class, int.class, int.class, int.class,
                            boolean.class, int.class, int.class, int.class, int.class, PBILLogger.class, Integer.class
                    );
                    break;
                case RandomForest:
                    constructor = RandomForest.class.getConstructor();
                    break;
                case J48:
                    constructor = J48.class.getConstructor();
                    break;
                case SimpleCart:
                    constructor = SimpleCart.class.getConstructor();
                    break;
                case JRip:
                    constructor = JRip.class.getConstructor();
                    break;
                case PART:
                    constructor = PART.class.getConstructor();
                    break;
                case DecisionTable:
                    constructor = DecisionTable.class.getConstructor();
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + algorithmName);
            }

            // iterates over combinations of hyper-parameters
            for(HashMap<String, Object> comb : combinations) {
                NCVMatrixHandler combinationMatrixHandler = new NCVMatrixHandler(external_train_data, algorithmName);
                for(int i = 0; i < n_internal_folds; i++) {
                    Instances internal_train_data = external_train_data.trainCV(n_internal_folds, i);
                    Instances internal_test_data = external_train_data.testCV(n_internal_folds, i);

                    AbstractClassifier abstractClassifier = NestedCrossValidation.getInternalCrossValidationAbstractClassifier(
                            algorithmName, constructor, comb, internal_train_data
                    );
                    combinationMatrixHandler.handle(algorithmName, abstractClassifier, internal_test_data);
                }  // ends for internal cross validation

                combinationMatrixHandler.compile();
                combinationsHandlers.add(combinationMatrixHandler);
            }  // ends for combinations

            int best_combination_index = -1;
            double best_combination_auc = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < combinationsHandlers.size(); i++) {
                if(combinationsHandlers.get(i).getAuc() > best_combination_auc) {
                    best_combination_index = i;
                    best_combination_auc = combinationsHandlers.get(i).getAuc();
                }
            }

            HashMap<String, Object> bestCombination = combinations.get(best_combination_index);
            NCVMatrixHandler bestCombinationHandler = combinationsHandlers.get(best_combination_index);

            switch(algorithmName) {
                case EDNEL:
                    bestCombination.put(
                            "pbilLogger",
                            new PBILLogger(
                                    dataset_name, dataset_experiment_path,
                                    (Integer)bestCombination.get("n_individuals"),
                                    (Integer)bestCombination.get("n_generations"),
                                    1, n_external_fold, true, false
                            )
                    );
                    EDNEL ednel = (EDNEL)NestedCrossValidation.getInternalCrossValidationAbstractClassifier(
                            algorithmName, constructor, bestCombination, external_train_data
                    );
                    ednel.writeParametersToFile(
                            dataset_experiment_path + File.separator + String.format("test_sample-01_fold-%02d_parameters.json", n_external_fold),
                            dataset_experiment_path, dataset_name, bestCombinationHandler.getBestUsersOverall()
                    );
                    HashMap<String, Individual> toReport = new HashMap<>(1);
                    if(bestCombinationHandler.getBestUsersOverall()) {
                        toReport.put("overall", ednel.getOverallBest());
                    } else {
                        toReport.put("last", ednel.getCurrentGenBest());
                    }
                    ednel.getPbilLogger().toFile(
                            ednel.getDependencyNetwork(), toReport, external_test_data
                    );
                    return true;

                case J48:
                case SimpleCart:
                case JRip:
                case PART:
                case DecisionTable:
                case RandomForest:
                    NestedCrossValidation.wekaClassifierParametersToFile(
                            n_external_fold, dataset_experiment_path, dataset_name, bestCombination
                    );

                    AbstractClassifier abstractClf = NestedCrossValidation.getInternalCrossValidationAbstractClassifier(
                            algorithmName, constructor, bestCombination, external_train_data
                    );

                    PBILLogger.write_predictions_to_file(
                            new AbstractClassifier[]{abstractClf},
                            external_test_data,
                            dataset_experiment_path + File.separator +
                                    String.format(
                                            "overall%stest_sample-01_fold-%02d_%s.preds",
                                            File.separator,
                                            n_external_fold,
                                            abstractClf.getClass().getSimpleName()
                                    )
                    );
                    System.out.printf("Done: %s,%d,%d%n", dataset_name, 1, n_external_fold);
                    return true;
                default:
                    throw new NotImplementedException();
            }
        } catch(Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace(System.err);
            return e;
        }
    }

    /**
     * Performs a nested-cross validation optimization, depending on the algorithm.
     *
     * @param algorithmName Name of classifier
     * @param n_external_folds Number of external folds.
     * @param n_internal_folds Number of internal folds.
     * @param dataset_name Name of datasets to test
     * @param datasets_path Path where datasets are stored, one folder for each dataset, and inside each dataset folder,
     *                      two csv files for each fold (one for the training data, and another for the testing data)
     * @param experiment_metadata_path Path to where write metadata on this experiment
     */
    private static void classifierOptimization(
            SupportedAlgorithms algorithmName, int n_jobs, int n_external_folds, int n_internal_folds,
            String dataset_name, String datasets_path, String experiment_metadata_path
    ) {

        ArrayList<HashMap<String, Object>> combinations;
        switch(algorithmName) {
            case EDNEL:
                combinations = NestedCrossValidation.getEDNELCombinations();
                break;
            case RandomForest:
                combinations = NestedCrossValidation.getRandomForestCombinations();
                break;
            case J48:
                combinations = NestedCrossValidation.getJ48Combinations();
                break;
            case SimpleCart:
                combinations = NestedCrossValidation.getSimpleCartCombinations();
                break;
            case JRip:
                combinations = NestedCrossValidation.getJRipCombinations();
                break;
            case PART:
                combinations = NestedCrossValidation.getPARTCombinations();
                break;
            case DecisionTable:
                combinations = NestedCrossValidation.getDecisionTableCombinations();
                break;
            default:
                throw new NotImplementedException();
        }

        n_jobs = Math.min(n_jobs, 10);

        int n_loops = (int)(n_external_folds / n_jobs);

        int counter = 0;
        Object[] answers;
        for(int i = 0; i < n_loops; i++) {
            answers = IntStream.range((i * n_jobs) + 1, ((i + 1) * n_jobs) + 1).parallel().mapToObj(
                    j -> NestedCrossValidation.runExternalCrossValidationFoldBareBones(
                            algorithmName, combinations,
                            j, n_internal_folds, dataset_name, datasets_path,
                            experiment_metadata_path + File.separator + dataset_name)
            ).toArray();
            counter += n_jobs;
        }

        if(counter < n_external_folds) {
            answers = IntStream.range(counter, n_external_folds + 1).parallel().mapToObj(
                    j -> NestedCrossValidation.runExternalCrossValidationFoldBareBones(
                            algorithmName, combinations,
                            j, n_internal_folds, dataset_name, datasets_path,
                            experiment_metadata_path + File.separator + dataset_name)
            ).toArray();
        }
    }

    private static ArrayList<HashMap<String, Object>> getDecisionTableCombinations() {
        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        String[] useIBk_array = {"-I", ""};
        String[] evaluationMeasure_array = {"-E acc", "-E rmse", "-E mae", "-E auc"};
        String crossVal = "-X 1";
        String[] search_array = {"-S weka.attributeSelection.BestFirst -D 1 -N 5", "-S weka.attributeSelection.GreedyStepwise"};

        for(String useIBk : useIBk_array) {
            for(String evaluationMeasure : evaluationMeasure_array) {
                for(String search : search_array) {
                    HashMap<String, Object> comb = new HashMap<>();

                    comb.put("useIBk", useIBk);
                    comb.put("evaluationMeasure", evaluationMeasure);
                    comb.put("crossVal", crossVal);
                    comb.put("search", search);

                    combinations.add(comb);
                }
            }
        }
        return combinations;
    }

    private static ArrayList<HashMap<String, Object>> getPARTCombinations() {
        String useMDLcorrection = "";  // always true
        String binarySplits = ""; // always false
        String[] minNumObj_array = {"-M 2", "-M 4", "-M 6", "-M 8"};
        String doNotMakeSplitPointActualValue = "";  // always false

        String[] pruning_array = {"unpruned", "reducedErrorPruning", "confidenceFactor"};

        String confidenceFactor = "-C 0.25";

        String numFolds = "-N 3";
        String seed = "-Q 1";

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();
        for(String minNumObj : minNumObj_array) {
            for(String pruning : pruning_array) {
                HashMap<String, Object> comb = new HashMap<>();

                switch(pruning) {
                    case "reducedErrorPruning":
                        comb.put("pruning", "-R");
                        comb.put("seed", seed);

                        comb.put("numFolds", numFolds);
                        break;
                    case "confidenceFactor":
                        comb.put("pruning", confidenceFactor);
                        break;
                    case "unpruned":
                        comb.put("pruning", "-U");
                        break;
                }

                comb.put("binarySplits", binarySplits);
                comb.put("doNotMakeSplitPointActualValue", doNotMakeSplitPointActualValue);
                comb.put("minNumObj", minNumObj);
                comb.put("useMDLcorrection", useMDLcorrection);

                combinations.add(comb);
            }
        }
        return combinations;
    }

    private static ArrayList<HashMap<String, Object>> getJRipCombinations() {
        boolean[] usePruning_array = {true, false};
        String checkErrorRate = "";  // always true
        String[] minNo_array = {"-N 2", "-N 4", "-N 6", "-N 8"};
        String seed = "-S 1";
        String optimizations = "-O 2";
        String folds = "-F 3";


        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();
        for(boolean usePruning : usePruning_array) {
            for(String minNo : minNo_array) {
                HashMap<String, Object> comb = new HashMap<>();

                if(usePruning) {
                    comb.put("usePruning", "");

                    comb.put("optimizations", optimizations);
                    comb.put("folds", folds);
                } else {
                    comb.put("usePruning", "-P");
                }

                comb.put("seed", seed);
                comb.put("minNo", minNo);
                comb.put("checkErrorRate", checkErrorRate);

                combinations.add(comb);
            }
        }
        return combinations;

    }

    private static ArrayList<HashMap<String, Object>> getSimpleCartCombinations() {

        String[] minNumObj_array = {"-M 2", "-M 5", "-M 8"};
        boolean[] usePrune_array = {true, false};
        String[] heuristic_array = {"", "-H"};

        String numFoldsPruning = "-N 5";
        String useOneSE = "";  // always false
        String sizePer = "-C 1.0";
        String seed = "-S 1";

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();
        for(String minNumObj : minNumObj_array) {
            for(boolean usePrune : usePrune_array) {
                for(String heuristic : heuristic_array) {
                    HashMap<String, Object> comb = new HashMap<>();

                    if(usePrune) {
                        comb.put("usePrune", "");
                        comb.put("sizePer", sizePer);
                        comb.put("seed", seed);

                        comb.put("useOneSE", useOneSE);

                        comb.put("numFoldsPruning", numFoldsPruning);
                    } else {
                        comb.put("usePrune", "-U");
                    }
                    comb.put("heuristic", heuristic);
                    comb.put("minNumObj", minNumObj);

                    combinations.add(comb);
                }
            }
        }
        return combinations;
    }

    private static ArrayList<HashMap<String, Object>> getJ48Combinations() {
        String binarySplits = ""; // always false
        String collapseTree = "";  // always true
        String confidenceFactor = "-C 0.25";
        String doNotMakeSplitPointActualValue = "";  // always false
        String[] minNumObj_array = {"-M 2", "-M 4", "-M 6", "-M 8"};
        String numFolds = "-N 3";
        String[] pruning_array = {"unpruned", "reducedErrorPruning", "confidenceFactor"};
        String subtreeRaising = "";  // always true
        String useLaplace = "";  // always false
        String useMDLcorrection = "";  // always true
        String seed = "-Q 1";

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();
        for(String minNumObj : minNumObj_array) {
            for(String pruning : pruning_array) {
                HashMap<String, Object> comb = new HashMap<>();

                switch(pruning) {
                    case "reducedErrorPruning":
                        comb.put("pruning", "-R");
                        comb.put("seed", seed);

                        comb.put("numFolds", numFolds);
                        comb.put("subtreeRaising", subtreeRaising);

                        break;
                    case "confidenceFactor":
                        comb.put("pruning", confidenceFactor);
                        comb.put("subtreeRaising", subtreeRaising);
                        break;
                    case "unpruned":
                        comb.put("pruning", "-U");
                        break;
                }

                comb.put("collapseTree", collapseTree);
                comb.put("binarySplits", binarySplits);
                comb.put("doNotMakeSplitPointActualValue", doNotMakeSplitPointActualValue);
                comb.put("minNumObj", minNumObj);
                comb.put("useLaplace", useLaplace);
                comb.put("useMDLcorrection", useMDLcorrection);

                combinations.add(comb);
            }
        }
        return combinations;
    }

    private static ArrayList<HashMap<String, Object>> getEDNELCombinations() {

        float[] learning_rate_values = {0.13f, 0.26f, 0.52f};
        float selection_share = 0.5f;
        int n_individuals = 50;
        int n_generations = 100;

        int timeout = 3600;
        int timeout_individual = 60;

        int burn_in = 100;
        int thinning_factor = 0;

        boolean no_cycles = false;

        int[] early_stop_generations_values = {10, 20};
        int[] max_parents_values = {0, 1};

        int delay_structure_learning = 5;

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        for(float learning_rate : learning_rate_values) {
            for(int early_stop_generations : early_stop_generations_values) {
                for(int max_parents : max_parents_values) {
                    HashMap<String, Object> comb = new HashMap<>();

                    comb.put("learning_rate", learning_rate);
                    comb.put("selection_share", selection_share);
                    comb.put("n_individuals", n_individuals);
                    comb.put("n_generations", n_generations);
                    comb.put("timeout", timeout);
                    comb.put("timeout_individual", timeout_individual);
                    comb.put("burn_in", burn_in);
                    comb.put("thinning_factor", thinning_factor);
                    comb.put("no_cycles", no_cycles);
                    comb.put("early_stop_generations", early_stop_generations);
                    comb.put("max_parents", max_parents);
                    comb.put("delay_structure_learning", delay_structure_learning);
                    comb.put("n_internal_folds", 0); // n_internal_folds for EDNEL is not the same for NestedCrossValidation
                    comb.put("pbilLogger", null);
                    comb.put("seed", null);

                    combinations.add(comb);
                }
            }
        }

        return combinations;
    }

    private static ArrayList<HashMap<String, Object>> getRandomForestCombinations() {
        String breakTiesRandomly = "";  // always false
        String maxDepth = "-depth 0";  // always unlimited
        String numIterations = "-I 1000";  // always 1000

        String[] bagSizePercent_array = {"-P 90", "-P 100"};
        String[] numFeatures_array = {
                "-K {Math.sqrt(p)/2}",
                "-K {Math.sqrt(p)}",
                "-K {Math.sqrt(p)*2}",
                "-K {Math.log(p)/(Math.log(2)*2)}",
                "-K {Math.log(p)/Math.log(2)}",
                "-K {(Math.log(p)/Math.log(2))*2}"
        };  // must be replaced by the actual number of features

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        for(String numFeatures : numFeatures_array) {
            for(String bagSizePercent : bagSizePercent_array) {
                HashMap<String, Object> comb = new HashMap<>();

                comb.put("breakTiesRandomly", breakTiesRandomly);
                comb.put("maxDepth", maxDepth);
                comb.put("numIterations", numIterations);
                comb.put("bagSizePercent", bagSizePercent);
                comb.put("numFeatures", numFeatures);
                combinations.add(comb);
            }
        }
        return combinations;
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = NestedCrossValidation.parseCommandLine(args);

        HashMap<String, String> options = new HashMap<>();
        for(Option opt : commandLine.getOptions()) {
            options.put(opt.getLongOpt(), commandLine.getOptionValue(opt.getLongOpt()));
        }

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        String str_time = dtf.format(LocalDateTime.now());
        options.put("datasets_names", options.get("dataset_name"));
        PBILLogger.metadata_path_start(str_time, options);
        options.remove("datasets_names");

        String experiment_metadata_path = options.get("metadata_path") + File.separator + str_time;

        String datasets_path = options.get("datasets_path");
        String dataset_name = options.get("dataset_name");
//        int n_external_folds = Integer.parseInt(options.get("n_external_folds"));  // TODO allow for any number of external folds to run!!!
        int n_external_folds = 10;  // TODO allow for any number of external folds to run!!!

        int n_jobs = 10;
        if(options.get("n_jobs") != null) {
            n_jobs = Integer.parseInt(options.get("n_jobs"));
            if(n_jobs <= 0) {
                throw new Exception("Number of jobs must be positive and non-negative!");
            }
        }

        System.err.println("TODO implement code to run for other values of n_external_folds! (currently only supports = 10)");

        int n_internal_folds = Integer.parseInt(options.get("n_internal_folds"));
        String clf_name = options.get("classifier");

        SupportedAlgorithms algorithm_name;

        switch(clf_name) {
            case "EDNEL":
                algorithm_name = SupportedAlgorithms.EDNEL;
                break;
            case "RandomForest":
                algorithm_name = SupportedAlgorithms.RandomForest;
                break;
            case "J48":
                algorithm_name = SupportedAlgorithms.J48;
                break;
            case "SimpleCart":
                algorithm_name = SupportedAlgorithms.SimpleCart;
                break;
            case "JRip":
                algorithm_name = SupportedAlgorithms.JRip;
                break;
            case "PART":
                algorithm_name = SupportedAlgorithms.PART;
                break;
            case "DecisionTable":
                algorithm_name = SupportedAlgorithms.DecisionTable;
                break;
            default:
                throw new Exception(String.format("Nested cross-validation for classifier %s is not implemented yet.", clf_name));
        }

        NestedCrossValidation.classifierOptimization(
                algorithm_name, n_jobs, n_external_folds, n_internal_folds, dataset_name, datasets_path, experiment_metadata_path
        );
    }
}
