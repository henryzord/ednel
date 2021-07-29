package ednel.utils.analysis.optimizers;

import ednel.Main;
import ednel.classifiers.trees.SimpleCart;
import ednel.eda.individual.FitnessCalculator;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

public class Boosters {

    public enum SupportedAlgorithms {
        DecisionStump, J48, SimpleCart, JRip, PART, DecisionTable
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
                .longOpt("classifier")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of classifier to optimize with nested cross validation")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_internal_folds")
                .required(true)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of internal folds to use")
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

    private static AbstractClassifier getInternalCrossValidationAbstractClassifier(
            HashMap<String, Object> comb, Instances data
    ) throws Exception {

        Object[] toUseParams = new String[comb.size() * 2];
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
        AdaBoostM1 abs = new AdaBoostM1();
        abs.setOptions((String[])toUseParams);
        abs.buildClassifier(data);
        return abs;
    }

    private static Object runExternalCrossValidationFoldBareBones(
            Boosters.SupportedAlgorithms algorithmName, int n_external_fold, int n_internal_folds,
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

            String clfFullName;

            switch(algorithmName) {
                case DecisionStump:
                    clfFullName = DecisionStump.class.getCanonicalName();
                    break;
                case J48:
                    clfFullName = J48.class.getCanonicalName();
                    break;
                case SimpleCart:
                    clfFullName = SimpleCart.class.getCanonicalName();
                    break;
                case JRip:
                    clfFullName = JRip.class.getCanonicalName();
                    break;
                case PART:
                    clfFullName = PART.class.getCanonicalName();
                    break;
                case DecisionTable:
                    clfFullName = DecisionTable.class.getCanonicalName();
                    break;
                default:
                    throw new IllegalStateException("Unexpected value: " + algorithmName);
            }

            ArrayList<HashMap<String, Object>> combinations = Boosters.getAdaboostCombinations(clfFullName);

            // NCVMatrixHandler doesn't really care of the subtype of weka classifier; this type verification is only
            // used to decide if NCVMatrixHandler should allocated two or one matrix for storing predictions.
            // NestedCrossValidation.SupportedAlgorithms.J48 will do just fine for any of the Boosters.SupportedAlgorithms
            // classifiers
            NestedCrossValidation.SupportedAlgorithms converted = NestedCrossValidation.SupportedAlgorithms.J48;

            // iterates over combinations of hyper-parameters
            for(HashMap<String, Object> comb : combinations) {
                NCVMatrixHandler combinationMatrixHandler = new NCVMatrixHandler(external_train_data, converted);
                for(int i = 0; i < n_internal_folds; i++) {
                    Instances internal_train_data = external_train_data.trainCV(n_internal_folds, i);
                    Instances internal_test_data = external_train_data.testCV(n_internal_folds, i);

                    AbstractClassifier abstractClassifier = Boosters.getInternalCrossValidationAbstractClassifier(
                            comb, internal_train_data
                    );
                    combinationMatrixHandler.handle(converted, abstractClassifier, internal_test_data);
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

            NestedCrossValidation.wekaClassifierParametersToFile(
                    n_external_fold, dataset_experiment_path, dataset_name, bestCombination
            );

            AbstractClassifier abstractClf = Boosters.getInternalCrossValidationAbstractClassifier(
                    bestCombination, external_train_data
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
        } catch(Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace(System.err);
            return e;
        }
    }

    private static ArrayList<HashMap<String, Object>> getAdaboostCombinations(String clfCanonicalName) {

        int[] numIterations_array = new int[]{50, 91, 132, 173, 214, 255, 295, 336, 377, 418, 459, 500};

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        for(int numIterations : numIterations_array) {
            HashMap<String, Object> comb = new HashMap<>();

            comb.put("numIterations", String.format("-I %d", numIterations));
            comb.put("classifier", String.format("-W %s", clfCanonicalName));
            combinations.add(comb);
        }

        return combinations;
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = Boosters.parseCommandLine(args);

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
        int n_internal_folds = Integer.parseInt(options.get("n_internal_folds"));

        int n_jobs = 10;
        if(options.get("n_jobs") != null) {
            n_jobs = Integer.parseInt(options.get("n_jobs"));
            if(n_jobs <= 0) {
                throw new Exception("Number of jobs must be positive and non-negative!");
            }
        }

        System.err.println("TODO implement code to run for other values of n_external_folds! (currently only supports = 10)");

        String clf_name = options.get("classifier");

        Boosters.SupportedAlgorithms algorithm_name;

        switch(clf_name) {
            case "DecisionStump":
                algorithm_name = Boosters.SupportedAlgorithms.DecisionStump;
                break;
            case "J48":
                algorithm_name = Boosters.SupportedAlgorithms.J48;
                break;
            case "SimpleCart":
                algorithm_name = Boosters.SupportedAlgorithms.SimpleCart;
                break;
            case "JRip":
                algorithm_name = Boosters.SupportedAlgorithms.JRip;
                break;
            case "PART":
                algorithm_name = Boosters.SupportedAlgorithms.PART;
                break;
            case "DecisionTable":
                algorithm_name = Boosters.SupportedAlgorithms.DecisionTable;
                break;
            default:
                throw new Exception(String.format("Booster for classifier %s is not implemented yet.", clf_name));
        }

        n_jobs = Math.min(n_jobs, 10);

        int n_loops = (int)(n_external_folds / n_jobs);

        int counter = 0;
        Object[] answers;
        for(int i = 0; i < n_loops; i++) {
            answers = IntStream.range((i * n_jobs) + 1, ((i + 1) * n_jobs) + 1).parallel().mapToObj(
                    j -> Boosters.runExternalCrossValidationFoldBareBones(
                            algorithm_name, j, n_internal_folds, dataset_name, datasets_path,
                            experiment_metadata_path + File.separator + dataset_name)
            ).toArray();
            counter += n_jobs;
        }

        answers = IntStream.range(counter, n_external_folds + 1).parallel().mapToObj(
                i -> Boosters.runExternalCrossValidationFoldBareBones(
                        algorithm_name, i, n_internal_folds, dataset_name, datasets_path,
                        experiment_metadata_path + File.separator + dataset_name)
        ).toArray();
    }
}
