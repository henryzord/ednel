package ednel.utils.analysis.optimizers;

import ednel.Main;
import ednel.classifiers.trees.SimpleCart;
import ednel.eda.EDNEL;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.File;
import java.lang.reflect.Constructor;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

public class Boosters {

    public enum SupportedAlgorithms {
        J48, SimpleCart, JRip, PART, DecisionTable
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

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
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

        System.err.println("TODO implement code to run for other values of n_external_folds! (currently only supports = 10)");

        String clf_name = options.get("classifier");

        NestedCrossValidation.SupportedAlgorithms algorithm_name;

        switch(clf_name) {
            case "J48":
                algorithm_name = NestedCrossValidation.SupportedAlgorithms.J48;
                break;
            case "SimpleCart":
                algorithm_name = NestedCrossValidation.SupportedAlgorithms.SimpleCart;
                break;
            case "JRip":
                algorithm_name = NestedCrossValidation.SupportedAlgorithms.JRip;
                break;
            case "PART":
                algorithm_name = NestedCrossValidation.SupportedAlgorithms.PART;
                break;
            case "DecisionTable":
                algorithm_name = NestedCrossValidation.SupportedAlgorithms.DecisionTable;
                break;
            default:
                throw new Exception(String.format("Booster for classifier %s is not implemented yet.", clf_name));
        }

        Object[] answers = IntStream.range(1, n_external_folds + 1).parallel().mapToObj(
                i -> Boosters.runExternalCrossValidationFoldBareBones(
                        algorithm_name, i, dataset_name, datasets_path,
                        experiment_metadata_path + File.separator + dataset_name)
        ).toArray();
    }

    private static Object runExternalCrossValidationFoldBareBones(
            NestedCrossValidation.SupportedAlgorithms algorithmName, int n_external_fold,
            String dataset_name, String datasets_path, String dataset_experiment_path) {
        try {

            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances external_train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances external_test_data = datasets.get("test_data");  // 1/10 of external cv folds

            String clfFullName;

            switch(algorithmName) {
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

            AdaBoostM1 adb = new AdaBoostM1();
            adb.setOptions(new String[]{"-W", clfFullName});
            adb.buildClassifier(external_train_data);

            HashMap<String, Object> bestCombination = new HashMap<>();
            bestCombination.put("classifier", clfFullName);

            NestedCrossValidation.wekaClassifierParametersToFile(
                    n_external_fold, dataset_experiment_path, dataset_name, bestCombination
            );

            PBILLogger.write_predictions_to_file(
                    new AbstractClassifier[]{adb},
                    external_test_data,
                    dataset_experiment_path + File.separator +
                            String.format(
                                    "overall%stest_sample-01_fold-%02d_%s.preds",
                                    File.separator,
                                    n_external_fold,
                                    adb.getClass().getSimpleName()
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
}
