package ednel.utils.analysis.optimizers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.Main;
import ednel.eda.aggregators.RuleExtractorAggregator;
import ednel.eda.individual.FitnessCalculator;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import org.apache.commons.math3.analysis.function.Abs;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

public class ExtractRulesFromRandomForest {
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

        options.addOption(Option.builder()
                .longOpt("string_options")
                .required(false)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("A string of all the options for classifiers in the ensemble.")
                .build());

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = ExtractRulesFromRandomForest.parseCommandLine(args);

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

        String dataset_experiment_path = options.get("metadata_path") + File.separator + str_time + File.separator + options.get("dataset_name");

        String string_options = commandLine.getOptionValue("string_options");
        if(string_options == null) {
            string_options = "";
        }

        String datasets_path = options.get("datasets_path");
        String dataset_name = options.get("dataset_name");
//        int n_external_folds = Integer.parseInt(options.get("n_external_folds"));  // TODO allow for any number of external folds to run!!!
        int n_external_folds = 10;  // TODO allow for any number of external folds to run!!!

        System.err.println("TODO implement code to run for other values of n_external_folds! (currently only supports = 10)");

        for(int n_external_fold = 1; n_external_fold <= n_external_folds; n_external_fold++) {

            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances test_data = datasets.get("test_data");  // 1/10 of external cv folds

            RandomForestRulesClassifier rfrc = new RandomForestRulesClassifier();
            rfrc.setOptions(string_options.split(" "));
            rfrc.buildClassifier(train_data);

            PBILLogger.write_predictions_to_file(
                    new AbstractClassifier[]{rfrc},
                    test_data,
                    dataset_experiment_path + File.separator +
                            String.format(
                                    "overall%stest_sample-01_fold-%02d_%s.preds",
                                    File.separator,
                                    n_external_fold,
                                    rfrc.getClass().getSimpleName()
                            )
            );
            // TODO annotate number of resulting rules!
            // TODO both input rules and output rules
//            ExtractRulesFromRandomForest.parametersToFile(n_external_fold, dataset_experiment_path, dataset_name);

            System.out.printf("Done: Dataset %s fold %02d\n", dataset_name, n_external_fold);
        }


    }
}
