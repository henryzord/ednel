package ednel;

import ednel.eda.CompileResultsTask;
import ednel.eda.RunTrainingPieceTask;
import ednel.eda.individual.FitnessCalculator;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class TestBaselines {

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
                .longOpt("datasets_names")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of the datasets to run experiments. Must be a list separated by a comma\n" +
                        "\tExample: iris,mushroom,adult")
                .build()
        );

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = parseCommandLine(args);

        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

        for(String dataset_name : dataset_names) {
            HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = Main.loadFoldsOfDatasets(
                    commandLine.getOptionValue("datasets_path"),
                    dataset_name
            );

            double sum = 0;

            for(int n_fold = 1; n_fold < 11; n_fold++) {  // 10 folds
                Instances train_data = curDatasetFolds.get(n_fold).get("train");
                Instances test_data = curDatasetFolds.get(n_fold).get("test");

                RandomForest rf = new RandomForest();
                Evaluation ev = new Evaluation(train_data);
                rf.buildClassifier(train_data);
                ev.evaluateModel(rf, test_data);

                sum += FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, rf);
            }

            sum /= 10;
            System.out.println(String.format(Locale.US, "%s,%f", dataset_name, sum));
        }
    }
}
