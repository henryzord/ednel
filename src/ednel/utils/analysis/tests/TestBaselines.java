/*
 * Puts baseline algorithms to run on given datasets.
 *
 * Currently only supports Random Forest.
 */

package ednel.utils.analysis.tests;

import ednel.Main;
import ednel.eda.individual.FitnessCalculator;
import org.apache.commons.cli.*;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Locale;

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
        CommandLine commandLine = TestBaselines.parseCommandLine(args);

        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

        for(String dataset_name : dataset_names) {
            double sum = 0;

            for(int n_fold = 1; n_fold < 11; n_fold++) {  // 10 folds
                HashMap<String, Instances> datasets = Main.loadDataset(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name,
                        n_fold
                );

                Instances train_data = datasets.get("train_data");
                Instances test_data = datasets.get("test_data");

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
