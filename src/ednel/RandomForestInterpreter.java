package ednel;

import ednel.eda.EDNEL;
import ednel.eda.aggregators.RuleExtractor;
import ednel.eda.rules.ExtractedRule;
import org.apache.commons.cli.*;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;

public class RandomForestInterpreter {
    public static void main(String[] args) {
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

        options.addOption(Option.builder()
                .longOpt("output_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where a markdown file with generated classifier will be written to.")
                .build());

        CommandLineParser parser = new DefaultParser();

        try {
            CommandLine commandLine = parser.parse(options, args);

            String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

            for(String dataset_name : dataset_names) {

                System.out.print(String.format("on dataset %s... ", dataset_name));

                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = TestDataset.loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );

                Instances train_data = curDatasetFolds.get(1).get("train");

                RandomForest rf = new RandomForest();
                rf.setNumIterations(2);
                rf.buildClassifier(train_data);

                ExtractedRule[] rules = RuleExtractor.fromRandomForestToRules(rf, train_data);

                int z = 0;



//                Instances test_data = curDatasetFolds.get(1).get("test");


            }
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
