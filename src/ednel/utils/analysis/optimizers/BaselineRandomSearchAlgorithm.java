/*
  A script for running a simple Random Search algorithm. This algorithm is basically EDNEL ran without learning anything,
  just random-guessing solutions and picking the best solution found overall as candidate.
 */

package ednel.utils.analysis.optimizers;

import ednel.Main;
import ednel.eda.EDNEL;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

public class BaselineRandomSearchAlgorithm {
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
                .longOpt("n_individuals")
                .required(true)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("How many solutions (individuals) to sample.")
                .build());

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = BaselineRandomSearchAlgorithm.parseCommandLine(args);

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

        int n_individuals = Integer.parseInt(options.get("n_individuals"));

        for(int n_external_fold = 1; n_external_fold <= n_external_folds; n_external_fold++) {

            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances test_data = datasets.get("test_data");  // 1/10 of external cv folds

            EDNEL ednel = new EDNEL(
                0.1f, // any value, doesn't matter
                0.1f, // any value, doesn't matter
                n_individuals,  // as many individuals as requested
                1,  // one generation
                0, // timeout: no limit
                0, // individuals timeout: no limit
                100, // burn in: default value
                0, // thinning factor: default value
                false, // no cycles: default value
                1, // early stop generations: any value, doesn't matter
                0, // max parents: zero, but doesn't matter, will not learn anything
                5, // delay structure learning: 5 but doesn't matter, won't learn any structure
                0, // n internal folds: left at zero to perform an internal holdout, but doesn't matter
                new PBILLogger(
                    dataset_name, dataset_experiment_path,
                    n_individuals,
                    1,
                    1, n_external_fold, true, false
                ),  // pbillogger: null, won't log anything
                null  // seed: null, use clock
            );

            ednel.buildClassifier(train_data);
            Individual overallBest = ednel.getOverallBest();

            HashMap<String, Individual> toReport = new HashMap<>(1);
            toReport.put("overall", ednel.getOverallBest());

            ednel.getPbilLogger().toFile(
                    ednel.getDependencyNetwork(), toReport, test_data
            );

            PBILLogger.write_predictions_to_file(
                    new AbstractClassifier[]{overallBest},
                    test_data,
                    dataset_experiment_path + File.separator +
                            String.format(
                                    "overall%stest_sample-01_fold-%02d_%s.preds",
                                    File.separator,
                                    n_external_fold,
                                    overallBest.getClass().getSimpleName()
                            )
            );
            System.out.printf("Done: Dataset %s fold %02d\n", dataset_name, n_external_fold);
        }
    }
}
