package ednel;

import ednel.eda.CompileResultsTask;
import ednel.eda.EDNEL;
import ednel.eda.RunTrainingPieceTask;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class ProfilableMain {
    /**
     * This method loads the whole dataset 10 times, since it is split in 10 pairs of training-test subsets.
     * @param datasets_path Path where all datasets are stored.
     * @param dataset_name Name of dataset desired to be open
     * @return A HashMap object where each entry is an iteration of a 10-fold cross-validation (starting at 1 and ending
     * at 10), and each value a subsequent HashMap with training and test subsets.
     */
    public static HashMap<Integer, HashMap<String, Instances>> loadFoldsOfDatasets(
            String datasets_path, String dataset_name) throws Exception {

        HashMap<Integer, HashMap<String, Instances>> datasets = new HashMap<>();

        for(int j = 1; j < 11; j++) {  // 10 folds
            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource(
                    datasets_path + File.separator +
                            dataset_name + File.separator + dataset_name + "-10-" + j + "tra.arff"
            );
            ConverterUtils.DataSource test_set = new ConverterUtils.DataSource(
                    datasets_path + File.separator +
                            dataset_name + File.separator + dataset_name + "-10-" + j + "tst.arff"
            );

            Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
            train_data.setClassIndex(train_data.numAttributes() - 1);
            test_data.setClassIndex(test_data.numAttributes() - 1);

            datasets.put(
                    j, new HashMap<String, Instances>(){{
                   put("train", train_data);
                   put("test", test_data);
                }}
            );
        }
        return datasets;
    }

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
                .longOpt("metadata_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where runs results will be stored.")
                .build());

        options.addOption(Option.builder()
                .longOpt("resources_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder that contains support files, such as .csv with probability distributions, blocking" +
                        "variables, etc.")
                .build());

        options.addOption(Option.builder()
                .longOpt("seed")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Seed used to initialize base eda.classifiers (i.e. Weka-related). It is not used to bias PBIL.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_jobs")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of jobs to use. Will use one job per sample per fold. " +
                        "If unspecified or set to 1, will run in a single core.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_generations")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of generations to run the algorithm")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_individuals")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of individuals in the population")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_samples")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of times to run the algorithm")
                .build());

        options.addOption(Option.builder()
                .longOpt("learning_rate")
                .type(Float.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Learning rate of PBIL")
                .build());

        options.addOption(Option.builder()
                .longOpt("selection_share")
                .type(Float.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Fraction of fittest population to use to update graphical model")
                .build());

        options.addOption(Option.builder()
                .longOpt("burn_in")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of samples to discard at the start of the gibbs sampling process.")
                .build());

        options.addOption(Option.builder()
                .longOpt("thinning_factor")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("thinning factor used in the dependency network (i.e. interval to select samples)")
                .build());

        options.addOption(Option.builder()
                .longOpt("early_stop_generations")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of generations tolerated to have an improvement less than early_stop_tolerance")
                .build());

        options.addOption(Option.builder()
                .longOpt("early_stop_tolerance")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum tolerance between two generations that do not improve in the best individual fitness. " +
                        "Higher values are less tolerant.")
                .build());

        options.addOption(Option.builder()
                .longOpt("max_parents")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of probabilistic parents a variable is allowed to have.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_jobs")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of threads to run in parallel.")
                .build());

        options.addOption(Option.builder()
                .longOpt("timeout")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum amount of time (in seconds) that EDA has to run, until being prematurely terminated. " +
                        "If the algorithm does not finish before timeout, then the best individual from the current " +
                        "generation will be reported as final solution.")
                .build());

        options.addOption(Option.builder()
                .longOpt("log")
                .type(Boolean.class)
                .required(false)
                .hasArg(false)
                .desc("Whether to log metadata to files.")
                .build());

        options.addOption("t", false, "Whether to allow cycles in dependency network.");

        options.addOption(Option.builder()
                .longOpt("no_cycles")
                .type(Boolean.class)
                .required(false)
                .hasArg(false)
                .desc("Whether to allow cycles in dependency network.")
                .build());

        CommandLineParser parser = new DefaultParser();

        try {
            CommandLine commandLine = parser.parse(options, args);

            // writes metadata
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            String str_time = dtf.format(now);
            PBILLogger.metadata_path_start(str_time, commandLine);

            String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

            HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = loadFoldsOfDatasets(
                    commandLine.getOptionValue("datasets_path"),
                    dataset_names[0]
            );

            Instances train_data = curDatasetFolds.get(1).get("train");
            Instances test_data = curDatasetFolds.get(1).get("test");

            PBILLogger pbilLogger = new PBILLogger(
                    dataset_names[0],
                    commandLine.getOptionValue("metadata_path") + File.separator +
                            str_time + File.separator + dataset_names[0],
                    Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                    Integer.parseInt(commandLine.getOptionValue("n_generations")),
                    1, 1,
                    commandLine.hasOption("log")

            );

            EDNEL ednel = new EDNEL(
                    Double.parseDouble(commandLine.getOptionValue("learning_rate")),
                    Float.parseFloat(commandLine.getOptionValue("selection_share")),
                    Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                    Integer.parseInt(commandLine.getOptionValue("n_generations")),
                    Integer.parseInt(commandLine.getOptionValue("timeout", "-1")),
                    Integer.parseInt(commandLine.getOptionValue("burn_in")),
                    Integer.parseInt(commandLine.getOptionValue("thinning_factor")),
                    commandLine.hasOption("no_cycles"),
                    Integer.parseInt(commandLine.getOptionValue("early_stop_generations")),
                    Float.parseFloat(commandLine.getOptionValue("early_stop_tolerance")),
                    Integer.parseInt(commandLine.getOptionValue("max_parents")),
                    Integer.parseInt(commandLine.getOptionValue("delay_structure_learning")),
                    pbilLogger,
                    commandLine.getOptionValue("seed") == null?
                            null : Integer.parseInt(commandLine.getOptionValue("seed"))
            );
            ednel.buildClassifier(train_data);
//            ednel.distributionsForInstances(test_data);
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
