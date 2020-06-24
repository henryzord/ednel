package eda;

import eda.individual.FitnessCalculator;
import org.apache.commons.cli.*;
import utils.PBILLogger;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Main {
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
                .longOpt("variables_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder with variables and their parameters.")
                .build());

        options.addOption(Option.builder()
                .longOpt("options_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to options file.")
                .build());

        options.addOption(Option.builder()
                .longOpt("sampling_order_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to sampling order file.")
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
                .longOpt("cheat")
                .type(Boolean.class)
                .required(false)
                .desc("Whether to log test metadata during evolution.")
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
            .longOpt("nearest_neighbor")
            .type(Integer.class)
            .required(true)
            .hasArg()
            .numberOfArgs(1)
            .desc("Number of nearest neighbors to consider when calculating mutual information between continuous " +
                    "and discrete variables pairs.")
            .build());

        options.addOption(Option.builder()
                .longOpt("max_parents")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of parents a variable is allowed to have.")
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
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Whether to log metadata to files.")
                .build());

        CommandLineParser parser = new DefaultParser();

        System.err.println("WARNING: using only J48 as classifier!");

        try {
            CommandLine commandLine = parser.parse(options, args);

            int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));
            int n_jobs = Integer.parseInt(commandLine.getOptionValue("n_jobs"));

            // writes metadata
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd-MM-yyyy-HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            String str_time = dtf.format(now);
            PBILLogger.metadata_path_start(str_time, commandLine);

            String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

            // always 10 folds
            CompileResultsTask compiler = new CompileResultsTask(dataset_names, n_samples, 10);

            ThreadPoolExecutor executor = null;
            if(n_jobs > 1) {
                executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(n_jobs + 1);
                executor.execute(compiler);
            }

            for(String dataset_name : dataset_names) {
                System.out.println(String.format("On dataset %s:", dataset_name));

                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );

                for(int n_sample = 1; n_sample < n_samples + 1; n_sample++) {
                    for(int n_fold = 1; n_fold < 11; n_fold++) {  // 10 folds
                        Instances train_data = curDatasetFolds.get(n_fold).get("train");
                        Instances test_data = curDatasetFolds.get(n_fold).get("test");

                        RunTrainingPieceTask task = new RunTrainingPieceTask(
                                dataset_name, n_sample, n_fold, commandLine, str_time, train_data, test_data,
                                compiler.getClass().getMethod(
                                        "accessAUCData",
                                        String.class, int.class, int.class, String.class, Double[].class
                                ),
                                compiler
                        );
                        if(n_jobs > 1) {
                            executor.execute(task);
                        } else {
                            task.run();
                        }

                        Thread.sleep(1000);  // prevents creating all threads at once
//                        TimeUnit.SECONDS.sleep(1);
                    }
                }
            }
            if(n_jobs > 1) {
                executor.shutdown();
            } else {
                compiler.run();
            }
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
