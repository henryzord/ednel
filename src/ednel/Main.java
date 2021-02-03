package ednel;

import ednel.eda.RunTrainingPieceTask;
import ednel.utils.PBILLogger;
import ednel.utils.operators.*;
import jdk.nashorn.internal.runtime.regexp.joni.exception.ValueException;
import org.apache.commons.cli.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

public class Main {

    private static final int init_fold = 1;
    private static final int end_fold = 11;
    private static final int n_folds = 10;

//    /**
//     * This method loads the whole dataset 10 times, since it is split in 10 pairs of training-test subsets.
//     * @param datasets_path Path where all datasets are stored.
//     * @param dataset_name Name of dataset desired to be open
//     * @return A HashMap object where each entry is an iteration of a 10-fold cross-validation (starting at 1 and ending
//     * at 10), and each value a subsequent HashMap with training and test subsets.
//     */
//    public static HashMap<Integer, HashMap<String, Instances>> loadFoldsOfDatasets(
//            String datasets_path, String dataset_name) throws Exception {
//
//        HashMap<Integer, HashMap<String, Instances>> datasets = new HashMap<>();
//
//        for(int j = 1; j < 11; j++) {  // 10 folds
//            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource(
//                    datasets_path + File.separator +
//                            dataset_name + File.separator + dataset_name + "-10-" + j + "tra.arff"
//            );
//            ConverterUtils.DataSource test_set = new ConverterUtils.DataSource(
//                    datasets_path + File.separator +
//                            dataset_name + File.separator + dataset_name + "-10-" + j + "tst.arff"
//            );
//
//            Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
//            train_data.setClassIndex(train_data.numAttributes() - 1);
//            test_data.setClassIndex(test_data.numAttributes() - 1);
//
//            datasets.put(
//                    j, new HashMap<String, Instances>(){{
//                   put("train", train_data);
//                   put("test", test_data);
//                }}
//            );
//        }
//        return datasets;
//    }

    public static HashMap<String, Instances> loadDataset(
            String datasets_path, String dataset_name, int n_fold) throws Exception {

        HashMap<String, Instances> datasets = new HashMap<>();

        ConverterUtils.DataSource train_set = new ConverterUtils.DataSource(
                datasets_path + File.separator +
                        dataset_name + File.separator + dataset_name + "-10-" + n_fold + "tra.arff"
        );
        ConverterUtils.DataSource test_set = new ConverterUtils.DataSource(
                datasets_path + File.separator +
                        dataset_name + File.separator + dataset_name + "-10-" + n_fold + "tst.arff"
        );

        Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
        train_data.setClassIndex(train_data.numAttributes() - 1);
        test_data.setClassIndex(test_data.numAttributes() - 1);

        datasets.put("train_data", train_data);
        datasets.put("test_data", test_data);

        return datasets;
    }


    public static HashMap<String, String> parseCommandLine(String[] args) throws ParseException {
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
                .longOpt("selection_share")
                .type(Float.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Fraction of population that should be resampled in the next generation. Assume " +
                        "population_size - (population_size * selection_share) as the fraction that updates GM.")
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
                .longOpt("max_parents")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of probabilistic parents a variable is allowed to have.")
                .build());

        options.addOption(Option.builder()
                .longOpt("delay_structure_learning")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("How many generations are needed to learn the structure of the network")
                .build());

        options.addOption(Option.builder()
                .longOpt("early_stop_generations")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of generations tolerated to have an improvement less than early_stop_tolerance")
                .build());

        options.addOption(Option.builder()
                .longOpt("burn_in")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of samples to discard at the start of the gibbs sampling process.")
                .build());

        options.addOption(Option.builder()
                .longOpt("thinning_factor")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("thinning factor used in the dependency network (i.e. interval to select samples)")
                .build());

        options.addOption(Option.builder()
                .longOpt("no_cycles")
                .type(Boolean.class)
                .required(false)
                .hasArg(false)
                .desc("Whether to allow cycles in dependency network.")
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
                .longOpt("n_samples")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of times to run the algorithm")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_internal_folds")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of internal folds to be used. use 0 for holdout (80/20) on the training set, 1 for leave-" +
                        "one-out, or any number greater or equal to 2 for n-fold cross-validation. Defaults to 5" +
                        "if no value is passed.")
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
                .longOpt("timeout_individual")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum amount of time (in seconds) that EDA has to generate and fully evaluate an individual. " +
                        "Will discard individual if building it exceeds this time limit, and generate a new individual.")
                .build());

        options.addOption(Option.builder()
                .longOpt("log")
                .type(Boolean.class)
                .required(false)
                .hasArg(false)
                .desc("Whether to log metadata to files.")
                .build());

        //        options.addOption(Option.builder()
//                .longOpt("early_stop_tolerance")
//                .type(Integer.class)
//                .required(false)
//                .hasArg()
//                .numberOfArgs(1)
//                .desc("Maximum tolerance between two generations that do not improve in the best individual fitness. " +
//                        "Higher values are less tolerant.")
//                .build());

        CommandLineParser parser = new DefaultParser();

        return Main.treatOptions(parser.parse(options, args));
    }

    /**
     * Treats commandline values, in a way that no uncompatible values are set at the same time -- otherwise throws an
     * exception.
     * @return A HashMap with treated values
     * @throws ValueException if conflicting parameter values are encountered
     */
    private static HashMap<String, String> treatOptions(CommandLine cmd) throws ValueException {
        HashMap<String, String> options = new HashMap<>();
        for(Option opt : cmd.getOptions()) {
            options.put(opt.getLongOpt(), opt.getValue());
        }
        // boolean parameters
        options.put("log", cmd.hasOption("log")? "true" : "false");
        options.put("no_cycles", cmd.hasOption("no_cycles")? "true" : "false");

        // checks if datasets_path and metadata_path exists
        String[] check_exists = {"datasets_path", "metadata_path"};
        for(String path : check_exists) {
            if(!Files.exists(Paths.get(options.get(path)))) {
                throw new ValueException(String.format("Path does not exists: %s", options.get(path)));
            }
        }

        // simple check for limits
        Boolean[] required = {true, true, true, true, false, false, false, false, false, false, false, false, false, false};
        String[] parameters = {"n_individuals", "n_generations", "selection_share", "learning_rate", "burn_in",
                "thinning_factor", "max_parents", "delay_structure_learning", "early_stop_generations", "n_jobs",
                "n_samples", "timeout", "timeout_individual", "n_internal_folds"};
        Double[] lower_limits = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 60.0, 60.0, 0.0};
        Double[] upper_limits = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1.0, 1.0, Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY,
                Double.POSITIVE_INFINITY, 30.0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
        AbstractOperator[] lower_operators = {new GreaterThan(), new GreaterThanOrEqualTo(), new GreaterThan(),
                new GreaterThan(), new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(),
                new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(),
                new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo(), new GreaterThanOrEqualTo()};
        AbstractOperator[] upper_operators = {new LessThan(), new LessThan(), new LessThan(), new LessThanOrEqualTo(),
                new LessThan(), new LessThan(), new LessThan(), new LessThan(), new LessThan(), new LessThanOrEqualTo(),
                new LessThanOrEqualTo(), new LessThan(), new LessThan(), new LessThan()};

        for(int i = 0; i < parameters.length; i++) {
            if(!options.containsKey(parameters[i])) {
                if(required[i]) {
                    throw new ValueException(String.format("Missing parameter %s", parameters[i]));
                }
            }
            try {
                double val = Double.parseDouble(options.get(parameters[i]));

                if(!lower_operators[i].operate(val, lower_limits[i]) || !upper_operators[i].operate(val, upper_limits[i])) {
                    throw new ValueException(String.format(
                            "%s must be between %s%f, %f%s",
                            parameters[i],
                            lower_operators[i].getClass() == GreaterThanOrEqualTo.class? "[" : "(",
                            lower_limits[i],
                            upper_limits[i],
                            upper_operators[i].getClass() == LessThanOrEqualTo.class? "]" : ")"
                    ));
                }

            } catch(NumberFormatException nfe) {
                throw new ValueException(String.format("%s does not contains a numeric value: %s", parameters[i], options.get(parameters[i])));
            } catch(NullPointerException npe) {
                // if the parameter is not required and is not present, does nothing; will treat later
            }
        }

        // sets default values for hyper-parameters that were not set
        if(!options.containsKey("burn_in")) {
            options.put("burn_in", "100");
        }
        if(!options.containsKey("thinning_factor")) {
            options.put("thinning_factor", "0");
        }
        if(!options.containsKey("max_parents")) {
            options.put("max_parents", "1");
        }
        if(!options.containsKey("delay_structure_learning")) {
            options.put("delay_structure_learning", "5");
        }
        if(!options.containsKey("early_stop_generations")) {
            options.put("early_stop_generations", "10");
        }
        if(!options.containsKey("seed")) {
            options.put("seed", null);
        }
        if(!options.containsKey("n_jobs")) {
            options.put("n_jobs", "1");
        }
        if(!options.containsKey("n_samples")) {
            options.put("n_samples", "1");
        }
        if(!options.containsKey("timeout")) {
            options.put("timeout", "-1");  // no timeout
        }
        if(!options.containsKey("timeout_individual")) {
            options.put("timeout_individual", "60");
        }
        if(!options.containsKey("n_internal_folds")) {
            options.put("n_internal_folds", "5");
        }

        // now that all hyper-parameters are set, treat their values
        if(Boolean.parseBoolean(options.get("no_cycles"))) {
            options.put("burn_in", "0");  // no burn_in is needed if using bayesian networks
            options.put("thinning_factor", "0");  // thinning_factor makes no sense if using bayesian networks
        }

        if(Integer.parseInt(options.get("max_parents")) == 0) {
            // no need to learn a structure in the network if no relationships are allowed
            options.put("delay_structure_learning", "0");
        }
        return options;
    }

    public static void main(String[] args) throws Exception {
        HashMap<String, String> commandLine = parseCommandLine(args);

        int n_samples = Integer.parseInt(commandLine.get("n_samples"));
        int n_jobs = Integer.parseInt(commandLine.get("n_jobs"));

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);
        PBILLogger.metadata_path_start(str_time, commandLine);

        String[] dataset_names = commandLine.get("datasets_names").split(",");

        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(n_jobs);

        final int n_tasks = dataset_names.length * n_samples * Main.n_folds;

        ArrayList<Callable<Object>> taskQueue = new ArrayList<>(n_tasks);

        for(String dataset_name : dataset_names) {
            for(int n_sample = 1; n_sample < n_samples + 1; n_sample++) {
                for(int n_fold = Main.init_fold; n_fold < Main.end_fold; n_fold++) {  // 10 folds
                    RunTrainingPieceTask task = new RunTrainingPieceTask(
                            dataset_name, n_sample, n_fold, commandLine, str_time
                    );
                    taskQueue.add(Executors.callable(task));
                }
            }
        }
        ArrayList<Future<Object>> answers = (ArrayList<Future<Object>>)executor.invokeAll(taskQueue);
        executor.shutdown();
        int finished = 0;
        for(int i = 0; i < answers.size(); i++) {
            finished += answers.get(i).isDone()? 1 : 0;
        }
        System.out.println(String.format("%d tasks completed", finished));

    }
}
