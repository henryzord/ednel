package ednel.utils.analysis.optimizers;

import ednel.eda.EDNEL;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

public class AUTOCVEProcedure {

    public static final int N_TRIALS = 10;
    public static final int DEFAULT_N_JOBS = 10;

    public enum SupportedAlgorithms {
        EDNEL, RandomForest
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

        options.addOption(Option.builder()
                .longOpt("classifier")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of classifier to optimize with nested cross validation")
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

    public static HashMap<String, Instances> loadHoldoutDataset(String datasets_path, String dataset_name, int id_trial) throws Exception {
        HashMap<String, Instances> datasets = new HashMap<>();

        ConverterUtils.DataSource train_set = new ConverterUtils.DataSource(
                datasets_path + File.separator + String.format("dataset_%s_trial_%02d_train.arff", dataset_name, id_trial)
        );
        ConverterUtils.DataSource test_set = new ConverterUtils.DataSource(
                datasets_path + File.separator + String.format("dataset_%s_trial_%02d_test.arff", dataset_name, id_trial)
        );

        Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
        train_data.setClassIndex(train_data.numAttributes() - 1);
        test_data.setClassIndex(test_data.numAttributes() - 1);

        datasets.put("train_data", train_data);
        datasets.put("test_data", test_data);

        return datasets;
    }

    private static Object runHoldout(
            SupportedAlgorithms algorithmName, int n_try,
            String dataset_name, String datasets_path, String dataset_experiment_path
    ) {
        try {
            HashMap<String, Instances> datasets = AUTOCVEProcedure.loadHoldoutDataset(
                    datasets_path,
                    dataset_name,
                    n_try
            );
            Instances train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances test_data = datasets.get("test_data");  // 1/10 of external cv folds

            switch(algorithmName) {
                case EDNEL:
                    HashMap<String, Object> hyperparameters = AUTOCVEProcedure.getEDNELCombinations();

                    // float learning_rate, float selection_share, int n_individuals, int n_generations,
                    // int timeout, int timeout_individual, int burn_in, int thinning_factor, boolean no_cycles, int early_stop_generations,
                    // int max_parents, int delay_structure_learning, int n_internal_folds, PBILLogger pbilLogger, Integer seed

                    EDNEL ednel = new EDNEL(
                        Float.parseFloat(hyperparameters.get("learning_rate").toString()),
                        Float.parseFloat(hyperparameters.get("selection_share").toString()),
                        Integer.parseInt(hyperparameters.get("n_individuals").toString()),
                        Integer.parseInt(hyperparameters.get("n_generations").toString()),
                        Integer.parseInt(hyperparameters.get("timeout").toString()),
                        Integer.parseInt(hyperparameters.get("timeout_individual").toString()),
                        Integer.parseInt(hyperparameters.get("burn_in").toString()),
                        Integer.parseInt(hyperparameters.get("thinning_factor").toString()),
                        Boolean.parseBoolean(hyperparameters.get("no_cycles").toString()),
                        Integer.parseInt(hyperparameters.get("early_stop_generations").toString()),
                        Integer.parseInt(hyperparameters.get("max_parents").toString()),
                        Integer.parseInt(hyperparameters.get("delay_structure_learning").toString()),
                        Integer.parseInt(hyperparameters.get("n_internal_folds").toString()),
                        FitnessCalculator.EvaluationMetric.BALANCED_ACCURACY,
                        new PBILLogger(
                            dataset_name, dataset_experiment_path,
                            Integer.parseInt(hyperparameters.get("n_individuals").toString()),
                            Integer.parseInt(hyperparameters.get("n_generations").toString()),
                            n_try, 0, true, false
                        ),
                        (Integer)hyperparameters.get("seed")
                    );
                    ednel.buildClassifier(train_data);
                    // dist = ednel.distributionsForInstances(test_data);

                    System.err.println("TODO using best individual from last generation!");

                    HashMap<String, Individual> toReport = new HashMap<String, Individual>(1){{
                        put("last", ednel.getCurrentGenBest());
                    }};

                    ednel.getPbilLogger().toFile(
                            ednel.getDependencyNetwork(), toReport, test_data
                    );
                    return true;
                case RandomForest:
                    RandomForest rf = new RandomForest();
                    rf.buildClassifier(train_data);

                    PBILLogger.write_predictions_to_file(
                            new AbstractClassifier[]{rf},
                            test_data,
                            dataset_experiment_path + File.separator +
                                    String.format(
                                            "overall%stest_sample-%02d_fold-%02d_%s.preds",
                                            File.separator,
                                            n_try,
                                            0,
                                            rf.getClass().getSimpleName()
                                    )
                    );
                    System.out.printf("Done: dataset %s, trial %02d%n", dataset_name, n_try);
                    return true;
                default:
                    throw new IllegalStateException("Unexpected value: " + algorithmName);
            }
        } catch(Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace(System.err);
            return e;
        }
    }

    /**
     * Performs the same experimental setup as described in the AUTOCVE work with dynamic resampling (i.e. a holdout
     * with 70/30 split).
     *
     * @param algorithmName Name of classifier
     * @param n_external_folds Number of external folds.
     * @param dataset_name Name of datasets to test
     * @param datasets_path Path where datasets are stored, one folder for each dataset, and inside each dataset folder,
     *                      two csv files for each fold (one for the training data, and another for the testing data)
     * @param experiment_metadata_path Path to where write metadata on this experiment
     */
    private static void classifierOptimization(
            SupportedAlgorithms algorithmName, int n_jobs,
            String dataset_name, String datasets_path, String experiment_metadata_path
    ) {
        n_jobs = Math.min(n_jobs, N_TRIALS);  // 10 trials

        int n_loops = (int)(N_TRIALS / n_jobs);

        int counter = 0;
        Object[] answers;
        for(int i = 0; i < n_loops; i++) {
            answers = IntStream.range((i * n_jobs) + 1, ((i + 1) * n_jobs) + 1).parallel().mapToObj(
                    j -> AUTOCVEProcedure.runHoldout(
                            algorithmName,
                            j, dataset_name, datasets_path,
                            experiment_metadata_path + File.separator + dataset_name)
            ).toArray();
            counter += n_jobs;
        }

        if(counter < N_TRIALS) {
            answers = IntStream.range(counter, N_TRIALS + 1).parallel().mapToObj(
                    j -> AUTOCVEProcedure.runHoldout(
                            algorithmName,
                            j, dataset_name, datasets_path,
                            experiment_metadata_path + File.separator + dataset_name)
            ).toArray();
        }
    }

    private static ArrayList<HashMap<String, Object>> getDecisionTableCombinations() {
        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        String[] useIBk_array = {"-I", ""};
        String[] evaluationMeasure_array = {"-E acc", "-E rmse", "-E mae", "-E auc"};
        String crossVal = "-X 1";
        String[] search_array = {"-S weka.attributeSelection.BestFirst -D 1 -N 5", "-S weka.attributeSelection.GreedyStepwise"};

        for(String useIBk : useIBk_array) {
            for(String evaluationMeasure : evaluationMeasure_array) {
                for(String search : search_array) {
                    HashMap<String, Object> comb = new HashMap<>();

                    comb.put("useIBk", useIBk);
                    comb.put("evaluationMeasure", evaluationMeasure);
                    comb.put("crossVal", crossVal);
                    comb.put("search", search);

                    combinations.add(comb);
                }
            }
        }
        return combinations;
    }

    private static HashMap<String, Object> getEDNELCombinations() {
        float learning_rate = 0.52f;  // ok, checked
        float selection_share = 0.5f;  // ok, checked
        int n_individuals = 50;  // ok, checked
        int n_generations = 100;  // ok, checked

        int timeout = 5400;  // ok, checked
        int timeout_individual = 60;  // ok, checked

        int burn_in = 100;  // ok, checked
        int thinning_factor = 0;  // ok, checked

        boolean no_cycles = false;  // ok, checked

        int early_stop_generations = 20;  // ok, checked
        int max_parents = 1;  // ok, checked

        int delay_structure_learning = 5;  // ok, checked

        int n_internal_folds = 5;  // ok, checked

        HashMap<String, Object> comb = new HashMap<>();

        comb.put("learning_rate", learning_rate);
        comb.put("selection_share", selection_share);
        comb.put("n_individuals", n_individuals);
        comb.put("n_generations", n_generations);
        comb.put("timeout", timeout);
        comb.put("timeout_individual", timeout_individual);
        comb.put("burn_in", burn_in);
        comb.put("thinning_factor", thinning_factor);
        comb.put("no_cycles", no_cycles);
        comb.put("early_stop_generations", early_stop_generations);
        comb.put("max_parents", max_parents);
        comb.put("delay_structure_learning", delay_structure_learning);
        comb.put("n_internal_folds", n_internal_folds);
        comb.put("pbilLogger", null);
        comb.put("seed", null);

        return comb;
    }

    private static ArrayList<HashMap<String, Object>> getRandomForestCombinations() {
        String breakTiesRandomly = "";  // always false
        String maxDepth = "-depth 0";  // always unlimited
        String numIterations = "-I 1000";  // always 1000

        String[] bagSizePercent_array = {"-P 90", "-P 100"};
        String[] numFeatures_array = {
                "-K {Math.sqrt(p)/2}",
                "-K {Math.sqrt(p)}",
                "-K {Math.sqrt(p)*2}",
                "-K {Math.log(p)/(Math.log(2)*2)}",
                "-K {Math.log(p)/Math.log(2)}",
                "-K {(Math.log(p)/Math.log(2))*2}"
        };  // must be replaced by the actual number of features

        ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();

        for(String numFeatures : numFeatures_array) {
            for(String bagSizePercent : bagSizePercent_array) {
                HashMap<String, Object> comb = new HashMap<>();

                comb.put("breakTiesRandomly", breakTiesRandomly);
                comb.put("maxDepth", maxDepth);
                comb.put("numIterations", numIterations);
                comb.put("bagSizePercent", bagSizePercent);
                comb.put("numFeatures", numFeatures);
                combinations.add(comb);
            }
        }
        return combinations;
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = AUTOCVEProcedure.parseCommandLine(args);

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

        int n_jobs = DEFAULT_N_JOBS;
        if(options.get("n_jobs") != null) {
            n_jobs = Integer.parseInt(options.get("n_jobs"));
            if(n_jobs <= 0) {
                throw new Exception("Number of jobs must be positive and non-negative!");
            }
        }

        System.err.println("TODO implement code to run for other values of n_external_folds! (currently only supports = 10)");

        String clf_name = options.get("classifier");

        SupportedAlgorithms algorithm_name;

        switch(clf_name) {
            case "EDNEL":
                algorithm_name = SupportedAlgorithms.EDNEL;
                break;
            case "RandomForest":
                algorithm_name = SupportedAlgorithms.RandomForest;
                break;
            default:
                throw new Exception(String.format("Nested cross-validation for classifier %s is not implemented yet.", clf_name));
        }

        AUTOCVEProcedure.classifierOptimization(
                algorithm_name, n_jobs, dataset_name, datasets_path, experiment_metadata_path
        );
    }
}
