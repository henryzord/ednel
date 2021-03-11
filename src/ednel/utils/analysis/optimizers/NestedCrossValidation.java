package ednel.utils.analysis.optimizers;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.Main;
import ednel.eda.EDNEL;
import ednel.eda.individual.EmptyEnsembleException;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import ednel.utils.analysis.FoldJoiner;
import jdk.nashorn.internal.runtime.arrays.ArrayIndex;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import javax.script.ScriptEngineManager;
import javax.script.ScriptEngine;
import javax.script.ScriptException;

import javax.script.ScriptException;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.Locale;
import java.util.stream.IntStream;

public class NestedCrossValidation {
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
                .longOpt("n_external_folds")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of external folds to use in nested cross validation.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_internal_folds")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of folds to use in nested cross validation.")
                .build());

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

    private static void RandomForestParametersToFile(
            int n_external_fold, String dataset_experiment_path, String dataset_name,
            String bagSizePercent, String breakTiesRandomly, String maxDepth, String numFeatures, String numIterations
    ) throws IOException {

        HashMap<String, String> obj = new HashMap<>();

        obj.put("dataset_experiment_path", dataset_experiment_path);
        obj.put("dataset_name", dataset_name);
        obj.put("bagSizePercent", bagSizePercent.split(" ")[1]);
        obj.put("breakTiesRandomly", breakTiesRandomly.length() > 0? "true" : "false");
        obj.put("maxDepth", maxDepth.split(" ")[1]);
        obj.put("numFeatures", numFeatures.split(" ")[1]);
        obj.put("numIterations", numIterations.split(" ")[1]);

        FileWriter fw = new FileWriter(
                dataset_experiment_path + File.separator + String.format("test_sample-01_fold-%02d_parameters.json", n_external_fold)
        );

        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        fw.write(converter.toJson(obj));
        fw.flush();
        fw.close();
    }

    private static void EDNELparametersToFile(
            int n_external_fold, String dataset_experiment_path, String dataset_name, float selection_share, int n_individuals,
            int n_generations, int timeout, int timeout_individual, int burn_in, int thinning_factor, boolean no_cycles,
            float learning_rate, int early_stop_generations, int max_parents, int delay_structure_learning, boolean bestUsesOverall
    ) throws IOException {

        HashMap<String, String> obj = new HashMap<>();

        obj.put("dataset_experiment_path", dataset_experiment_path);
        obj.put("dataset_name", dataset_name);
        obj.put("selection_share", String.valueOf(selection_share));
        obj.put("n_individuals", String.valueOf(n_individuals));
        obj.put("n_generations", String.valueOf(n_generations));
        obj.put("timeout", String.valueOf(timeout));
        obj.put("timeout_individual", String.valueOf(timeout_individual));
        obj.put("burn_in", String.valueOf(burn_in));
        obj.put("thinning_factor", String.valueOf(thinning_factor));
        obj.put("no_cycles", String.valueOf(no_cycles));
        obj.put("learning_rate", String.valueOf(learning_rate));
        obj.put("early_stop_generations", String.valueOf(early_stop_generations));
        obj.put("max_parents", String.valueOf(max_parents));
        obj.put("delay_structure_learning", String.valueOf(delay_structure_learning));
        obj.put("individual", bestUsesOverall? "overall" : "last");


        FileWriter fw = new FileWriter(
                dataset_experiment_path + File.separator + String.format("test_sample-01_fold-%02d_parameters.json", n_external_fold)
        );

        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        fw.write(converter.toJson(obj));
        fw.flush();
        fw.close();
    }

    private static String getRandomForestNumFeatures(String param, Instances data) throws ScriptException {
        ScriptEngineManager mgr = new ScriptEngineManager();
        ScriptEngine engine = mgr.getEngineByName("JavaScript");
        String segment = param.substring(param.indexOf("{") + 1, param.indexOf("}"));
        segment = segment.replace("p", String.valueOf(data.numAttributes()));
        String new_param = param.substring(0, param.indexOf("{")) + Math.round(Double.valueOf(engine.eval(segment).toString()));
        return new_param;
    }

    private static String[] atomize(String[] params) {
        String[] new_string = new String[params.length * 2];
        int counter = 0;
        for(int i = 0; i < params.length; i++) {
            if(params[i].length() > 0) {
                String[] splitted = params[i].split(" ");
                new_string[counter] = splitted[0];
                try {
                    new_string[counter + 1] = splitted[1];
                } catch(ArrayIndexOutOfBoundsException aei) {
                    new_string[counter + 1] = "";
                }
            } else {
                new_string[counter] = "";
                new_string[counter + 1] = "";
            }
            counter += 2;
        }
        return new_string;
    }

    private static Object runExternalCrossValidationFoldRandomForest(
            int n_external_fold, int n_internal_folds, String datasets_path, String dataset_name,
            String dataset_experiment_path, String clf_name) {

        try {
//            String[] samplingOrder = new String[]{"bagSizePercent", "breakTiesRandomly", "maxDepth", "numFeatures", "numIterations"};
    //        String[] optionNames = new String[]{"-P", ";-B", "-depth", "-K", "-I"};
    //        int[][] ranges = new int[][]{{20, 90}, {0, 1}, {0, 10}, {1, 100}, {1000, 1000}};

            String[] bagSizePercents = {"-P 90", "-P 100"};
            String[] breakTiesRandomly = {""};  // always false
            String[] maxDepths = {"-depth 0"};  // always unlimited
            String[] numFeatures = {
                    "-K {Math.sqrt(p)/2}",
                    "-K {Math.sqrt(p)}",
                    "-K {Math.sqrt(p)*2}",
                    "-K {Math.log(p)/(Math.log(2)*2)}",
                    "-K {Math.log(p)/Math.log(2)}",
                    "-K {(Math.log(p)/Math.log(2))*2}"
            };  // must be replaced by the actual number of features
            String[] numIterations = {"-I 1000"};  // always 1000


            String[][] combinations = new String[
                    bagSizePercents.length * breakTiesRandomly.length * maxDepths.length *
                    numFeatures.length * numIterations.length
            ][5];  // 5 options

            int counter_comb = 0;
            for(String bsp : bagSizePercents) {
                for(String btr : breakTiesRandomly) {
                    for(String mp : maxDepths) {
                        for(String nf : numFeatures) {
                            for(String ni : numIterations) {
                                combinations[counter_comb] = new String[]{bsp, btr, mp, nf, ni}; // , "-S 1", "-num-slots 1", "-M 1.0"};  // puts seed, n_jobs, and minNumObjs
                                counter_comb += 1;
                            }
                        }
                    }
                }
            }

            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances external_train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances external_test_data = datasets.get("test_data");  // 1/10 of external cv folds

            // prepares data for future internal cross validation
            external_train_data = FitnessCalculator.betterStratifier(external_train_data, n_internal_folds);

            double[][] predictionMatrix = new double[external_train_data.size()][];
            double[] actualClasses = new double[external_train_data.size()];

            int counter_combination = 0;

            int best_combination_index = -1;
            double best_combination_auc = Double.NEGATIVE_INFINITY;

            for(String[] comb : combinations) {  // iterates over combinations of hyper-parameters
                int counter_instance = 0;
                for(int i = 0; i < n_internal_folds; i++) {
                    Instances internal_train_data = external_train_data.trainCV(n_internal_folds, i);
                    Instances internal_test_data = external_train_data.testCV(n_internal_folds, i);

                    RandomForest rf = new RandomForest();
                    String[] copy = comb.clone();
                    copy[3] = NestedCrossValidation.getRandomForestNumFeatures(comb[3], internal_train_data);
                    rf.setOptions(NestedCrossValidation.atomize(copy));

                    rf.buildClassifier(internal_train_data);

                    double[][] preds = rf.distributionsForInstances(internal_test_data);

                    for(int j = 0; j < internal_test_data.size(); j++) {
                        predictionMatrix[counter_instance] = preds[j];
                        actualClasses[counter_instance] = internal_test_data.instance(j).value(internal_test_data.classIndex());
                        counter_instance += 1;
                    }
                }
                FoldJoiner foldJoiner = new FoldJoiner(predictionMatrix, actualClasses);

                double auc = foldJoiner.getAUC("classifier");

                if(auc > best_combination_auc) {
                    best_combination_index = counter_combination;
                    best_combination_auc = auc;
                }
                counter_combination += 1;
            }

            NestedCrossValidation.RandomForestParametersToFile(
                    n_external_fold,
                    dataset_experiment_path, dataset_name,
                    combinations[best_combination_index][0],
                    combinations[best_combination_index][1],
                    combinations[best_combination_index][2],
                    combinations[best_combination_index][3],
                    combinations[best_combination_index][4]
            );

            BufferedWriter opt_file = new BufferedWriter(
                    new FileWriter(
                            new File(
                                    dataset_experiment_path + File.separator +
                                    String.format(
                                            "overall%stest_sample-01_fold-%02d_RandomForest_nestedcv_optimization.csv",
                                            File.separator, n_external_fold
                                    )
                            )
                    )
            );
            opt_file.write("dataset_name,n_sample,n_fold,unweighted_area_under_roc,classifier size,elapsed time (seconds),characteristics,options\n");

            LocalDateTime t1 = LocalDateTime.now();

            RandomForest rf = new RandomForest();

            String[] copy = combinations[best_combination_index].clone();
            copy[3] = NestedCrossValidation.getRandomForestNumFeatures(combinations[best_combination_index][3], external_train_data);
            rf.setOptions(NestedCrossValidation.atomize(copy));

            rf.buildClassifier(external_train_data);

            PBILLogger.write_predictions_to_file(
                    new AbstractClassifier[]{rf},
                    external_test_data,
                    dataset_experiment_path + File.separator + String.format("overall%stest_sample-01_fold-%02d_RandomForest.preds", File.separator, n_external_fold)
            );

            Evaluation ev = new Evaluation(external_train_data);
            ev.evaluateModel(rf, external_test_data);
            double unweightedAuc = FitnessCalculator.getUnweightedAreaUnderROC(external_train_data, external_test_data, rf);
            LocalDateTime t2 = LocalDateTime.now();

            long elapsed_time = t1.until(t2, ChronoUnit.SECONDS);
            // "dataset_name,n_sample,n_fold,unweighted_area_under_roc,classifier size,elapsed time (seconds)\n"

            String characteristics_str = String.format(
                    Locale.US,
                    "bagSizePercent=%d, breakTiesRandomly=%s, maxDepth=%d, numFeatures=%s, numIterations=%d",
                    Integer.parseInt(combinations[best_combination_index][0].split(" ")[1]),
                    combinations[best_combination_index][1].contains("-B")? "true" : "false",
                    Integer.parseInt(combinations[best_combination_index][2].split(" ")[1]),
                    combinations[best_combination_index][3],
                    Integer.parseInt(combinations[best_combination_index][4].split(" ")[1])
            );

            StringBuilder options_str_builder = new StringBuilder("");
            for(int j = 0; j < combinations[best_combination_index].length; j++) {
                options_str_builder.append(combinations[best_combination_index][j]).append(" ");
            }

            opt_file.write(String.format(
                    Locale.US,
                    "%s,%d,%d,%f,%d,%d,\"%s\",\"%s\"\n",
                    dataset_name, 1, n_external_fold, unweightedAuc, 1000, elapsed_time, characteristics_str, options_str_builder.toString().trim()
            ));
            System.out.println(String.format("Done: %s,%d,%d", dataset_name, 1, n_external_fold));
            opt_file.flush();
            opt_file.close();
            return true;
        } catch(Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace(System.err);
            return e;
        }
    }

    private static Object runExternalCrossValidationFoldEDNEL(
            int n_external_fold, int n_internal_folds, String datasets_path, String dataset_name,
            String dataset_experiment_path, String clf_name
    ) {
        try {

            float selection_share = 0.5f;
            int n_individuals = 100;
            int n_generations = 100;
            int timeout = 3600;
            int timeout_individual = 60;
            int burn_in = 100;
            int thinning_factor = 0;
            boolean no_cycles = false;
            float[] learning_rates = {0.13f, 0.26f, 0.52f};
            int[] early_stop_generations = {10, 20};
            int[] max_parents = {0, 1};
            int delay_structure_learning = 5;

            double[][] combinations = new double[learning_rates.length * early_stop_generations.length * max_parents.length][3];
            int counter_comb = 0;
            for(double learning_rate : learning_rates) {
                for(int early_stop_generation : early_stop_generations) {
                    for(int max_parent : max_parents) {
                        combinations[counter_comb] = new double[]{learning_rate, early_stop_generation, max_parent};
                        counter_comb += 1;
                    }
                }
            }

            HashMap<String, Instances> datasets = Main.loadDataset(
                    datasets_path,
                    dataset_name,
                    n_external_fold
            );
            Instances external_train_data = datasets.get("train_data");  // 9/10 of external cv folds
            Instances external_test_data = datasets.get("test_data");  // 1/10 of external cv folds

            // prepares data for future internal cross validation
            external_train_data = FitnessCalculator.betterStratifier(external_train_data, n_internal_folds);

            double[][] lastPredictionMatrix = new double[external_train_data.size()][];
            double[][] overallPredictionMatrix = new double[external_train_data.size()][];
            double[] actualClasses = new double[external_train_data.size()];

            int counter_combination = 0;

            int best_combination_index = -1;
            double best_combination_auc = Double.NEGATIVE_INFINITY;
            boolean bestUsesOverall = false;

            for(double[] comb : combinations) {  // iterates over combinations of hyper-parameters
                int counter_instance = 0;
                for(int i = 0; i < n_internal_folds; i++) {
                    Instances internal_train_data = external_train_data.trainCV(n_internal_folds, i);
                    Instances internal_test_data = external_train_data.testCV(n_internal_folds, i);

                    double[][] lastFoldPreds;
                    double[][] overallFoldPreds;

                    try {
                        EDNEL ednel = new EDNEL(
                                comb[0],  // learning rate
                                selection_share,
                                n_individuals,
                                n_generations,
                                timeout,
                                timeout_individual,
                                burn_in,
                                thinning_factor,
                                false,
                                (int)comb[1],  // early stop generations
                                (int)comb[2],  // max parents
                                comb[2] == 0? 0 : delay_structure_learning,
                                0, // holdout
                                null,
                                null
                        );

                        ednel.buildClassifier(internal_train_data);

                        lastFoldPreds = ednel.getCurrentGenBest().distributionsForInstances(internal_test_data);
                        overallFoldPreds = ednel.getOverallBest().distributionsForInstances(internal_test_data);
                    } catch(EmptyEnsembleException eee) {
                        lastFoldPreds = new double[internal_test_data.size()][];
                        overallFoldPreds = new double[internal_test_data.size()][];

                        int n_classes = internal_test_data.numClasses();

                        for(int j = 0; j < internal_test_data.size(); j++) {
                            lastFoldPreds[j] = new double[n_classes];
                            overallFoldPreds[j] = new double[n_classes];
                            for(int k = 0; k < n_classes; k++) {
                                lastFoldPreds[j][k] = 0.0f;
                                overallFoldPreds[j][k] = 0.0f;
                            }
                        }
                    }

                    for(int j = 0; j < internal_test_data.size(); j++) {
                        lastPredictionMatrix[counter_instance] = lastFoldPreds[j];
                        overallPredictionMatrix[counter_instance] = overallFoldPreds[j];
                        actualClasses[counter_instance] = internal_test_data.instance(j).value(internal_test_data.classIndex());
                        counter_instance += 1;
                    }
                }
                FoldJoiner lastFJ = new FoldJoiner(lastPredictionMatrix, actualClasses);
                FoldJoiner overallFJ = new FoldJoiner(overallPredictionMatrix, actualClasses);

                double lastAUC = lastFJ.getAUC("classifier");
                double overallAUC = overallFJ.getAUC("classifier");

                if((lastAUC > best_combination_auc) && (lastAUC > overallAUC)) {
                    best_combination_index = counter_combination;
                    best_combination_auc = lastAUC;
                    bestUsesOverall = false;
                } else if((overallAUC > best_combination_auc) && (overallAUC > lastAUC)) {
                    best_combination_index = counter_combination;
                    best_combination_auc = overallAUC;
                    bestUsesOverall = true;
                }
                counter_combination += 1;
            }
            PBILLogger pbilLogger = new PBILLogger(dataset_name, dataset_experiment_path,
            n_individuals, n_generations, 1, n_external_fold, true, false);

            NestedCrossValidation.EDNELparametersToFile(
                    n_external_fold,
                    dataset_experiment_path, dataset_name,
                    selection_share,
                    n_individuals,
                    n_generations,
                    timeout,
                    timeout_individual,
                    burn_in,
                    thinning_factor,
                    no_cycles,
                    (float)combinations[best_combination_index][0],  // learning rate
                    (int)combinations[best_combination_index][1],  // early stop generations
                    (int)combinations[best_combination_index][2],  // max parents
                    delay_structure_learning,
                    bestUsesOverall
                    );

            EDNEL ednel = new EDNEL(
                    combinations[best_combination_index][0],  // learning rate
                    selection_share,
                    n_individuals,
                    n_generations,
                    timeout,
                    timeout_individual,
                    burn_in,
                    thinning_factor,
                    false,
                    (int)combinations[best_combination_index][1],  // early stop generations
                    (int)combinations[best_combination_index][2],  // max parents
                    combinations[best_combination_index][2] == 0? 0 : delay_structure_learning,
                    0, // holdout
                    pbilLogger,
                    null
            );
            ednel.buildClassifier(external_train_data);

            HashMap<String, Individual> toReport = new HashMap<>(1);
            if(bestUsesOverall) {
                toReport.put("overall", ednel.getOverallBest());
            } else {
                toReport.put("last", ednel.getCurrentGenBest());
            }
            ednel.getPbilLogger().toFile(ednel.getDependencyNetwork(), toReport, external_train_data, external_test_data);
            return true;
        } catch(Exception e) {
            System.err.println(e.getMessage());
            e.printStackTrace(System.err);
            return e;
        }
    }

    private static void classifierOptimization(
            int n_external_folds, int n_internal_folds,
            String datasets_path, String dataset_name, String experiment_metadata_path, String clf_name
    ) throws Exception {

        if(clf_name.equals("EDNEL")) {
            Object[] answers = IntStream.range(1, n_external_folds + 1).parallel().mapToObj(
                    i -> NestedCrossValidation.runExternalCrossValidationFoldEDNEL(
                            i, n_internal_folds, datasets_path, dataset_name,
                            experiment_metadata_path + File.separator + dataset_name, clf_name)
            ).toArray();
        } else if(clf_name.equals("RandomForest")) {
            Object[] answers = IntStream.range(1, n_external_folds + 1).parallel().mapToObj(
                    i -> NestedCrossValidation.runExternalCrossValidationFoldRandomForest(
                            i, n_internal_folds, datasets_path, dataset_name,
                            experiment_metadata_path + File.separator + dataset_name, clf_name)
            ).toArray();
        } else {
            throw new Exception("not implemented yet!");
        }
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = NestedCrossValidation.parseCommandLine(args);

        HashMap<String, String> options = new HashMap<>();
        for(Option opt : commandLine.getOptions()) {
            options.put(opt.getLongOpt(), commandLine.getOptionValue(opt.getLongOpt()));
        }

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);
        options.put("datasets_names", options.get("dataset_name"));
        PBILLogger.metadata_path_start(str_time, options);
        options.remove("datasets_names");

        String experiment_metadata_path = options.get("metadata_path") + File.separator + str_time;

        String datasets_path = options.get("datasets_path");
        String dataset_name = options.get("dataset_name");
        int n_external_folds = Integer.parseInt(options.get("n_external_folds"));
        int n_internal_folds = Integer.parseInt(options.get("n_internal_folds"));
        String clf_name = options.get("classifier");

        NestedCrossValidation.classifierOptimization(
                n_external_folds, n_internal_folds, datasets_path, dataset_name, experiment_metadata_path, clf_name
        );
    }
}
