/**
 * Applies the result of a random search to a group of datasets.
 *
 * Syntax and examples:
 *
 * If using only J48, enter the following command in commandline:
 *
 * java -classpath ednel.jar ednel.utils.analysis.optimizers.RandomSearchApply --datasets_path <datasets_path>
 *   --datasets_names <comma_separated_datasets> --metadata_path <metadata_path>
 *   --string_options "-J48 -U -O -M 8 -A" --string_characteristics "J48_pruning=unpruned;J48_collapseTree=false;J48_subtreeRaising=null;J48_binarySplits=false;J48_minNumObj=8;J48_useLaplace=true;J48_useMDLcorrection=true;J48_confidenceFactorValue=null;J48_doNotMakeSplitPointActualValue=false;J48_numFolds=null;"
 *   --n_samples 10
 *
 * If using all ensemble members, type the following:
 *
 * -J48 -C 0.25 -M 2 -SimpleCart -M 2 -N 5 -C 1 -S 1 -PART -M 2 -C 0.25 -Q 1 -JRip -F 3 -N 2.0 -O 2 -S 1
 * -DecisionTable -R -X 1 -S weka.attributeSelection.BestFirst -D 1 -N 5
 */

package ednel.utils.analysis.optimizers;

import ednel.Main;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import ednel.utils.sorters.Argsorter;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;

public class RandomSearchApply {
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

        options.addOption(Option.builder()
                .longOpt("metadata_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where runs results will be stored.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_samples")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of samples to average.")
                .build());

        options.addOption(Option.builder()
                .longOpt("string_characteristics")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("A string of all the hyper-parameters of classifiers in the ensemble.")
                .build());

        options.addOption(Option.builder()
                .longOpt("string_options")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("A string of all the options for classifiers in the ensemble.")
                .build());

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    public static HashMap<String, String> fromCharacteristicsStringToHashMap(String c) {
        HashMap<String, String> characteristics = new HashMap<>();

        String[] components = c.split(";");
        for(String component : components) {
            if(component.length() > 0) {
                String[] atom = component.split("=");
                String val = atom[1].toString().equals("null")? null : atom[1];
                characteristics.put(atom[0], val);
            }
        }

        if(!characteristics.containsKey("Aggregator") && characteristics.containsKey("RandomForest")) {
            characteristics.put("Aggregator", "MajorityVotingAggregator");
        }

        return characteristics;
    }

    private static HashMap<String, String> captureEDNELClassifierOptions(String c) {
        String[] tokens = {"-J48", "-SimpleCart", "-PART", "-JRip", "-DecisionTable", "-GreedyStepwise", "-BestFirst", "-Aggregator"};
        HashMap<String, String> optionsTable = new HashMap<>();
        Integer[] indices = new Integer [tokens.length];

        for(int i = 0; i < tokens.length; i++) {
            indices[i] = c.indexOf(tokens[i]);
        }
        Integer[] sortedIndices = Argsorter.crescent_argsort(indices);

        for(int i = 0; i < tokens.length; i++) {
            if(indices[sortedIndices[i]] == -1) {
                optionsTable.put(tokens[sortedIndices[i]].replace("-", ""), null);
                continue;
            }

            int startIndex = indices[sortedIndices[i]] + tokens[sortedIndices[i]].length();
            int endIndex;

            if((i + 1) < tokens.length) {
                endIndex = indices[sortedIndices[i + 1]] - 1;
            } else {
                endIndex = c.length();
            }
            String value = c.substring(startIndex, endIndex).trim();
            optionsTable.put(tokens[sortedIndices[i]].replace("-", ""), value);
        }

        if(
                !optionsTable.containsKey("Aggregator") ||
                (optionsTable.get("Aggregator").length() == 0) ||
                (optionsTable.get("Aggregator") == null)
        ) {
            throw new IllegalArgumentException("should pass an Aggregator too!");
        }
        return optionsTable;
    }

    public static HashMap<String, String> fromOptionsStringToHashMap(String c) {
        // test string (equals to default classifiers hyper-parameters):
        // -J48 -C 0.25 -M 2 -SimpleCart -M 2 -N 5 -C 1 -S 1 -PART -M 2 -C 0.25 -Q 1 -JRip -F 3 -N 2.0 -O 2 -S 1
        // -DecisionTable -R -X 1 -S weka.attributeSelection.BestFirst -D 1 -N 5

        if(!c.contains("RandomForest")) {
            return captureEDNELClassifierOptions(c);
        } else {
            return captureRandomForestOptions(c);
        }


    }

    private static HashMap<String, String> captureRandomForestOptions(String c) {
        HashMap<String, String> optionsTable = new HashMap<>();
        optionsTable.put("RandomForest", c.substring(c.indexOf("RandomForest") + "RandomForest".length()).trim());
        return optionsTable;
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = RandomSearchApply.parseCommandLine(args);

        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));
        String characteristics_str = commandLine.getOptionValue("string_characteristics");
        String options_str = commandLine.getOptionValue("string_options");
        HashMap<String, String> optionTable = RandomSearchApply.fromOptionsStringToHashMap(options_str);
        HashMap<String, String> characteristics = RandomSearchApply.fromCharacteristicsStringToHashMap(characteristics_str);

        boolean isRF = optionTable.containsKey("RandomForest");

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);

        File metadata_file = new File(commandLine.getOptionValue("metadata_path") + File.separator + str_time);
        metadata_file.mkdirs();

//        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File(
//                commandLine.getOptionValue("metadata_path") + File.separator + str_time + "_applied_algorithm.csv")
//        ));
        Exception except = null;

        try {
//            bfw.write("dataset_name,n_sample,n_fold,unweighted_area_under_roc,classifier size,elapsed time (seconds),characteristics,options\n");

            for(String dataset_name : dataset_names) {
                File dataset_folder_file = new File(metadata_file.getAbsolutePath() + File.separator + dataset_name);
                dataset_folder_file.mkdirs();
                dataset_folder_file = new File(dataset_folder_file.getAbsolutePath() + File.separator + "overall");
                dataset_folder_file.mkdirs();

                for(int n_fold = 1; n_fold <= 10; n_fold++) {  // 10 folds
                    HashMap<String, Instances> datasets = Main.loadDataset(
                            commandLine.getOptionValue("datasets_path"),
                            dataset_name,
                            n_fold
                    );
                    Instances train_data = datasets.get("train_data");
                    Instances test_data = datasets.get("test_data");

                    for(int n_sample = 1; n_sample <= n_samples; n_sample++) {
                        try {
                            File preds_file = new File(
                                    String.format(
                                        dataset_folder_file.getAbsolutePath() + File.separator +
                                                "test_sample-%02d_fold-%02d_%s.preds",
                                            n_sample, n_fold, isRF? "RandomForest" : "EDNEL"
                                )
                            );

                            LocalDateTime t1 = LocalDateTime.now();
                            AbstractClassifier ind;
                            if(isRF) {
                                ind = new RandomForest();
                                String treatedOptions = optionTable.get("RandomForest");
                                treatedOptions = RandomSearch.replaceNumFeaturesParameterRandomForest(
                                        treatedOptions, train_data.numAttributes() - 1
                                );
                                ind.setOptions(treatedOptions.split(" "));
                            } else {
                                ind = new Individual(optionTable, characteristics);
                            }
//                            ind.buildClassifier(train_data);
//                            Evaluation ev = new Evaluation(train_data);
//
//                            ev.evaluateModel(ind, test_data);
//                            double unweightedAuc = FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ind);
                            LocalDateTime t2 = LocalDateTime.now();

                            long elapsed_time = t1.until(t2, ChronoUnit.SECONDS);

                            PBILLogger.train_and_write_predictions_to_file(new AbstractClassifier[]{ind}, train_data, test_data, preds_file.getAbsolutePath());

//                            bfw.write(String.format(
//                                    Locale.US,
//                                    "%s,%d,%d,%f,%d,%d,\"%s\",\"%s\"\n",
//                                    dataset_name, n_sample, n_fold, unweightedAuc, ind instanceof Individual? ((Individual)ind).getNumberOfRules() : -1, elapsed_time, characteristics_str, options_str
//                            ));
                            System.out.println(String.format("Done: %s,%d,%d", dataset_name, n_sample, n_fold));
                        } catch(Exception e) {
                            System.err.println(String.format("Error: %s,%d,%d,%s", dataset_name, n_sample, n_fold, e.getMessage()));
                        }
                    }
                }
            }
        } catch(Exception e) {
            except = e;
        } finally {
//            bfw.close();
        }
        if(except != null) {
            throw except;
        }
    }
}
