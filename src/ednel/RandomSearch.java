/*
  Randomly samples hyper-parameters for one of the following classifiers: J48, SimpleCart, PART, JRip, DecisionTable,
  RandomForest. Performs a 10-fold cross validation on the provided datasets with the sampled hyper-parameters. Outputs
  a csv to a specified folder with the results of the experiments.

  J48, SimpleCart, PART, JRip, and DecisionTable have all their hyper-parameters uniformly sampled from the Dependency
  Network used by EDNEL; that is to say that only variables specified in the probability tables of EDNEL are considered
  as hyper-parameters of this class.

  RandomForest is not a part of EDNEL, and hence the following hyper-parameters are sampled:

  * bagSizePercent - % of training instances considered for each decision tree. Same range of values used in
  "Hyperparameters and tuning strategies for random forest".
  * breakTiesRandomly - on the occurrence of two attributes being equally good for splitting data in a given node in a
    decision tree, should we break the tie simply using the first (alphabetically sorted) variable, or randomly decide?
  * maxDepth - maximum depth allowed for ensemble trees (0 = unlimited)
  * numFeatures -- Sets the number of randomly chosen attributes. If 0, int(log_2(predictive_attrs) + 1) is used.

  NOTE: numIterations (i.e. the number of trees in the ensemble) is not randomly optimized. Instead, a high number (1000)
  is used, because more trees in the ensemble always equal to better results.
 */

package ednel;

import ednel.eda.individual.FitnessCalculator;
import ednel.network.DependencyNetwork;
import ednel.network.variables.AbstractVariable;
import org.apache.commons.cli.*;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;

public class RandomSearch {
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
                .longOpt("classifier")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of classifier to randomly search hyper-parameters.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_draws")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of hyper-parameterizations to draw.")
                .build());

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    /**
     * Given the simple name of a classifier, gets the whole path to it, for reflection use.
     * @param name The simple name of the classifier (e.g. J48, PART)
     * @return A path to the class (e.g. weka.classifiers.trees.J48, weka.classifiers.rules.PART)
     * @throws Exception if is not possible to find the required class
     */
    public static AbstractClassifier getClassifier(String name) throws Exception {
        String header;
        if(name.equals("J48")) {
            header = "weka.classifiers.trees";
        } else if(name.equals("SimpleCart")) {
            header = "ednel.classifiers.trees";
        } else if(name.equals("JRip") || name.equals("PART") || name.equals("DecisionTable")) {
            header = "weka.classifiers.rules";
        } else {
            throw new Exception("unrecognized classifier name: " + name);
        }
        String lookup = header + "." + name;

        Class c1 = Class.forName(lookup);
        AbstractClassifier clf = (AbstractClassifier)c1.getConstructor().newInstance();
        return clf;
    }

    private static void EDNELClassifierOptimization(int n_samples, int n_draws, String clfSimpleName,
                                                    String[] datasets_names, String datasets_path, String metadata_path
    ) throws Exception {

        DependencyNetwork dn = new DependencyNetwork(
                new MersenneTwister(), 0, 0, false, 0.0,
                0, 0, 60
        );

        HashMap<String, AbstractVariable> variables = dn.getVariables();
        HashMap<String, AbstractVariable> relevantVariables = new HashMap<>();
        ArrayList<String> samplingOrder = dn.getSamplingOrder();
        ArrayList<String> relevantSamplingOrder = new ArrayList<>();

        for(String var : samplingOrder) {
            if(var.contains(clfSimpleName) ||
                    (clfSimpleName.equals("DecisionTable") && (var.contains("BestFirst") || var.contains("GreedyStepwise")))
            ) {
                relevantVariables.put(var, variables.get(var));
                relevantSamplingOrder.add(var);
            }
        }
        relevantVariables.remove(clfSimpleName);
        relevantSamplingOrder.remove(clfSimpleName);

        ArrayList<HashMap<String, String>> draws = new ArrayList<>();
        ArrayList<HashMap<String, String>> optionTables = new ArrayList<>();
        ArrayList<String> characteristicsString = new ArrayList<>();

        while(draws.size() < n_draws) {
            HashMap<String, String> thisDraw = new HashMap<>();
            thisDraw.put(clfSimpleName, "true");
            HashMap<String, String> optionTable = new HashMap<>();
            StringBuilder strChar = new StringBuilder();

            for(String var : relevantSamplingOrder) {
                String sampled = relevantVariables.get(var).unconditionalUniformSampling(thisDraw);
                if(!String.valueOf(sampled).equals("null")) {
                    optionTable = dn.getOptionHandler().handle(optionTable, var, clfSimpleName, sampled);
                }
                thisDraw.put(var, sampled);
                strChar.append(String.format("%s=%s;", var, sampled));
            }

            try {
                AbstractClassifier clf = RandomSearch.getClassifier(clfSimpleName);
                clf.setOptions(optionTable.get(clfSimpleName).split(" "));
                // succeeded; registers option table and characteristics of this sample
                optionTables.add(optionTable);
                draws.add(thisDraw);
                characteristicsString.add(strChar.toString());
            } catch(Exception e) {
                // does nothing
            }
        }

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);

        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File(
                metadata_path + File.separator + str_time + "_" + clfSimpleName + "_randomOptimization.csv")
        ));

        Exception except = null;

        try {
            bfw.write("classifier,dataset_name,n_sample,n_fold,n_draw,unweighted_area_under_roc," +
                    "elapsed time (seconds),characteristics,options\n");

            for(String dataset_name : datasets_names) {
                for(int n_fold = 1; n_fold <= 10; n_fold++) {  // 10 folds
                    HashMap<String, Instances> datasets = Main.loadDataset(
                            datasets_path,
                            dataset_name,
                            n_fold
                    );
                    Instances train_data = datasets.get("train_data");
                    Instances test_data = datasets.get("test_data");

                    for(int n_draw = 1; n_draw <= n_draws; n_draw++) {
                        for(int n_sample = 1; n_sample <= n_samples; n_sample++) {
                            try {
                                LocalDateTime t1 = LocalDateTime.now();

                                Evaluation ev = new Evaluation(train_data);

                                AbstractClassifier clf = RandomSearch.getClassifier(clfSimpleName);
                                clf.setOptions(optionTables.get(n_draw - 1).get(clfSimpleName).split(" "));
                                clf.buildClassifier(train_data);
                                ev.evaluateModel(clf, test_data);

                                double unweightedAuc = FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, clf);
                                LocalDateTime t2 = LocalDateTime.now();

                                long seconds = t1.until(t2, ChronoUnit.SECONDS);

                                bfw.write(String.format(
                                        Locale.US,
                                        "%s,%s,%d,%d,%d,%f,%d,\"%s\",\"%s\"\n",
                                        clfSimpleName, dataset_name, n_sample, n_fold, n_draw, unweightedAuc, seconds, characteristicsString.get(n_draw - 1),
                                        optionTables.get(n_draw - 1).get(clfSimpleName)
                                ));
                                System.out.println(String.format("Done: %s,%03d,%03d,%03d", dataset_name, n_fold, n_draw, n_sample));
                            } catch(Exception e) {
                                System.err.println(String.format("Error: %s,%03d,%03d,%03d", dataset_name, n_fold, n_draw, n_sample));
                            }
                        }
                    }
                }
            }
        } catch(Exception e) {
            except = e;
        } finally {
            bfw.close();
        }
        if(except != null) {
            throw except;
        }
    }

    public static String replaceNumFeaturesParameterRandomForest(String treatedOptions, int numPredictiveAttr) {
        String numFeaturesMultiplier = treatedOptions.substring(
                treatedOptions.indexOf("<numFeaturesMultiplier>") + "<numFeaturesMultiplier>".length(),
                treatedOptions.indexOf("</numFeaturesMultiplier>")
        );
        int numFeatures = (int)Math.ceil(Float.parseFloat(numFeaturesMultiplier) * numPredictiveAttr);
        treatedOptions = treatedOptions.replace(
                String.format("<numFeaturesMultiplier>%s</numFeaturesMultiplier>", numFeaturesMultiplier),
                String.valueOf(numFeatures)
        );
        return treatedOptions;
    }

    private static void randomForestOptimization(int n_samples, int n_draws, String[] datasets_names,
                                                 String datasets_path, String metadata_path
    ) throws Exception {

        MersenneTwister mt = new MersenneTwister();

        String[] samplingOrder = new String[]{"bagSizePercent", "breakTiesRandomly", "maxDepth", "numFeatures", "numIterations"};
        String[] optionNames = new String[]{"-P", ";-B", "-depth", "-K", "-I"};
        int[][] ranges = new int[][]{{20, 90}, {0, 1}, {0, 10}, {1, 100}, {1000, 1000}};

        HashMap<String, Integer[]> variables = new HashMap<>();
        for(int i = 0; i < samplingOrder.length; i++) {
            String var = samplingOrder[i];
            int low = ranges[i][0];
            int high = ranges[i][1] + 1;

            Integer[] values = new Integer[high - low];
            int counter = 0;
            for(int j = low; j < high; j++) {
                values[counter] = j;
                counter += 1;
            }
            variables.put(var, values);
        }

        ArrayList<HashMap<String, String>> draws = new ArrayList<>();
        ArrayList<String> optionTables = new ArrayList<>();
        ArrayList<String> characteristicsString = new ArrayList<>();

        while(draws.size() < n_draws) {
            HashMap<String, String> thisDraw = new HashMap<>();
            StringBuilder strChar = new StringBuilder();
            StringBuilder strOpt = new StringBuilder();

            for(int j = 0; j < samplingOrder.length; j++) {
                String var = samplingOrder[j];

                Integer[] values = variables.get(var);
                int index = mt.nextInt(values.length);
                String sampled = String.valueOf(values[index]);

                if(var.equals("breakTiesRandomly")) {
                    strOpt.append(sampled.equals("1")? "-B " : "");
                    strChar.append(String.format("%s=%s;", var, sampled.equals("1")? "true" : "false"));
                } else if (var.equals("maxDepth") && sampled.equals("0")) {
                    // if maxDepth is unlimited (i.e. 0), should not set flag
                    strChar.append(String.format("%s=%s;", var, sampled));
                } else if(var.equals("numFeatures")) {
                    strOpt.append(optionNames[j]).append(" ").append(String.format(
                            "<numFeaturesMultiplier>0.%s</numFeaturesMultiplier>", sampled)).append(" ");
                    strChar.append(String.format("%s=%s * predictiveAttributes;", var, Integer.parseInt(sampled)/100.));
                } else {
                    strOpt.append(optionNames[j]).append(" ").append(sampled).append(" ");
                    strChar.append(String.format("%s=%s;", var, sampled));
                }
                thisDraw.put(var, sampled);
            }

            optionTables.add(strOpt.toString());
            draws.add(thisDraw);
            characteristicsString.add(strChar.toString());
        }

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);

        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File(
                metadata_path + File.separator + str_time + "_RandomForest_randomOptimization.csv")
        ));

        Exception except = null;

        try {
            bfw.write("classifier,dataset_name,n_sample,n_fold,n_draw,unweighted_area_under_roc," +
                    "elapsed time (seconds),characteristics,options\n");

            for(String dataset_name : datasets_names) {
                for(int n_fold = 1; n_fold <= 10; n_fold++) {  // 10 folds
                    HashMap<String, Instances> datasets = Main.loadDataset(
                            datasets_path,
                            dataset_name,
                            n_fold
                    );
                    Instances train_data = datasets.get("train_data");
                    Instances test_data = datasets.get("test_data");

                    for(int n_draw = 1; n_draw <= n_draws; n_draw++) {
                        for(int n_sample = 1; n_sample <= n_samples; n_sample++) {
                            try {
                                LocalDateTime t1 = LocalDateTime.now();

                                Evaluation ev = new Evaluation(train_data);

                                RandomForest rf = new RandomForest();
                                String treatedOptions = optionTables.get(n_draw - 1);
                                treatedOptions = RandomSearch.replaceNumFeaturesParameterRandomForest(
                                        treatedOptions, train_data.numAttributes() - 1
                                );

                                rf.setOptions(treatedOptions.split(" "));
                                rf.buildClassifier(train_data);
                                ev.evaluateModel(rf, test_data);

                                double unweightedAuc = FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, rf);
                                LocalDateTime t2 = LocalDateTime.now();

                                long seconds = t1.until(t2, ChronoUnit.SECONDS);

                                bfw.write(String.format(
                                        Locale.US,
                                        "%s,%s,%d,%d,%d,%f,%d,\"%s\",\"%s\"\n",
                                        "RandomForest", dataset_name, n_sample, n_fold, n_draw, unweightedAuc, seconds, characteristicsString.get(n_draw - 1),
                                        optionTables.get(n_draw - 1)
                                ));
                                System.out.println(String.format("Done: %s,%03d,%03d,%03d", dataset_name, n_fold, n_draw, n_sample));
                            } catch(Exception e) {
                                System.err.println(String.format("Error: %s,%03d,%03d,%03d", dataset_name, n_fold, n_draw, n_sample));
                            }
                        }
                    }
                }
            }
        } catch(Exception e) {
            except = e;
        } finally {
            bfw.close();
        }
        if(except != null) {
            throw except;
        }
    }


    public static void main(String[] args) throws Exception {
        CommandLine commandLine = RandomSearch.parseCommandLine(args);

        String[] datasets_names = commandLine.getOptionValue("datasets_names").split(",");
        int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));
        int n_draws = Integer.parseInt(commandLine.getOptionValue("n_draws"));
        String clfSimpleName = commandLine.getOptionValue("classifier");
        String metadata_path = commandLine.getOptionValue("metadata_path");
        String datasets_path = commandLine.getOptionValue("datasets_path");

        if(clfSimpleName.equals("RandomForest")) {
            RandomSearch.randomForestOptimization(n_samples, n_draws, datasets_names, datasets_path, metadata_path);
        } else {
            RandomSearch.EDNELClassifierOptimization(n_samples, n_draws, clfSimpleName, datasets_names, datasets_path, metadata_path);
        }
    }
}
