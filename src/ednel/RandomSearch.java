/**
 * Randomly search hyper-parameters for a given classifier. Outputs results to file.
 */

package ednel;

import ednel.eda.individual.FitnessCalculator;
import ednel.network.DependencyNetwork;
import ednel.network.variables.AbstractVariable;
import org.apache.commons.cli.*;
import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
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

    public static AbstractClassifier getClassifier(String name) throws Exception {
        String header;
        if(name.equals("J48") || name.equals("SimpleCart") || name.equals("RandomForest")) {
            header = "weka.classifiers.trees";
        } else if(name.equals("JRip") || name.equals("PART") || name.equals("DecisionTable")) {
            header = "weka.classifiers.rules";
        } else {
            throw new Exception("unrecognized classifier name: " + name);
        }
        String lookup = header + "." + name;

        Class c1 = Class.forName(lookup);
//        ClassLoader classLoader = c1.getClassLoader();
//        Class clf = classLoader.loadClass(lookup);
        AbstractClassifier clf = (AbstractClassifier)c1.getConstructor().newInstance();
        return clf;
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = RandomSearch.parseCommandLine(args);

        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        String clfSimpleName = commandLine.getOptionValue("classifier");
        int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));
        int n_draws = Integer.parseInt(commandLine.getOptionValue("n_draws"));

        DependencyNetwork dn = new DependencyNetwork(
                new MersenneTwister(), 0, 0, false, 0.0,
                0, 0, 60
        );

        HashMap<String, AbstractVariable> variables = dn.getVariables();
        HashMap<String, AbstractVariable> relevantVariables = new HashMap<>();
        ArrayList<String> samplingOrder = dn.getSamplingOrder();
        ArrayList<String> relevantSamplingOrder = new ArrayList<>();
        for(String var : samplingOrder) {
            if(var.contains(clfSimpleName)) {
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

        ArrayList<String> lines = new ArrayList<>();
        lines.add("dataset_name,n_sample,n_fold,n_draw,unweighted_area_under_roc," +
                "elapsed time (seconds),characteristics");

        for(String dataset_name : dataset_names) {
            System.out.println("On dataset " + dataset_name);

            for(int n_fold = 1; n_fold <= 10; n_fold++) {  // 10 folds
                System.out.println("\tFold " + n_fold);

                HashMap<String, Instances> datasets = Main.loadDataset(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name,
                        n_fold
                );
                Instances train_data = datasets.get("train_data");
                Instances test_data = datasets.get("test_data");

                for(int n_draw = 1; n_draw <= n_draws; n_draw++) {
                    System.out.println("\t\tDraw " + n_draw);

                    for(int n_sample = 1; n_sample <= n_samples; n_sample++) {
                        System.out.println("\t\t\tSample " + n_sample);
                        LocalDateTime t1 = LocalDateTime.now();

                        Evaluation ev = new Evaluation(train_data);

                        AbstractClassifier clf = RandomSearch.getClassifier(clfSimpleName);
                        clf.setOptions(optionTables.get(n_draw - 1).get(clfSimpleName).split(" "));
                        clf.buildClassifier(train_data);
                        ev.evaluateModel(clf, test_data);

                        double unweightedAuc = FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, clf);
                        LocalDateTime t2 = LocalDateTime.now();

                        long seconds = t1.until(t2, ChronoUnit.SECONDS);

                        lines.add(String.format(
                                Locale.US,
                                "%s,%d,%d,%d,%f,%d,%s",
                                dataset_name, n_sample, n_fold, n_draw, unweightedAuc, seconds, characteristicsString.get(n_draw - 1)
                        ));
                    }
                }
            }
        }
        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File(
                commandLine.getOptionValue("metadata_path") + File.separator + clfSimpleName + "_randomOptimization.csv")
            )
        );

        for(String line : lines) {
            bfw.write(line + "\n");
        }
        bfw.close();
    }
}
