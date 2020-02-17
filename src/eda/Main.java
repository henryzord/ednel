package eda;

import eda.ednel.EDNEL;
import eda.individual.FitnessCalculator;
import org.apache.commons.cli.*;
import org.apache.commons.math3.genetics.Fitness;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Main {
    public static void main(String[] args) {
        Options options = new Options();

        options.addOption(Option.builder()
                .required(true)
                .longOpt("datasets_path")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Must lead to a path that contains several subpaths, one for each dataset. Each subpath, in turn, must have the arff files.")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("datasets_names")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of the datasets to run experiments. Must be a list separated by a comma\n\tExample:\n\tpython script.py datasets-names iris,mushroom,adult")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("classifiers_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to file with classifiers and their hyper-parameters.")
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
                .desc("Seed used to initialize base classifiers (i.e. Weka-related). It is not used to bias PBIL.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_jobs")
                .type(Integer.class)
                .required(false)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of jobs to use. Will use one job per sample per fold. If unspecified or set to 1, will run in a single core.")
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
                .longOpt("thining_factor")
                .type(Integer.class)
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Thining factor used in the dependency network (i.e. interval to select samples)")
                .build());

        CommandLineParser parser = new DefaultParser();

        try {
            CommandLine commandLine = parser.parse(options, args);
            int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));

            // writes metadata
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd-MM-yyyy-HH:mm:ss");
            LocalDateTime now = LocalDateTime.now();
            String str_time = dtf.format(now);
            EDNEL.metadata_path_start(str_time, commandLine);

            for(String dataset_name : commandLine.getOptionValue("datasets_names").split(",")) {
                System.out.println(String.format("On dataset %s:", dataset_name));
                double meanOverallAUC = 0, meanLastAUC = 0;
                double overallAUC, lastAUC;
                for(int i = 0; i < n_samples; i++) {
                    overallAUC = 0;
                    lastAUC = 0;
                    for(int j = 1; j < 11; j++) {  // 10 folds
                        ConverterUtils.DataSource train_set = new ConverterUtils.DataSource(
                                commandLine.getOptionValue("datasets_path") + File.separator +
                                        dataset_name + File.separator + dataset_name + "-10-" + j + "tra.arff"
                        );
                        ConverterUtils.DataSource test_set = new ConverterUtils.DataSource(
                                commandLine.getOptionValue("datasets_path") + File.separator +
                                        dataset_name + File.separator + dataset_name + "-10-" + j + "tst.arff"
                        );

                        Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
                        train_data.setClassIndex(train_data.numAttributes() - 1);
                        test_data.setClassIndex(test_data.numAttributes() - 1);

                        EDNEL ednel = new EDNEL(
                                Float.parseFloat(commandLine.getOptionValue("learning_rate")),
                                Float.parseFloat(commandLine.getOptionValue("selection_share")),
                                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                                Integer.parseInt(commandLine.getOptionValue("thining_factor")),
                                commandLine.getOptionValue("variables_path"),
                                commandLine.getOptionValue("options_path"),
                                commandLine.getOptionValue("sampling_order_path"),
                                commandLine.getOptionValue("metadata_path") + File.separator +
                                        str_time + dataset_name,
                                commandLine.getOptionValue("seed") == null? null : Integer.parseInt(commandLine.getOptionValue("seed"))
                        );

                        ednel.buildClassifier(train_data);
                        Evaluation overallEval = new Evaluation(train_data);
                        Evaluation lastEval = new Evaluation(train_data);
                        overallEval.evaluateModel(ednel.getOverallBest(), test_data);
                        lastEval.evaluateModel(ednel.getCurrentGenBest(), test_data);

                        overallAUC += FitnessCalculator.getUnweightedAreaUnderROC(overallEval);
                        lastAUC += FitnessCalculator.getUnweightedAreaUnderROC(lastEval);
                    }
                    overallAUC /= 10;
                    lastAUC /= 10;

                    meanOverallAUC += overallAUC;
                    meanLastAUC += lastAUC;

                    System.out.println(String.format("Partial results for sample %d on dataset %s:", i, dataset_name));
                    System.out.println(String.format("\tOverall: %.8f\t\tLast: %.8f", overallAUC, lastAUC));
                }
                meanLastAUC /= n_samples;
                meanOverallAUC /= n_samples;
                System.out.println(String.format("Average of %d samples of 10-fcv on dataset %s:", n_samples, dataset_name));
                System.out.println(String.format("\tOverall: %.8f\t\tLast: %.8f", meanOverallAUC, meanLastAUC));
            }

        } catch (ParseException exception) {
            System.out.print("Parse error: ");
            System.out.println(exception.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}