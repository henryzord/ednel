package eda;

import eda.ednel.EDNEL;
import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import org.apache.commons.cli.*;
import utils.PBILLogger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

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
                .desc("Maximum tolerance between two generations that do not improve in the best individual fitness. Higher values are less tolerant.")
                .build());

        options.addOption(Option.builder()
            .longOpt("nearest_neighbor")
            .type(Integer.class)
            .required(true)
            .hasArg()
            .numberOfArgs(1)
            .desc("Number of nearest neighbors to consider when calculating mutual information between continuous and discrete variables pairs.")
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

        try {
            CommandLine commandLine = parser.parse(options, args);
            int n_samples = Integer.parseInt(commandLine.getOptionValue("n_samples"));

            // writes metadata
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("dd-MM-yyyy-HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            String str_time = dtf.format(now);
            PBILLogger.metadata_path_start(str_time, commandLine);

            for(String dataset_name : commandLine.getOptionValue("datasets_names").split(",")) {
                System.out.println(String.format("On dataset %s:", dataset_name));
                double meanOverallAUC = 0, meanLastAUC = 0;
                double overallAUC, lastAUC;
                for(int i = 1; i < n_samples + 1; i++) {
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

                        PBILLogger pbilLogger = new PBILLogger(
                                commandLine.getOptionValue("metadata_path") + File.separator +
                                        str_time + File.separator + dataset_name,
                                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                                i, j,
                                Boolean.parseBoolean(commandLine.getOptionValue("log"))

                        );

                        EDNEL ednel = new EDNEL(
                                Float.parseFloat(commandLine.getOptionValue("learning_rate")),
                                Float.parseFloat(commandLine.getOptionValue("selection_share")),
                                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                                Integer.parseInt(commandLine.getOptionValue("burn_in")),
                                Integer.parseInt(commandLine.getOptionValue("thinning_factor")),
                                Integer.parseInt(commandLine.getOptionValue("early_stop_generations")),
                                Float.parseFloat(commandLine.getOptionValue("early_stop_tolerance")),
                                Integer.parseInt(commandLine.getOptionValue("nearest_neighbor")),
                                commandLine.getOptionValue("variables_path"),
                                commandLine.getOptionValue("options_path"),
                                pbilLogger,
                                commandLine.getOptionValue("seed") == null?
                                    null : Integer.parseInt(commandLine.getOptionValue("seed"))
                        );

                        ednel.buildClassifier(train_data);

                        HashMap<String, Individual> toReport = new HashMap<>(2);
                        toReport.put("overall", ednel.getOverallBest());
                        toReport.put("last", ednel.getCurrentGenBest());

                        ednel.getPbilLogger().toFile(ednel.getDependencyNetwork(), toReport, train_data, test_data);

                        overallAUC += FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getOverallBest()) / 10;
                        lastAUC += FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getCurrentGenBest()) / 10;
                    }
                    meanOverallAUC += overallAUC / n_samples;
                    meanLastAUC += lastAUC / n_samples;

                    System.out.println(String.format("Partial results for sample %d on dataset %s:", i, dataset_name));
                    System.out.println(String.format("\tOverall: %.8f\t\tLast: %.8f", overallAUC, lastAUC));
                }
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
