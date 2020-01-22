package eda;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.Option.Builder;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;
import org.json.simple.JSONObject;

import javax.annotation.processing.FilerException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.FileReader;

public class EDNEL {

    private float learning_rate;
    private float selection_share;
    private int n_individuals;
    private int n_generations;
    private String variables_path;
    private String output_path;

    private boolean fitted;

    public EDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations,
                 String variables_path, String output_path) throws Exception {

        this.learning_rate = learning_rate;
        this.selection_share = selection_share;
        this.n_individuals = n_individuals;
        this.n_generations = n_generations;
        this.variables_path = variables_path;
        this.output_path = output_path;
        this.fitted = false;



    }

    private void readVariables(String variables_path) {

    }

    public void fit(Instances train_data) {


        this.fitted = true;
    }

    public String[] predict(Instances data) {
        return null;
    }

    private static void createFolder(String path) throws FilerException {
        File file = new File(path);
        boolean successful = file.mkdir();
        if(!successful) {
            throw new FilerException("could not create directory " + path);
        }
    }

    public static void metadata_path_start(String str_time, CommandLine commandLine) throws ParseException, IOException {
        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        String metadata_path = commandLine.getOptionValue("metadata_path");

        // create one folder for each dataset
        EDNEL.createFolder(metadata_path + File.separator + str_time);
        for(String dataset : dataset_names) {
            EDNEL.createFolder(metadata_path + File.separator + str_time + File.separator + dataset);
        }

        JSONObject obj = new JSONObject();
        for(Option parameter : commandLine.getOptions()) {
            obj.put(parameter.getLongOpt(), parameter.getValue());
        }

        FileWriter fw = new FileWriter(metadata_path + File.separator + str_time + File.separator + "parameters.json");
        fw.write(obj.toJSONString());
        fw.flush();

    }

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
                .longOpt("seed")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Seed used to initialize base classifiers (i.e. Weka-related). It is not used to bias PBIL.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_jobs")
                .required(false)
                .type(Integer.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Number of jobs to use. Will use one job per sample per fold. If unspecified or set to 1, will run in a single core.")
                .build());

        options.addOption(Option.builder()
                .longOpt("cheat")
                .required(false)
                .type(Boolean.class)
                .desc("Whether to log test metadata during evolution.")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_generations")
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Maximum number of generations to run the algorithm")
                .build());

        options.addOption(Option.builder()
                .longOpt("n_individuals")
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
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Learning rate of PBIL")
                .build());

        options.addOption(Option.builder()
                .longOpt("selection_share")
                .required(true)
                .hasArg()
                .numberOfArgs(1)
                .desc("Fraction of fittest population to use to update graphical model")
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

                for(int i = 0; i < n_samples; i++) {
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
                                commandLine.getOptionValue("variables_path"),
                                commandLine.getOptionValue("metadata_path") + File.separator +
                                str_time + dataset_name
                        );
                        ednel.fit(train_data);
                    }
                }
            }

        } catch (ParseException exception) {
            System.out.print("Parse error: ");
            System.out.println(exception.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

