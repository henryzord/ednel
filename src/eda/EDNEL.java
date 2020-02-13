package eda;

import dn.DependencyNetwork;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.Option.Builder;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;

import javax.annotation.processing.FilerException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.format.DateTimeFormatter;
import java.time.LocalDateTime;

import org.omg.CORBA.INTERNAL;
import utils.ArrayIndexComparator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;

public class EDNEL {

    private int thining_factor;
    private String options_path;
    private String variables_path;
    private String sampling_order_path;
    private float learning_rate;
    private float selection_share;
    private int n_individuals;
    private int n_generations;
    private String output_path;
    private Integer seed;

    private boolean fitted;

    private MersenneTwister mt;

    private DependencyNetwork dn;

    public EDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations, int thining_factor,
                 String variables_path, String options_path, String sampling_order_path, String output_path, Integer seed) throws Exception {

        this.learning_rate = learning_rate;
        this.selection_share = selection_share;
        this.n_individuals = n_individuals;
        this.n_generations = n_generations;
        this.thining_factor = thining_factor;

        this.output_path = output_path;
        this.variables_path = variables_path;
        this.options_path = options_path;
        this.sampling_order_path = sampling_order_path;

        this.fitted = false;

        if(seed == null) {
            this.mt = new MersenneTwister();
            this.seed = mt.nextInt();
        } else {
            this.seed = seed;
            this.mt = new MersenneTwister(seed);
        }

        this.dn = new DependencyNetwork(mt, variables_path, options_path, sampling_order_path);
    }

    public void fit(Instances train_data) throws Exception {
        FitnessCalculator fc = new FitnessCalculator(5, train_data, null);

        BaselineIndividual bi = new BaselineIndividual(train_data);
        HashMap<String, String> startPoint = bi.getCharacteristics();

        for(int c = 0; c < this.n_generations; c++) {
            Individual[] population = dn.gibbsSample(startPoint, thining_factor, this.n_individuals, train_data);
            Double[][] fitnesses = fc.evaluateEnsembles(seed, population);

            ArrayIndexComparator comparator = new ArrayIndexComparator(fitnesses[0]);
            Integer[] sortedIndices = comparator.createIndexArray();
            Arrays.sort(sortedIndices, comparator);

            int to_select = Math.round(this.selection_share * sortedIndices.length);

            this.dn.updateStructure(population, sortedIndices, to_select);
            this.dn.updateProbabilities(population, sortedIndices, to_select);

            // TODO update dependency network probabilities!

            System.out.println("WARNING: breaking after first generation!");  // TODO remove
            break;  // TODO remove
        }

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
                                Integer.parseInt(commandLine.getOptionValue("thining_factor")),
                                commandLine.getOptionValue("variables_path"),
                                commandLine.getOptionValue("options_path"),
                                commandLine.getOptionValue("sampling_order_path"),
                                commandLine.getOptionValue("metadata_path") + File.separator +
                                str_time + dataset_name,
                                commandLine.getOptionValue("seed") == null? null : Integer.parseInt(commandLine.getOptionValue("seed"))
                        );

                        ednel.fit(train_data);
                        // TODO predict!

                        System.out.println("WARNING: exiting after first fold!");  // TODO remove!
                        break;  // TODO remove!
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

