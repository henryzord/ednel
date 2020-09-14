package ednel;

import ednel.eda.EDNEL;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.CommandLine;
import weka.core.Instances;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

public class TestDatasets {

    public static void main(String[] args) throws Exception {

        CommandLine commandLine = Main.parseCommandLine(args);

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);
//        PBILLogger.metadata_path_start(str_time, commandLine);

        File f = new File(commandLine.getOptionValue("datasets_path"));
        String[] dataset_names = f.list();

//        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

        // always 10 folds
//            CompileResultsTask compiler = new CompileResultsTask(dataset_names, n_samples, 10);


        for(String dataset_name : dataset_names) {
            System.out.println("on dataset " + dataset_name + "...");
            try {
                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = Main.loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );
                Instances train_data = curDatasetFolds.get(1).get("train");
                Instances test_data = curDatasetFolds.get(1).get("test");

                PBILLogger pbilLogger = new PBILLogger(
                        dataset_name,
                        commandLine.getOptionValue("metadata_path") + File.separator +
                                str_time + File.separator + dataset_name,
                        Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                        Integer.parseInt(commandLine.getOptionValue("n_generations")),
                        1, 1,
                        commandLine.hasOption("log")

                );

                EDNEL ednel = new EDNEL(
                        Double.parseDouble(commandLine.getOptionValue("learning_rate")),
                        Float.parseFloat(commandLine.getOptionValue("selection_share")),
                        Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                        Integer.parseInt(commandLine.getOptionValue("n_generations")),
                        Integer.parseInt(commandLine.getOptionValue("timeout", "-1")),
                        Integer.parseInt(commandLine.getOptionValue("burn_in")),
                        Integer.parseInt(commandLine.getOptionValue("thinning_factor")),
                        commandLine.hasOption("no_cycles"),
                        Integer.parseInt(commandLine.getOptionValue("early_stop_generations")),
                        Float.parseFloat(commandLine.getOptionValue("early_stop_tolerance")),
                        Integer.parseInt(commandLine.getOptionValue("max_parents")),
                        Integer.parseInt(commandLine.getOptionValue("delay_structure_learning")),
                        pbilLogger,
                        commandLine.getOptionValue("seed") == null?
                                null : Integer.parseInt(commandLine.getOptionValue("seed"))
                );
                ednel.buildClassifier(train_data);

                System.out.println("\tSuccess running dataset " + dataset_name);

            } catch(Exception e) {
                System.err.println("\tError running dataset " + dataset_name);
            }
        }
    }
}
