package ednel;

import ednel.eda.EDNEL;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

public class ProfilableMain {

    public static void main(String[] args) {
        try {
            CommandLine commandLine = Main.parseCommandLine(args);

            // writes metadata
            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
            LocalDateTime now = LocalDateTime.now();
            String str_time = dtf.format(now);
            PBILLogger.metadata_path_start(str_time, commandLine);

            String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

            HashMap<String, Instances> datasets = Main.loadDataset(commandLine.getOptionValue("datasets_path"), dataset_names[0],  1);
            Instances train_data = datasets.get("train_data");

            PBILLogger pbilLogger = new PBILLogger(
                    dataset_names[0],
                    commandLine.getOptionValue("metadata_path") + File.separator +
                            str_time + File.separator + dataset_names[0],
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
//            ednel.distributionsForInstances(test_data);
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
