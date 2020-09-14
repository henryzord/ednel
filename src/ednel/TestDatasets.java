/**
 * To test datasets use the following instruction:
 *
 * java -classpath ednel.jar ednel.TestDatasets --datasets_path C:\Users\henry\Projects\ednel\keel_datasets_10fcv --datasets_names pima --metadata_path C:\Users\henry\Projects\trash\ednel\ --n_generations 2 --n_individuals 25 --n_samples 1 --learning_rate 0.7 --selection_share 0.5 --burn_in 2 --thinning_factor 0 --early_stop_generations 200 --early_stop_tolerance 0.001 --max_parents 0 --delay_structure_learning 1 --n_jobs 1 --timeout 300
 *
 */

package ednel;

import ednel.eda.RunTrainingPieceTask;
import org.apache.commons.cli.CommandLine;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class TestDatasets {

    public static void main(String[] args) throws Exception {

        CommandLine commandLine = Main.parseCommandLine(args);

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);

        File f = new File(commandLine.getOptionValue("datasets_path"));
        String[] dataset_names = f.list();

        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(1);

        long timeout_seconds = Long.parseLong(commandLine.getOptionValue("timeout"));
        System.out.println("timeout seconds: " + timeout_seconds);

        RunTrainingPieceTask[] tasks = new RunTrainingPieceTask [dataset_names.length];
        int counter_task = 0;

        for(String dataset_name : dataset_names) {
            try {
                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = Main.loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );
                Instances train_data = curDatasetFolds.get(1).get("train");
                Instances test_data = curDatasetFolds.get(1).get("test");

                RunTrainingPieceTask task = new RunTrainingPieceTask(
                        dataset_name, 1, 1, commandLine, str_time, train_data, test_data, null, null
                );
                tasks[counter_task] = task;
                counter_task += 1;
                executor.execute(task);
            } catch(Exception e) {
                System.err.println("Exception outside scope of EDA on dataset " + dataset_name + ": " + e.getMessage());
            }
        }
         executor.awaitTermination(timeout_seconds * dataset_names.length, TimeUnit.SECONDS);

        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File("datasets_results.csv")));

        for(int i = 0; i < counter_task; i++) {
            RunTrainingPieceTask task = tasks[i];
            if(task.hasCompleted()) {
                if(task.hasSetAnException()) {
                    bfw.write(String.format("%s,%s,\"%s\"\n", task.getDatasetName(), "error", task.getSetException().getMessage()));
                } else {
                    bfw.write(String.format("%s,%s,\"%s\"\n", task.getDatasetName(), "success", ""));
                }
            } else {
                bfw.write(String.format("%s,%s,\"%s\"\n", task.getDatasetName(), "not finished", ""));
            }
        }
        bfw.close();
        System.out.println("finished!");
        executor.shutdownNow();
    }
}
