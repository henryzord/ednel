/**
 * Tests whether EDNEL will run on a given list of datasets.
 *
 * To test datasets use the following instruction:
 *
 * java -classpath ednel.jar ednel.TestDatasets --datasets_path C:\Users\henry\Projects\ednel\keel_datasets_10fcv --datasets_names pima --metadata_path C:\Users\henry\Projects\trash\ednel\ --n_generations 2 --n_individuals 25 --n_samples 1 --learning_rate 0.7 --selection_share 0.5 --burn_in 2 --thinning_factor 0 --early_stop_generations 200 --early_stop_tolerance 0.001 --max_parents 0 --delay_structure_learning 1 --n_jobs 1 --timeout 300
 *
 */

package ednel;

import ednel.eda.RunTrainingPieceTask;
import ednel.eda.individual.FitnessCalculator;
import ednel.utils.sorters.Argsorter;
import org.apache.commons.cli.CommandLine;
import smile.neighbor.lsh.Hash;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.sql.SQLOutput;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class TestDatasets {

    private static HashMap<String, Double[]> getDatasetsSizes(String[] dataset_names, String datasets_path) {
        Double[] sizes = new Double[dataset_names.length];
        Double[] n_instances = new Double[dataset_names.length];
        Double[] n_attributes = new Double[dataset_names.length];
        Double[] n_classes = new Double[dataset_names.length];

        int counter = 0;

        for(String dataset_name : dataset_names) {
            try {
                HashMap<String, Instances> curDatasetFolds = Main.loadDataset(datasets_path, dataset_name, 1);
                Instances train_data = curDatasetFolds.get("train_data");
                Instances test_data = curDatasetFolds.get("test_data");

                sizes[counter] = (double)((train_data.numInstances() + test_data.numInstances()) * train_data.numAttributes());
                n_instances[counter] = (double)(train_data.numInstances() + test_data.numInstances());
                n_attributes[counter] = (double)train_data.numAttributes();
                n_classes[counter] = (double)train_data.numClasses();

            } catch(Exception e) {
                sizes[counter] = Double.POSITIVE_INFINITY;
                n_instances[counter] = Double.POSITIVE_INFINITY;
                n_attributes[counter] = Double.POSITIVE_INFINITY;
                n_classes[counter] = Double.POSITIVE_INFINITY;
            } finally {
                counter += 1;
            }
        }
        HashMap<String, Double[]> dict = new HashMap<>();
        dict.put("sizes", sizes);
        dict.put("n_instances", n_instances);
        dict.put("n_attributes", n_attributes);
        dict.put("n_classes", n_classes);

        return dict;
    }

    public static RunTrainingPieceTask startAndRunTrainingTask(String dataset_name, String str_time, CommandLine cmd) {
        RunTrainingPieceTask task = null;
        try {
            HashMap<String, Instances> curDatasetFolds = Main.loadDataset(
                    cmd.getOptionValue("datasets_path"),
                    dataset_name,
                    1
            );
//            Instances train_data = curDatasetFolds.get("train_data");
//            Instances test_data = curDatasetFolds.get("test_data");

            task = new RunTrainingPieceTask(
                    dataset_name, 1, 1, cmd, str_time
            );
            task.run();
        } catch(Exception e) {
            System.err.println("Exception outside scope of EDA on dataset " + dataset_name + ": " + e.getMessage());
        } finally {
            return task;
        }
    }

    public static void main(String[] args) throws Exception {
        CommandLine commandLine = Main.parseCommandLine(args);

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);

        File f = new File(commandLine.getOptionValue("datasets_path"));
        String[] datasets_names = f.list();

        int n_jobs = Integer.parseInt(commandLine.getOptionValue("n_jobs"));
        
        System.out.println(String.format(
                "Will run %d experiments in %d threads; has %d cores available",
                datasets_names.length, n_jobs, Runtime.getRuntime().availableProcessors()
        ));

        HashMap<String, Double[]> datasetsInfo = TestDatasets.getDatasetsSizes(datasets_names, commandLine.getOptionValue("datasets_path"));

        HashMap<String, HashMap<String, Double>> infoPerDataset = new HashMap();
        for(int i = 0; i < datasets_names.length; i++) {
            HashMap<String, Double> localInfo = new HashMap<>();
            localInfo.put("n_instances", datasetsInfo.get("n_instances")[i]);
            localInfo.put("n_attributes", datasetsInfo.get("n_attributes")[i]);
            localInfo.put("n_classes", datasetsInfo.get("n_classes")[i]);
            infoPerDataset.put(datasets_names[i], localInfo);
        }

        Integer[] sortedIndices = Argsorter.crescent_argsort(datasetsInfo.get("sizes"));
        
        BufferedWriter bfw = new BufferedWriter(new FileWriter(new File("datasets_results.csv")));
        bfw.write("Dataset name,instances,attributes,classes,status,elapsed time (seconds),error message,test AUC (last),test AUC (overall)\n");

        for(int j = 0; j < datasets_names.length; j += n_jobs) {
             Object[] localTasks = IntStream.range(j, Math.min(j + n_jobs, datasets_names.length)).parallel().mapToObj(
                    i -> TestDatasets.startAndRunTrainingTask(datasets_names[sortedIndices[i]], str_time, commandLine)).toArray();

            for(Object localTask : localTasks) {
                RunTrainingPieceTask task = (RunTrainingPieceTask)localTask;

                if(task.hasCompleted()) {
                    if(task.hasSetAnException()) {
                        bfw.write(String.format(Locale.US,
                                "%s,%d,%d,%d,%s,%d,\"%s\",%f,%f\n", task.getDatasetName(),
                                infoPerDataset.get(task.getDatasetName()).get("n_instances").intValue(),
                                infoPerDataset.get(task.getDatasetName()).get("n_attributes").intValue(),
                                infoPerDataset.get(task.getDatasetName()).get("n_classes").intValue(),
                                "error",
                                task.getElapsedTimeInSeconds(), task.getSetException().getMessage(), -1.0, -1.0
                        ));
                    } else {
                        double last, overall;
                        try {
                            last = task.lastIndividualPerformance();
                            overall = task.overallIndividualPerformance();
                        } catch (Exception e) {
                            last = -1;
                            overall = -1;
                        }
                        bfw.write(String.format(Locale.US,
                                "%s,%d,%d,%d,%s,%d,\"%s\",%f,%f\n", task.getDatasetName(),
                                infoPerDataset.get(task.getDatasetName()).get("n_instances").intValue(),
                                infoPerDataset.get(task.getDatasetName()).get("n_attributes").intValue(),
                                infoPerDataset.get(task.getDatasetName()).get("n_classes").intValue(),
                                "success",
                                task.getElapsedTimeInSeconds(), "", last, overall
                        ));
                    }
                } else {
                    bfw.write(String.format(Locale.US,
                            "%s,%d,%d,%d,%s,%d,\"%s\",%f,%f\n", task.getDatasetName(),
                            infoPerDataset.get(task.getDatasetName()).get("n_instances").intValue(),
                            infoPerDataset.get(task.getDatasetName()).get("n_attributes").intValue(),
                            infoPerDataset.get(task.getDatasetName()).get("n_classes").intValue(),
                            "not finished",
                            0, "", -1.0, -1.0
                    ));
                }
            }
        }
        bfw.close();
        System.out.println("finished!");
    }
}
