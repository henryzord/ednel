/*
  This class is intended to investigate overfitting in EDNEL.
  WARNING: IT LOGS THE TEST DATA. DO NOT USE THIS CLASS FOR EXPERIMENTS.

  Its sole purpose is to check if overfitting is occurring.

  Usage:

  java -classpath ednel.jar ednel.utils.analysis.InvestigateOverfitting --datasets_path <datasets_path>
  --datasets_names <datasets_name> --metadata_path <metadata_path> --n_generations <n_generations>
  --n_individuals <n_individuals> --n_samples <n_samples> --learning_rate <learning_rate>
  --selection_share <selection_share> --burn_in <burn_in> --thinning_factor <thinning_factor>
  --early_stop_generations <early_stop_generations> --max_parents <max_parents>
  --delay_structure_learning <delay_structure_learning> --n_jobs <n_jobs> --timeout <timeout>
  --timeout_individual <timeout_individual>

 */

package ednel.utils.analysis;

import ednel.Main;
import ednel.eda.RunFoldOfTenFoldCrossValidation;
import ednel.utils.PBILLogger;
import org.apache.commons.math3.random.MersenneTwister;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

public class InvestigateOverfitting {

    public static void main(String[] args) throws Exception {
        HashMap<String, String> commandLine = Main.parseCommandLine(args);
        commandLine.put("log_test", "true");  // this option can only be set internally
        System.out.println("n_samples will not be used in this class");

        int n_jobs = Integer.parseInt(commandLine.get("n_jobs"));

        // writes metadata
        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd-HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String str_time = dtf.format(now);
        PBILLogger.metadata_path_start(str_time, commandLine);

        String[] dataset_names = commandLine.get("datasets_names").split(",");

        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(n_jobs);

        final int n_tasks = dataset_names.length;

        ArrayList<Callable<Object>> taskQueue = new ArrayList<>(n_tasks);

        MersenneTwister mt = new MersenneTwister();

        for(String dataset_name : dataset_names) {
            int n_fold = mt.nextInt(10) + 1;

            RunFoldOfTenFoldCrossValidation task = new RunFoldOfTenFoldCrossValidation(
                    dataset_name, 1, n_fold, commandLine, str_time
            );
            taskQueue.add(Executors.callable(task));
        }
        ArrayList<Future<Object>> answers = (ArrayList<Future<Object>>)executor.invokeAll(taskQueue);
        executor.shutdown();
        int finished = 0;
        for (Future<Object> answer : answers) {
            finished += answer.isDone() ? 1 : 0;
        }
        System.out.printf("%d tasks completed%n", finished);
    }
}
