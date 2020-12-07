package ednel.eda;

import ednel.Main;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import weka.core.Instances;

import java.io.*;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.concurrent.Callable;

public class RunTrainingPieceTask implements Runnable, Callable {
    // hyper-parameters of ednel

    private String dataset_name;
    private String datasets_path;

    private int n_sample;
    private int n_fold;

    /** The whole training set: learning + validation */
    private Instances train_data;
    /** Data to be used to evaluate quality of method */
    private Instances test_data;
    private EDNEL ednel;
    private PBILLogger pbilLogger;

    private boolean log;

    private LocalDateTime start, end;
    private Exception except;
    private boolean hasCompleted = false;
    private boolean hasSetAnException = false;

    /**
     * Runs a given fold of a 10-fold cross validation on EDNEL.
     *
     * @param dataset_name
     * @param n_sample
     * @param n_fold
     * @param cmd
     * @param str_time
     * @throws Exception
     */
    public RunTrainingPieceTask(
            String dataset_name, int n_sample, int n_fold, HashMap<String, String> cmd, String str_time
    ) throws Exception {
        start = LocalDateTime.now();

        this.dataset_name = dataset_name;
        this.n_sample = n_sample;
        this.n_fold = n_fold;

        this.log = Boolean.parseBoolean(cmd.get("log"));

        this.datasets_path = cmd.get("datasets_path");

        this.pbilLogger = new PBILLogger(
                dataset_name,
                cmd.get("metadata_path") + File.separator + str_time + File.separator + dataset_name,
                Integer.parseInt(cmd.get("n_individuals")),
                Integer.parseInt(cmd.get("n_generations")),
                this.n_sample, this.n_fold,
                Boolean.parseBoolean(cmd.get("log"))
        );

        this.ednel = new EDNEL(
                Double.parseDouble(cmd.get("learning_rate")),
                Float.parseFloat(cmd.get("selection_share")),
                Integer.parseInt(cmd.get("n_individuals")),
                Integer.parseInt(cmd.get("n_generations")),
                Integer.parseInt(cmd.get("timeout")),
                Integer.parseInt(cmd.get("timeout_individual")),
                Integer.parseInt(cmd.get("burn_in")),
                Integer.parseInt(cmd.get("thinning_factor")),
                Boolean.parseBoolean(cmd.get("no_cycles")),
                Integer.parseInt(cmd.get("early_stop_generations")),
//                Float.parseFloat(commandLine.get("early_stop_tolerance", "0.001")),
                Integer.parseInt(cmd.get("max_parents")),
                Integer.parseInt(cmd.get("delay_structure_learning")),
                pbilLogger,
                cmd.get("seed") == null?
                        null : Integer.parseInt(cmd.get("seed"))
        );
    }

    private void core() {
        try {
            HashMap<String, Instances> datasets = Main.loadDataset(
                    this.datasets_path,
                    this.dataset_name,
                    this.n_fold
            );
            this.train_data = datasets.get("train_data");
            this.test_data = datasets.get("test_data");

            this.ednel.buildClassifier(this.train_data);

            HashMap<String, Individual> toReport = new HashMap<>(2);
            toReport.put("overall", this.ednel.getOverallBest());
            toReport.put("last", this.ednel.getCurrentGenBest());
        } catch(Exception e) {
            if(this.log) {
                try {
                    File log_file = new File(String.format(
                            this.pbilLogger.getDatasetMetadataPath() + File.separator + "error_sample_%02d_fold_%02d.txt",
                            this.n_sample, this.n_fold
                    ));
                    PrintStream ps = new PrintStream(log_file);
                    e.printStackTrace(ps);
                    ps.close();
                } catch (IOException ex) {
                    // does nothing; too many exceptions to handle
                }
            }
            System.err.println("An error occurred, but could not be thrown:");
            System.err.println(e.getMessage());
            StackTraceElement[] track = e.getStackTrace();
            for(StackTraceElement trace : track) {
                System.err.println(trace.toString());
            }
            this.hasSetAnException = true;
            this.except = e;
        } finally {
            this.hasCompleted = true;
            this.end = LocalDateTime.now();
        }
    }

    @Override
    public Object call() throws Exception {
        this.core();
        return !this.hasSetAnException;
    }

    @Override
    public void run() {
        this.core();
    }

    /**
     * Whether an exception was set during execution.
     */
    public boolean hasSetAnException() {
        return hasSetAnException;
    }

    /**
     * The set exception (if any). Check it by using method hasSetAnException.
     */
    public Exception getSetException() {
        return except;
    }

    /**
     * Whether this code successfully completed execution.
     */
    public boolean hasCompleted() {
        return hasCompleted;
    }

    /**
     * Name of experimented dataset.
     */
    public String getDatasetName() {
        return this.dataset_name;
    }

    /**
     * Elapsed time (in seconds) of this algorithm run.
     */
    public int getElapsedTimeInSeconds() {
        return (int)this.start.until(this.end, ChronoUnit.SECONDS);
    }

    /**
     * How well the best individual found (according to fitness) performs on the test set.
     * @return Unweighted area under the ROC curve (Unweighted AUC) on the test set.
     */
    public Double overallIndividualPerformance() throws Exception {
        return FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getOverallBest());
    }

    /**
     * How well the best individual of the last generation (according to fitness) performs on the test set.
     * @return Unweighted area under the ROC curve (Unweighted AUC) on the test set.
     */
    public Double lastIndividualPerformance() throws Exception {
        return FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getCurrentGenBest());
    }
}
