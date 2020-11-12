package ednel.eda;

import ednel.Main;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.CommandLine;
import weka.core.Instances;

import java.io.File;
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
     * @param commandLine
     * @param str_time
     * @throws Exception
     */
    public RunTrainingPieceTask(
            String dataset_name, int n_sample, int n_fold, CommandLine commandLine, String str_time
    ) throws Exception {
        start = LocalDateTime.now();

        this.dataset_name = dataset_name;
        this.n_sample = n_sample;
        this.n_fold = n_fold;

        this.log = commandLine.hasOption("log");

        this.datasets_path = commandLine.getOptionValue("datasets_path");

//        this.log_test = commandLine.hasOption("log_test");

        this.pbilLogger = new PBILLogger(
                dataset_name,
                commandLine.getOptionValue("metadata_path") + File.separator +
                        str_time + File.separator + dataset_name,
                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                this.n_sample, this.n_fold,
                commandLine.hasOption("log")
        );

        this.ednel = new EDNEL(
                Double.parseDouble(commandLine.getOptionValue("learning_rate")),
                Float.parseFloat(commandLine.getOptionValue("selection_share")),
                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                Integer.parseInt(commandLine.getOptionValue("timeout", "-1")),
                Integer.parseInt(commandLine.getOptionValue("timeout_individual", "60")),
                Integer.parseInt(commandLine.getOptionValue("burn_in", "100")),
                Integer.parseInt(commandLine.getOptionValue("thinning_factor", "0")),
                commandLine.hasOption("no_cycles"),
                Integer.parseInt(commandLine.getOptionValue("early_stop_generations")),
                Float.parseFloat(commandLine.getOptionValue("early_stop_tolerance", "0.001")),
                Integer.parseInt(commandLine.getOptionValue("max_parents")),
                Integer.parseInt(commandLine.getOptionValue("delay_structure_learning")),
                pbilLogger,
                commandLine.getOptionValue("seed") == null?
                        null : Integer.parseInt(commandLine.getOptionValue("seed"))
        );
    }

    private void core() {
        try {
//            throw new Exception("use validation set!");

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

            if(this.log) {
                this.ednel.getPbilLogger().toFile(this.ednel.getDependencyNetwork(), toReport, this.train_data, this.test_data);
            }
        } catch(Exception e) {
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
