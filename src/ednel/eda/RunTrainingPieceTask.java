package ednel.eda;

import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import org.apache.commons.cli.CommandLine;
import ednel.utils.PBILLogger;
import weka.core.Instances;

import java.io.File;
import java.lang.reflect.Method;
import java.util.HashMap;

public class RunTrainingPieceTask implements Runnable {
    private String dataset_name;
    private int n_sample;
    private int n_fold;
    private String str_time;
    private Instances train_data;
    private Instances test_data;
    private EDNEL ednel;

    private Method writeMethod;
    private Object writeObj;

    private boolean log;

    private boolean setException = false;

    public RunTrainingPieceTask(
            String dataset_name, int n_sample, int n_fold, CommandLine commandLine, String str_time,
            Instances train_data, Instances test_data, Method writeMethod, Object writeObj
    ) throws Exception {
        this.dataset_name = dataset_name;
        this.n_sample = n_sample;
        this.n_fold = n_fold;

        this.log = commandLine.hasOption("log");

        this.str_time = str_time;

        this.train_data = train_data;
        this.test_data = test_data;

        this.writeMethod = writeMethod;
        this.writeObj = writeObj;

        PBILLogger pbilLogger = new PBILLogger(
                dataset_name,
                commandLine.getOptionValue("metadata_path") + File.separator +
                        str_time + File.separator + dataset_name,
                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                n_sample, n_fold,
                commandLine.hasOption("log")

        );

        this.ednel = new EDNEL(
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
                commandLine.getOptionValue("resources_path"),
                pbilLogger,
                commandLine.getOptionValue("seed") == null?
                        null : Integer.parseInt(commandLine.getOptionValue("seed"))
        );
    }

    @Override
    public void run() {
        try {
            this.ednel.buildClassifier(this.train_data);

            HashMap<String, Individual> toReport = new HashMap<>(2);
            toReport.put("overall", this.ednel.getOverallBest());
            toReport.put("last", this.ednel.getCurrentGenBest());

            if(this.log) {
                this.ednel.getPbilLogger().toFile(this.ednel.getDependencyNetwork(), toReport, this.train_data, this.test_data);
            }

            if((this.writeMethod != null) && (this.writeObj != null)) {
                Double[] overall_auc = {FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getOverallBest())};
                Double[] last_auc = {FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getCurrentGenBest())};

                this.writeMethod.invoke(this.writeObj, this.dataset_name, this.n_sample, this.n_fold, "last", last_auc);
                this.writeMethod.invoke(this.writeObj, this.dataset_name, this.n_sample, this.n_fold, "overall", overall_auc);
            }

        } catch(Exception e) {
            System.err.println("An error occurred, but could not be thrown:");
            System.err.println(e.getMessage());
            e.printStackTrace();
            this.setException = true;
        }
    }

    public boolean triggerException() {
        return setException;
    }
}
