package eda;

import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import org.apache.commons.cli.CommandLine;
import utils.PBILLogger;
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

    public RunTrainingPieceTask(
            String dataset_name, int n_sample, int n_fold, CommandLine commandLine,
            String str_time,
            Instances train_data, Instances test_data, Method writeMethod, Object writeObj
    ) throws Exception {
        this.dataset_name = dataset_name;
        this.n_sample = n_sample;
        this.n_fold = n_fold;

        this.str_time = str_time;

        this.train_data = train_data;
        this.test_data = test_data;

        this.writeMethod = writeMethod;
        this.writeObj = writeObj;

        PBILLogger pbilLogger = new PBILLogger(
                commandLine.getOptionValue("metadata_path") + File.separator +
                        str_time + File.separator + dataset_name,
                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                n_sample, n_fold,
                Boolean.parseBoolean(commandLine.getOptionValue("log"))

        );

        this.ednel = new EDNEL(
                Float.parseFloat(commandLine.getOptionValue("learning_rate")),
                Float.parseFloat(commandLine.getOptionValue("selection_share")),
                Integer.parseInt(commandLine.getOptionValue("n_individuals")),
                Integer.parseInt(commandLine.getOptionValue("n_generations")),
                Integer.parseInt(commandLine.getOptionValue("burn_in")),
                Integer.parseInt(commandLine.getOptionValue("thinning_factor")),
                Integer.parseInt(commandLine.getOptionValue("early_stop_generations")),
                Float.parseFloat(commandLine.getOptionValue("early_stop_tolerance")),
                Integer.parseInt(commandLine.getOptionValue("nearest_neighbor")),
                Integer.parseInt(commandLine.getOptionValue("max_parents")),
                commandLine.getOptionValue("variables_path"),
                commandLine.getOptionValue("options_path"),
                commandLine.getOptionValue("sampling_order_path"),
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

            try {
                this.ednel.getPbilLogger().toFile(this.ednel.getDependencyNetwork(), toReport, this.train_data, this.test_data);
            } catch(Exception e) {
                System.err.println("An error occurred. Could not write metadata to files:");
                System.err.println(e.getCause());
            }

            Double[] overall_auc = {FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getOverallBest())};
            Double[] last_auc = {FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, ednel.getCurrentGenBest())};

            this.writeMethod.invoke(this.writeObj, this.dataset_name, this.n_sample, this.n_fold, "last", last_auc);
            this.writeMethod.invoke(this.writeObj, this.dataset_name, this.n_sample, this.n_fold, "overall", overall_auc);

        } catch(Exception e) {
            System.err.println("An error occured, but could not be throwed:");
            System.err.println(e.getMessage());
            e.printStackTrace();
        }


    }
}
