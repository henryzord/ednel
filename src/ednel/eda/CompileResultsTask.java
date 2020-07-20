package ednel.eda;

import ednel.eda.individual.FitnessCalculator;
import weka.core.Instances;

import java.security.KeyException;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

public class CompileResultsTask implements Runnable {

    private int n_folds;
    private String[] dataset_names;
    private int n_samples;
    private HashMap<String, HashMap<Integer, HashMap<Integer, Double>>> last_aucs;
    private HashMap<String, HashMap<Integer, HashMap<Integer, Double>>> overall_aucs;
    private HashMap<String, Boolean> checked;

    public CompileResultsTask(String[] dataset_names, int n_samples, int n_folds) {
        last_aucs = new HashMap<>();
        overall_aucs = new HashMap<>();

        this.dataset_names = dataset_names;
        this.n_samples = n_samples;
        this.n_folds = n_folds;

        checked = new HashMap<>();
        for(String dataset_name : dataset_names) {
            checked.put(dataset_name, Boolean.FALSE);
            last_aucs.put(dataset_name, new HashMap<>());
            overall_aucs.put(dataset_name, new HashMap<>());
            for(int n_sample = 1; n_sample < n_samples + 1; n_sample++) {
                checked.put(String.format("%s_%d", dataset_name, n_sample), Boolean.FALSE);
                last_aucs.get(dataset_name).put(n_sample, new HashMap<>());
                overall_aucs.get(dataset_name).put(n_sample, new HashMap<>());
                for(int n_fold = 1; n_fold < n_folds + 1; n_fold++) {
                    checked.put(String.format("%s_%d_%d", dataset_name, n_sample, n_fold), Boolean.FALSE);
                }
            }
        }
    }

    private synchronized Boolean accessCheckingData(String dataset_name, Object... args) {
        Boolean  write = false;
        Boolean res = null;
        String query_string = dataset_name;

        if(args != null && args.length > 0) {
            for(int i = 0; i < args.length; i++) {
                if(args[i] instanceof Boolean) {
                    write = (Boolean)args[i];
                    break;
                }
                query_string += "_" + args[i];
            }
        }
        if(write) {
            this.checked.put(query_string, true);
        } else {
            res = this.checked.get(query_string);
        }
        return res;
    }

    public synchronized double accessAUCData(String dataset_name, int n_sample, int n_fold, String set, Double... value) {
        double res_value = -1;

        // writing
        if(value != null && value.length > 0) {
            if(set.equals("last")) {
                last_aucs.get(dataset_name).get(n_sample).put(n_fold, value[0]);
                Object data = overall_aucs.get(dataset_name).get(n_sample).getOrDefault(n_fold, null);
                if(data != null) {
                    this.accessCheckingData(dataset_name, n_sample, n_fold, true);
                }
            } else if(set.equals("overall")) {
                overall_aucs.get(dataset_name).get(n_sample).put(n_fold, value[0]);
                Object data = last_aucs.get(dataset_name).get(n_sample).getOrDefault(n_fold, null);
                if(data != null) {
                    this.accessCheckingData(dataset_name, n_sample, n_fold, true);
                }
            }
        } else { // reading
            if(set.equals("last")) {
                res_value = last_aucs.getOrDefault(
                        dataset_name, new HashMap<>()
                ).getOrDefault(
                        n_sample, new HashMap<>()
                ).getOrDefault(n_fold, -1.0);
                this.accessCheckingData(dataset_name, n_sample, n_fold, true);
            } else if(set.equals("overall")) {
                res_value = overall_aucs.getOrDefault(
                        dataset_name, new HashMap<>()
                ).getOrDefault(
                        n_sample, new HashMap<>()
                ).getOrDefault(n_fold, -1.0);
            }
        }
        return res_value;
    }

    @Override
    public void run() {
        while(true) {  // all datasets not completed yet
            int count_datasets = 0;
            for(String dataset_name : this.dataset_names) {
                boolean this_dataset = this.accessCheckingData(dataset_name);
                count_datasets += this_dataset? 1 : 0;
                if(!this_dataset) {
                    int count_samples = 0;
                    for(int n_sample = 1; n_sample < this.n_samples + 1; n_sample++) {
                        boolean this_sample = this.accessCheckingData(dataset_name, n_sample);
                        count_samples += this_sample? 1 : 0;
                        if(!this_sample) {
                            int count_folds = 0;
                            for(int n_fold = 1; n_fold < this.n_folds + 1; n_fold++) {
                                boolean this_fold = this.accessCheckingData(dataset_name, n_sample, n_fold);
                                count_folds += this_fold? 1 : 0;
                            }
                            if(count_folds == this.n_folds) {
                                // finished this sample

                                double last_auc = 0;
                                double overall_auc = 0;
                                for(int n_fold = 1; n_fold < this.n_folds + 1; n_fold++) {
                                    last_auc += this.accessAUCData(dataset_name, n_sample, n_fold, "last") / this.n_folds;
                                    overall_auc += this.accessAUCData(dataset_name, n_sample, n_fold, "overall") / this.n_folds;
                                }
                                System.out.println(String.format("Results for dataset %s sample %d:", dataset_name, n_sample));
                                System.out.println(String.format("\tOverall: %02.4f\n\tLast: %02.4f", overall_auc, last_auc));

                                this.accessCheckingData(dataset_name, n_sample, true);
                            }
                        }
                    }
                    if(count_samples == this.n_samples) {  // finished this dataset
                        double last_auc = 0;
                        double overall_auc = 0;
                        for(int n_sample = 1; n_sample < this.n_samples + 1; n_sample++) {
                            for (int n_fold = 1; n_fold < this.n_folds + 1; n_fold++) {
                                last_auc += this.accessAUCData(dataset_name, n_sample, n_fold, "last") / (this.n_folds * this.n_samples);
                                overall_auc += this.accessAUCData(dataset_name, n_sample, n_fold, "overall") / (this.n_folds * this.n_samples);
                            }
                        }
                        System.out.println(String.format("Results for dataset %s/mean of %d samples:", dataset_name, this.n_samples));
                        System.out.println(String.format("\tOverall: %02.4f\n\tLast: %02.4f", overall_auc, last_auc));

                        this.accessCheckingData(dataset_name, true);
                    }
                }
            }
            if(count_datasets == this.dataset_names.length) {
                // finished all datasets
                break;
            }
            try {
                TimeUnit.SECONDS.sleep(60);
            } catch (InterruptedException e) {
                // nothing happens
            }
        }
    }
}

