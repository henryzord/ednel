package eda.ednel;

import weka.core.Instance;
import weka.core.Instances;

public class OverallEDNEL extends EDNEL {
    public OverallEDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations, int thining_factor, String variables_path, String options_path, String sampling_order_path, String output_path, Integer seed) throws Exception {
        super(learning_rate, selection_share, n_individuals, n_generations, thining_factor, variables_path, options_path, sampling_order_path, output_path, seed);
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return this.overallBest.distributionsForInstances(batch);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return this.overallBest.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.overallBest.distributionForInstance(instance);
    }
}
