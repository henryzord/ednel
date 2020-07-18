package eda.aggregators;

import java.io.Serializable;

public class RuleExtractorAggregator extends Aggregator implements Serializable {


    @Override
    protected void setOptions(Object... args) {

    }

    @Override
    public double[][] aggregateProba(double[][][] distributions) {
        return new double[0][];
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
