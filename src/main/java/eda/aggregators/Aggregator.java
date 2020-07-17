package eda.aggregators;

public abstract class Aggregator {
    protected double[] competences;

    protected abstract void setOptions(Object... args);

    public abstract double[][] aggregateProba(double[][][] distributions);

    public abstract String[] getOptions();

//    public void setCompetences(double[] competences) {
//        this.competences = competences;
//    }
}
