package dn.stoppers;

public class EarlyStop {

    protected int delayGenerations;
    protected double tolerance;

    private double minFitness;
    private double maxFitness;
    private double[] lastBests;

    public EarlyStop(int delayGenerations, double tolerance) {
        this.delayGenerations = delayGenerations;
        this.tolerance = tolerance;

        this.lastBests = new double [this.delayGenerations];
        for(int i = 0; i < this.delayGenerations; i++) {
            this.lastBests[i] = (double)i/this.delayGenerations;
        }
        this.minFitness = 100;
        this.maxFitness = -100;
    }

    public EarlyStop() {
        this(10, 0.005);
    }

    public boolean isStopping() {
        return Math.abs(this.maxFitness - this.minFitness) < this.tolerance;
    }

    public void update(int gen, double fitness) {
        this.lastBests[gen % this.delayGenerations] = fitness;
        this.minFitness = fitness;
        this.maxFitness = fitness;
        for(int i = 0; i < this.lastBests.length; i++) {
            if(this.lastBests[i] < this.minFitness) {
                this.minFitness = this.lastBests[i];
            }
            if(this.lastBests[i] > this.maxFitness) {
                this.maxFitness = this.lastBests[i];
            }
        }
    }

}