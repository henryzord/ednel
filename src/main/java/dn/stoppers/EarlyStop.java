package dn.stoppers;

public class EarlyStop {
    private int delayGenerations;
    private double tolerance;
    private double[] lastBests;

    public EarlyStop(int delayGenerations, double tolerance) {
        this.delayGenerations = delayGenerations;
        this.tolerance = tolerance;

        this.lastBests = new double [this.delayGenerations];
        for(int i = 0; i < this.delayGenerations; i++) {
            this.lastBests[i] = 0;
        }
        this.lastBests[0] = this.tolerance * 2;
    }

    public boolean isStopping() {
        return Math.abs(this.lastBests[this.lastBests.length - 1] - this.lastBests[0]) < this.tolerance;
    }

    public void update(int gen, double fitness) {
        this.lastBests[gen % this.delayGenerations] = fitness;
    }

    public int getDelayGenerations() {
        return delayGenerations;
    }

    public double getTolerance() {
        return tolerance;
    }
}
