package ednel.eda.stoppers;

public class EarlyStop {
    private int delayGenerations;
    private double tolerance;
    private double[] lastMedians;

    public EarlyStop(int delayGenerations, double tolerance) {
        this.delayGenerations = delayGenerations;
        this.tolerance = tolerance;

        this.lastMedians = new double [this.delayGenerations];
        for(int i = 0; i < this.delayGenerations; i++) {
            this.lastMedians[i] = 0;
        }
        this.lastMedians[0] = this.tolerance * 2;
    }

    public boolean isStopping() {
        return Math.abs(this.lastMedians[this.lastMedians.length - 1] - this.lastMedians[0]) < this.tolerance;
    }

    public void update(int gen, double medianFitness) {
        this.lastMedians[gen % this.delayGenerations] = medianFitness;
    }

    public int getDelayGenerations() {
        return delayGenerations;
    }

    public double getTolerance() {
        return tolerance;
    }
}
