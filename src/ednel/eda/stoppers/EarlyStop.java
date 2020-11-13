package ednel.eda.stoppers;

import ednel.eda.individual.Individual;

public class EarlyStop {
    private int windowSize;
    private int startGen;
    private double tolerance;
    private Double bestFitness;
    private Individual bestIndividual;

    private int faultCounter;


    public EarlyStop(int windowSize, double tolerance, int startGen) {
        this.windowSize = windowSize;
        this.tolerance = tolerance;
//        this.startGen = Math.max(startGen, this.windowSize);
        this.startGen = startGen;

        this.faultCounter = 0;
        this.bestFitness = -1.0;
        this.bestIndividual = null;
    }

    public boolean isStopping(int gen) {
        return ((gen > this.startGen) && (this.faultCounter > this.windowSize));
    }

    /**
     *
     * @param gen Current generation
     * @param ind Current generation best individual
     */
    public void update(int gen, Individual ind) {
        if(gen > this.startGen) {
            if(ind.getFitness().getValQuality() >= this.bestFitness) {
//            if((ind.getFitness().getValQuality() - this.bestFitness) >= this.tolerance) {
                this.faultCounter = 0;
                this.bestFitness = ind.getFitness().getValQuality();
                this.bestIndividual = ind;
            } else {
                this.faultCounter += 1;
            }
        } else {
            this.bestIndividual = ind;
            this.bestFitness = ind.getFitness().getValQuality();
        }
    }

    public int getWindowSize() {
        return windowSize;
    }

    public double getTolerance() {
        return tolerance;
    }

    public Individual getLastBestIndividual() {
        return this.bestIndividual;
    }

}
