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
        this.startGen = Math.max(startGen, this.windowSize);

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
     * @param fitness Fitness of the best individual of the current generation
     * @param ind Current generation best individual
     */
    public void update(int gen, double fitness, Individual ind) {
        if(gen > this.startGen) {
            if((fitness - this.bestFitness) >= this.tolerance) {
                this.faultCounter = 0;
                this.bestFitness = fitness;
                this.bestIndividual = ind;
            } else {
                this.faultCounter += 1;
            }
        } else {
            this.bestIndividual = ind;
            this.bestFitness = fitness;
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
