package ednel.eda.stoppers;

import ednel.eda.individual.Individual;

public class EarlyStop {
    private int startGen;
    private int windowSize;
    private Double bestFitness;
    private Individual bestIndividual;

    private int faultCounter;


    public EarlyStop(int windowSize, int startGen) {
        this.startGen = startGen;
        this.windowSize = windowSize;

        this.faultCounter = 0;
        this.bestFitness = -1.0;
        this.bestIndividual = null;
    }

    /**
     * Updates early stop with the best individual from the current generation.
     * The fitness value depends on the evaluation method. Please see documentation for FitnessCalculator.
     *
     * @param gen Current generation
     * @param ind Current generation best individual, based on fitness (either on learn set or validation set. Depends
     *            on the evaluation method - holdout, leave-one-out, cross-validation)
     */
    public void update(int gen, Individual ind, Double bestFitness) {
        if(gen > this.startGen) {
            if(bestFitness >= this.bestFitness) {
                this.faultCounter = 0;
                this.bestFitness = bestFitness;
                this.bestIndividual = ind;
            } else {
                this.faultCounter += 1;
            }
        } else {
            this.bestIndividual = ind;
            this.bestFitness = bestFitness;
        }
    }

    public boolean isStopping(int gen) {
        return ((gen > this.startGen) && (this.faultCounter > this.windowSize));
    }

    public int getWindowSize() {
        return windowSize;
    }

    public Individual getBestIndividual() {
        return this.bestIndividual;
    }

}
