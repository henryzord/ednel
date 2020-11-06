package ednel.eda;

import ednel.network.DependencyNetwork;
import ednel.eda.stoppers.EarlyStop;
import ednel.eda.individual.BaselineIndividual;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.utils.comparators.Argsorter;
import ednel.utils.PBILLogger;
import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;

public class EDNEL extends AbstractClassifier {

    private final int timeout;
    protected final int burn_in;
    protected final int early_stop_generations;
    protected final float early_stop_tolerance;
    protected final int thinning_factor;
    protected final double learning_rate;
    protected final float selection_share;
    protected final int n_individuals;
    protected final int n_generations;

    protected final int max_parents;
    private final int delay_structure_learning;
    private final int timeout_individual;

    protected PBILLogger pbilLogger;
    protected final Integer seed;

    protected boolean fitted;

    protected MersenneTwister mt;

    protected DependencyNetwork dn;

    protected Individual currentGenBest;
    protected Individual overallBest;
    protected Double currentGenFitness;
    protected Double overallFitness;

    protected boolean no_cycles;

    protected EarlyStop earlyStop;

    public EDNEL(double learning_rate, float selection_share, int n_individuals, int n_generations,
                 int timeout, int timeout_individual, int burn_in, int thinning_factor, boolean no_cycles, int early_stop_generations,
                 float early_stop_tolerance, int max_parents, int delay_structure_learning, PBILLogger pbilLogger,
                 Integer seed
    ) throws Exception {

        this.learning_rate = learning_rate;
        this.selection_share = selection_share;
        this.n_individuals = n_individuals;
        this.n_generations = n_generations;
        this.timeout = timeout;
        this.timeout_individual = timeout_individual;
        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;
        this.early_stop_generations = early_stop_generations;
        this.early_stop_tolerance = early_stop_tolerance;
        this.max_parents = max_parents;
        this.delay_structure_learning = delay_structure_learning;

        this.no_cycles = no_cycles;

        this.pbilLogger = pbilLogger;

        this.overallFitness = -1.0;

        // at least 10 generations
        this.earlyStop = new EarlyStop(this.early_stop_generations, this.early_stop_tolerance, 10);

        this.fitted = false;

        if(seed == null) {
            this.mt = new MersenneTwister();
            this.seed = mt.nextInt();
        } else {
            this.seed = seed;
            this.mt = new MersenneTwister(seed);
        }

        this.dn = new DependencyNetwork(
                mt, this.burn_in, this.thinning_factor, this.no_cycles, this.learning_rate,
                this.max_parents, this.delay_structure_learning, this.timeout_individual
        );
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        LocalDateTime start = LocalDateTime.now();
        LocalDateTime t1 = LocalDateTime.now(), t2;

        FitnessCalculator fc = new FitnessCalculator(5, data);

        this.currentGenBest = new BaselineIndividual();
        this.overallBest = this.currentGenBest;

        for(int c = 0; c < this.n_generations; c++) {
            HashMap<String, Object[]> sampled = dn.gibbsSample(
                    this.currentGenBest.getCharacteristics(), this.n_individuals, fc, this.seed
            );
            Individual[] population = (Individual[])sampled.get("population");
            Double[] fitnesses = (Double[])sampled.get("fitnesses");
            Integer[] sortedIndices = Argsorter.decrescent_argsort(fitnesses);

            this.currentGenBest = population[sortedIndices[0]];
            this.currentGenFitness = fitnesses[sortedIndices[0]];

            if(this.currentGenFitness > this.overallFitness) {
                this.overallFitness = this.currentGenFitness;
                this.overallBest = this.currentGenBest;
            }

            t2 = LocalDateTime.now();
            if(this.pbilLogger != null) {
                this.pbilLogger.log_and_print(
                        fitnesses, sortedIndices, population, this.overallBest, this.currentGenBest, this.dn, t1, t2
                );
            }
            t1 = t2;

            this.earlyStop.update(c, this.currentGenFitness, this.currentGenBest);

            boolean overTime = (this.timeout > 0) && ((int)start.until(t1, ChronoUnit.SECONDS) > this.timeout);
            if(this.earlyStop.isStopping(c) || overTime) {
                break;
            }
            this.dn.update(population, sortedIndices, this.selection_share, c);
        }
        this.overallBest = this.earlyStop.getLastBestIndividual();  // TODO is this correct?

        this.overallBest.buildClassifier(data);
        this.currentGenBest.buildClassifier(data);

        this.fitted = true;
    }

    public PBILLogger getPbilLogger() {
        return pbilLogger;
    }

    public Individual getCurrentGenBest() {
        return currentGenBest;
    }

    public Individual getOverallBest() {
        return overallBest;
    }

    public DependencyNetwork getDependencyNetwork() {
        return this.dn;
    }
}

