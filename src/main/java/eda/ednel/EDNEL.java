package eda.ednel;

import dn.DependencyNetwork;
import dn.stoppers.EarlyStop;
import eda.individual.BaselineIndividual;
import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import org.apache.commons.math3.random.MersenneTwister;
import utils.Argsorter;
import utils.PBILLogger;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class EDNEL extends AbstractClassifier {

    protected int burn_in;
    protected int early_stop_generations;
    protected float early_stop_tolerance;
    protected int thinning_factor;
    protected String options_path;
    protected String variables_path;
    protected float learning_rate;
    protected float selection_share;
    protected int n_individuals;
    protected int n_generations;

    protected int nearest_neighbor;

    protected PBILLogger pbilLogger;
    protected Integer seed;

    protected boolean fitted;

    protected MersenneTwister mt;

    protected DependencyNetwork dn;

    protected Individual currentGenBest;
    protected Individual overallBest;
    protected Double currentGenFitness;
    protected Double overallFitness;

    protected EarlyStop earlyStop;

    public EDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations,
                 int burn_in, int thinning_factor, int early_stop_generations, float early_stop_tolerance,
                 int nearest_neighbor,
                 String variables_path, String options_path, String sampling_order_path, PBILLogger pbilLogger,
                 Integer seed) throws Exception {

        this.learning_rate = learning_rate;
        this.selection_share = selection_share;
        this.n_individuals = n_individuals;
        this.n_generations = n_generations;
        this.burn_in = burn_in;
        this.thinning_factor = thinning_factor;
        this.early_stop_generations = early_stop_generations;
        this.early_stop_tolerance = early_stop_tolerance;
        this.nearest_neighbor = nearest_neighbor;

        this.pbilLogger = pbilLogger;
        this.variables_path = variables_path;
        this.options_path = options_path;

        this.overallFitness = -1.0;

        this.earlyStop = new EarlyStop(this.early_stop_generations, this.early_stop_tolerance);

        this.fitted = false;

        if(seed == null) {
            this.mt = new MersenneTwister();
            this.seed = mt.nextInt();
        } else {
            this.seed = seed;
            this.mt = new MersenneTwister(seed);
        }

        this.dn = new DependencyNetwork(
            mt, variables_path, options_path, sampling_order_path,
            this.burn_in, this.thinning_factor,
            this.learning_rate, this.n_generations,
            this.nearest_neighbor
        );
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        FitnessCalculator fc = new FitnessCalculator(5, data, null);

        BaselineIndividual bi = new BaselineIndividual(data);
        this.currentGenBest = bi;
        LocalDateTime t1, t2;
        for(int c = 0; c < this.n_generations; c++) {
             t1 = LocalDateTime.now();

            if(this.earlyStop.isStopping()) {
                break;
            }

            Individual[] population = dn.gibbsSample(
                    this.currentGenBest.getCharacteristics(), this.n_individuals, data
            );

            Double[][] fitnesses = fc.evaluateEnsembles(seed, population);
            Integer[] sortedIndices = Argsorter.decrescent_argsort(fitnesses[0]);

            this.currentGenBest = population[sortedIndices[0]];
            this.currentGenFitness = fitnesses[0][sortedIndices[0]];

            if(this.currentGenFitness > this.overallFitness) {
                this.overallFitness = this.currentGenFitness;
                this.overallBest = this.currentGenBest;
            }

            this.earlyStop.update(c, this.currentGenFitness);
            t2 = LocalDateTime.now();
            this.pbilLogger.log(
                    fitnesses[0], sortedIndices, population, this.overallBest, this.currentGenBest, this.dn, t1, t2
            );
            this.dn.update(population, sortedIndices, this.selection_share);
            this.pbilLogger.print();
        }
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

