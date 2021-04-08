package ednel.eda;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.eda.individual.BaselineIndividual;
import ednel.eda.individual.Fitness;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.eda.stoppers.EarlyStop;
import ednel.network.DependencyNetwork;
import ednel.utils.PBILLogger;
import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;

public class EDNEL extends AbstractClassifier {

    /** How much time (in seconds) this classifier has to be trained. Early stops if exceeds time */
    private final int timeout;
    // TODO: fazer talvez com que indivíduo tenha sua fitness setada para zero? Tem que ver as implicações na população
    // TODO: (e.g. não pode ter uma população inteira de indivíduos inválidos)
    /** How much time each individual has to be trained. If an individual exceeds this time, it is discarded and resampled */
    protected final int timeout_individual;

    /** How many generations to tolerate when a decrease in validation fitness is detected. Larger values are more tolerant */
    protected final int early_stop_generations;

    /** The ratio at which probabilities must be changed in GM. Higher values imply faster changes. Must be in (0, 1] */
    protected final float learning_rate;
    /** Ratio of individuals, in the current generation, that will be used to update GM probabilities. Must be in (0, 1] */
    protected final float selection_share;
    /** Number of individuals to have concomitantly in the same generation */
    protected final int n_individuals;
    /** Number of generations to run EDA */
    protected final int n_generations;

    /** How many samples to discard when sampling solutions in Dependency Network */
    protected final int burn_in;
    /** How many samples to discard between valid samples when sampling solution in Dependency Network */
    protected final int thinning_factor;
    /** How many generations of individuals that updated GM probabilities to collect before updating the structure of GM */
    protected final int delay_structure_learning;
    /**
     * Maximum number of probabilistic parents a variable might have in Graphical Model.
     * Does not affect deterministic parents count (which is pre-built and never changes) */
    protected final int max_parents;
    /** whether to allow inner cycles in GM structure */
    protected boolean no_cycles;

    /** Number of internal folds to use when evaluating fitness */
    protected int n_internal_folds;

    /** Random number generation (RNG) */
    protected MersenneTwister mt;
    /** Seed to use. Defaults to system clock if not passed */
    protected final Integer seed;


    /** PBILLogger instance that will log metadata of this EDNEL run */
    protected PBILLogger pbilLogger;

    /** True if this classifier was already trained, false otherwise */
    protected boolean fitted;

    /** Instance of Dependency Network used by this classifier */
    protected DependencyNetwork dn;

    /** Best individual (based on fitness) from current generation */
    protected Individual currentGenBest;
    /** Individual with overall best validation fitness */
    protected Individual overallBest;

    /** EarlyStop instance */
    protected EarlyStop earlyStop;

    public EDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations,
                 int timeout, int timeout_individual, int burn_in, int thinning_factor, boolean no_cycles, int early_stop_generations,
                 int max_parents, int delay_structure_learning, int n_internal_folds, PBILLogger pbilLogger, Integer seed
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
        this.max_parents = max_parents;

        this.delay_structure_learning = this.max_parents == 0? 0 : delay_structure_learning;

        this.n_internal_folds = n_internal_folds;

        this.earlyStop = null;

        this.no_cycles = no_cycles;

        this.pbilLogger = pbilLogger;

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
    public void buildClassifier(Instances input_data) throws Exception {
        LocalDateTime start = LocalDateTime.now();
        LocalDateTime t1 = LocalDateTime.now(), t2;

        Instances train_data;
        Instances learn_data;
        Instances val_data;

        if(this.n_internal_folds == 0) {  // holdout
            train_data = FitnessCalculator.betterStratifier(input_data, 10);  // uses 10 as default
            RemovePercentage rp = new RemovePercentage();
            rp.setInputFormat(train_data);
            rp.setPercentage(20);
            rp.setInvertSelection(false);
            learn_data = new Instances(Filter.useFilter(train_data, rp));
            rp = new RemovePercentage();
            rp.setInputFormat(train_data);
            rp.setPercentage(20);
            rp.setInvertSelection(true);
            val_data = new Instances(Filter.useFilter(train_data, rp));
        } else if(this.n_internal_folds == 1) {  // leave one out
            throw new Exception("not implemented yet!");
        } else {  // n-fold cross validation
            train_data = FitnessCalculator.betterStratifier(input_data, n_internal_folds + 1);  // 5 folds of interval CV + 1 for validation
            val_data = train_data.testCV(n_internal_folds + 1, 1);
            learn_data = FitnessCalculator.betterStratifier(train_data.trainCV(n_internal_folds + 1, 1), n_internal_folds);
        }

        if(this.pbilLogger != null) {
            pbilLogger.setDatasets(null, learn_data, val_data, null);
        }
        FitnessCalculator fc = new FitnessCalculator(this.n_internal_folds, learn_data, val_data);
        this.earlyStop = new EarlyStop(this.early_stop_generations, 0);

        this.currentGenBest = new BaselineIndividual();
        Fitness baselineFitness = fc.evaluateEnsemble(seed, this.currentGenBest, null, true);
        this.currentGenBest.setFitness(baselineFitness);

        this.earlyStop.update(-1, this.currentGenBest, this.currentGenBest.getFitness().getValQuality());
        this.overallBest = this.earlyStop.getBestIndividual();

        int to_select = 0;  // no carry over for first generation
        int to_sample = this.n_individuals;  // samples whole population

        Integer[] sortedIndices = new Integer[0];
        Individual[] population = new Individual[this.n_individuals];

        for(int g = 0; g < this.n_generations; g++) {
            Individual[] sampled = dn.gibbsSampleAndAssignFitness(
                    this.currentGenBest.getCharacteristics(), to_sample, fc, this.seed, start, this.timeout
            );
            if((sampled.length < to_sample)) {
                break;
            }

            // removes old individuals
            int pop_counter = 0;

            // selects old population - all sampled individuals on first generation, and 1 from then on
            for(int i = 0; i < to_select; i++) {
                population[pop_counter] = population[sortedIndices[i]];
                pop_counter += 1;
            }

            // adds new population
            for(int i = 0; i < sampled.length; i++) {
                population[pop_counter] = sampled[i];
                pop_counter += 1;
            }

            sortedIndices = fc.getSortedIndices(population);

            to_select = 1;
            to_sample = this.n_individuals - 1;

            // current gen best is the individual which presents the best fitness in
            // learning set (if using n-fold cross-validation, including leave-one-out) or validation set (holdout)
            this.currentGenBest = population[sortedIndices[0]];
            this.currentGenBest.setFitness(fc.getEnsembleValidationFitness(this.currentGenBest));

            t2 = LocalDateTime.now();
            if(this.pbilLogger != null) {
                this.pbilLogger.log_and_print(
                        sortedIndices, population, this.overallBest, this.currentGenBest, this.dn, t1, t2
                );
            }
            t1 = t2;

            // early stop always use validation fitness
            this.earlyStop.update(g, this.currentGenBest, this.currentGenBest.getFitness().getValQuality());
            this.overallBest = this.earlyStop.getBestIndividual();

            boolean overTime = (this.timeout > 0) && ((int)start.until(t1, ChronoUnit.SECONDS) > this.timeout);

            if(this.earlyStop.isStopping(g) || overTime) {
                break;
            }
            this.dn.update(population, sortedIndices, this.selection_share, g);
        }

        this.trainReturnIndividuals(train_data);

        this.fitted = true;
    }

    /**
     * Trains currentGenBest and overallBest individuals on the whole training set.
     *
     * @param train_data Training data, as provided to EDNEL.
     * @throws Exception If anything wrong happens.
     */
    protected void trainReturnIndividuals(final Instances train_data) throws Exception {
        Thread buildCurrentGenBest = new Thread() {
            @Override
            public synchronized void start() {
                try {
                    currentGenBest.setTimeoutIndividual(null);
                    currentGenBest.buildClassifier(train_data);
                } catch(Exception e) {
                    // does nothing
                }

            }
        };
        if(this.currentGenBest != this.overallBest) {
            buildCurrentGenBest.start();
        }

        this.overallBest.setTimeoutIndividual(null);
        this.overallBest.buildClassifier(train_data);

        buildCurrentGenBest.join();
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

    public boolean isLogging() {
        return this.pbilLogger != null;
    }

    public void writeParametersToFile(
            String write_path, String dataset_experiment_path, String dataset_name, boolean bestUsesOverall
    ) throws IOException {

        HashMap<String, String> obj = new HashMap<>();

        obj.put("dataset_experiment_path", dataset_experiment_path);
        obj.put("dataset_name", dataset_name);
        obj.put("selection_share", String.valueOf(selection_share));
        obj.put("n_individuals", String.valueOf(n_individuals));
        obj.put("n_generations", String.valueOf(n_generations));
        obj.put("timeout", String.valueOf(timeout));
        obj.put("timeout_individual", String.valueOf(timeout_individual));
        obj.put("burn_in", String.valueOf(burn_in));
        obj.put("thinning_factor", String.valueOf(thinning_factor));
        obj.put("no_cycles", String.valueOf(no_cycles));
        obj.put("learning_rate", String.valueOf(learning_rate));
        obj.put("early_stop_generations", String.valueOf(early_stop_generations));
        obj.put("max_parents", String.valueOf(max_parents));
        obj.put("delay_structure_learning", String.valueOf(delay_structure_learning));
        obj.put("individual", bestUsesOverall? "overall" : "last");


        FileWriter fw = new FileWriter(write_path);

        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        fw.write(converter.toJson(obj));
        fw.flush();
        fw.close();
    }
}

