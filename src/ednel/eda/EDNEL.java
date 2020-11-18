package ednel.eda;

import ednel.eda.individual.BaselineIndividual;
import ednel.eda.individual.Fitness;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.eda.stoppers.EarlyStop;
import ednel.network.DependencyNetwork;
import ednel.utils.PBILLogger;
import ednel.utils.comparators.Argsorter;
import org.apache.commons.math3.random.MersenneTwister;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
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

    protected Individual[] bestGenInd;

    protected Individual currentGenBest;
    protected Individual overallBest;

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

        this.earlyStop = new EarlyStop(this.early_stop_generations, this.early_stop_tolerance, 0);

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

    /**
     * Checks whether a dominates b (i.e. a is a better solution, in all criteria, than b).
     *
     * @param a first solution
     * @param b second solution
     * @return -1 if b dominates a, +1 if a dominates b, and 0 if there is no dominance
     */
    private int a_dominates_b(Fitness a, Fitness b) {
        boolean a_dominates = ((a.getLearnQuality() >= b.getLearnQuality()) && (a.getSize() <= b.getSize())) &&
                ((a.getLearnQuality() > b.getLearnQuality()) || (a.getSize() < b.getSize()));
        boolean b_dominates = ((b.getLearnQuality() >= a.getLearnQuality()) && (b.getSize() <= a.getSize())) &&
                ((b.getLearnQuality() > a.getLearnQuality()) || (b.getSize() < a.getSize()));

        if(a_dominates) {
            if(b_dominates) {
                return 0;
            } else {
                return 1;
            }
        } else if(b_dominates) {
            return -1;
        } else {
            return 0;
        }
    }

    private Integer[] paretoSort(Individual[] population) {
        HashMap<Integer, ArrayList<Integer>> dominates = new HashMap<>();
        HashMap<Integer, Integer> dominated = new HashMap<>();

        ArrayList<Integer> cur_front = new ArrayList<>();

        Integer[] sortedIndices = new Integer[population.length];
        int counter = 0;

        ArrayList<ArrayList<Integer>> fronts = new ArrayList<>();

        for(int i = 0; i < population.length; i++) {
            dominated.put(i, 0);
            dominates.put(i, new ArrayList<>());
        }

        for(int i = 0; i < population.length; i++) {
            for(int j = i + 1; j < population.length; j++) {
                int res = a_dominates_b(population[i].getFitness(), population[j].getFitness());
                if(res == 1) {
                    dominated.put(j, dominated.getOrDefault(j, 0) + 1);  // signals that j is dominated by one solution

                    ArrayList<Integer> thisDominates = dominates.getOrDefault(i, new ArrayList<Integer>());
                    thisDominates.add(j);
                    dominates.put(i, thisDominates);  // add j to the list of dominated solutions by i
                } else if(res == -1) {
                    dominated.put(i, dominated.getOrDefault(i, 0) + 1);  // signals that i is dominated by one solution

                    ArrayList<Integer> thisDominates = dominates.getOrDefault(j, new ArrayList<Integer>());
                    thisDominates.add(i);
                    dominates.put(j, thisDominates);  // add i to the list of dominated solutions by j
                }
            }
            if(dominated.get(i) == 0) {
                cur_front.add(i);
            }
        }

        while(cur_front.size() != 0) {
            ArrayList<Integer> some_set = new ArrayList<>();

            for(Integer master : cur_front) {
                for(Integer slave : dominates.get(master)) {
                    dominated.put(slave, dominated.get(slave) - 1);
                    if(dominated.get(slave) == 0) {
                        some_set.add(slave);
                    }
                }
            }
            Double[] cur_front_double = new Double[cur_front.size()];
            for(int i = 0; i < cur_front.size(); i++) {
                cur_front_double[i] = population[cur_front.get(i)].getFitness().getLearnQuality();
            }
            Integer[] local_indices = Argsorter.decrescent_argsort(cur_front_double);

            for(Integer item : local_indices) {
                sortedIndices[counter] = cur_front.get(item);
                counter += 1;
            }

            fronts.add((ArrayList<Integer>)cur_front.clone());
            cur_front = some_set;
        }
        return sortedIndices;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        LocalDateTime start = LocalDateTime.now();
        LocalDateTime t1 = LocalDateTime.now(), t2;

        this.bestGenInd = new Individual[this.n_generations];  // TODO new code!

        data = FitnessCalculator.betterStratifier(data, 6);  // 5 folds of interval CV + 1 for validation
        Instances val_data = data.testCV(6, 1);
        Instances learn_data = data.trainCV(6, 1);
        learn_data = FitnessCalculator.betterStratifier(learn_data, 5);

        pbilLogger.setDatasets(learn_data, val_data);

        FitnessCalculator fc = new FitnessCalculator(5, learn_data, val_data);

        this.currentGenBest = new BaselineIndividual();
        this.overallBest = this.currentGenBest;

        int to_sample = this.n_individuals;
        int to_select = 0;

        Integer[] sortedIndices = new Integer[0];

        Individual[] population = new Individual[this.n_individuals];

        for(int g = 0; g < this.n_generations; g++) {
            Individual[] sampled = dn.gibbsSample(
                    this.currentGenBest.getCharacteristics(), to_sample, fc, this.seed
            );
            // removes old individuals
            int counter = 0;
            for(int i = 0; i < to_select; i++) {
                population[counter] = population[sortedIndices[i]];
                counter += 1;
            }
            // adds new population
            for(int i = 0; i < to_sample; i++) {
                population[counter] = sampled[i];
                counter += 1;
            }
            to_sample = (int)(this.selection_share * this.n_individuals);
            to_select = this.n_individuals - to_sample;

            // sortedIndices = paretoSort(population);
            sortedIndices = simpleSort(population);

            this.bestGenInd[g] = population[sortedIndices[0]];

//            Integer[] sortedIndices = Argsorter.decrescent_argsort(fitnesses);

            this.currentGenBest = population[sortedIndices[0]];

            t2 = LocalDateTime.now();
            if(this.pbilLogger != null) {
                this.pbilLogger.log_and_print(
                        sortedIndices, population, this.overallBest, this.currentGenBest, this.dn, t1, t2
                );
            }
            t1 = t2;

            this.earlyStop.update(g, this.currentGenBest);

            boolean overTime = (this.timeout > 0) && ((int)start.until(t1, ChronoUnit.SECONDS) > this.timeout);

            if(this.earlyStop.isStopping(g) || overTime) {
                break;
            }
            this.dn.update(population, sortedIndices, this.selection_share, g);
        }

        this.overallBest = this.earlyStop.getLastBestIndividual();  // TODO is this correct?

        this.overallBest.buildClassifier(data);
        this.currentGenBest.buildClassifier(data);

        this.fitted = true;
    }

    private Integer[] simpleSort(Individual[] population) {
        Double[] cur_front_double = new Double[population.length];
        for(int i = 0; i < population.length; i++) {
            cur_front_double[i] = population[i].getFitness().getLearnQuality();
        }
        return Argsorter.decrescent_argsort(cur_front_double);
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

