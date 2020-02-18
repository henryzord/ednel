package eda.ednel;

import dn.DependencyNetwork;
import dn.stoppers.EarlyStop;
import eda.individual.BaselineIndividual;
import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.random.MersenneTwister;
import org.json.simple.JSONObject;

import javax.annotation.processing.FilerException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import utils.ArrayIndexComparator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Arrays;
import java.util.HashMap;

public class EDNEL extends AbstractClassifier {

    protected int thining_factor;
    protected String options_path;
    protected String variables_path;
    protected String sampling_order_path;
    protected float learning_rate;
    protected float selection_share;
    protected int n_individuals;
    protected int n_generations;
    protected String output_path;
    protected Integer seed;

    protected boolean fitted;

    protected MersenneTwister mt;

    protected DependencyNetwork dn;

    protected Individual currentGenBest;
    protected Individual overallBest;
    protected Double currentGenFitness;
    protected Double overallFitness;

    protected EarlyStop earlyStop;

    public EDNEL(float learning_rate, float selection_share, int n_individuals, int n_generations, int thining_factor,
                 String variables_path, String options_path, String sampling_order_path, String output_path, Integer seed) throws Exception {

        this.learning_rate = learning_rate;
        this.selection_share = selection_share;
        this.n_individuals = n_individuals;
        this.n_generations = n_generations;
        this.thining_factor = thining_factor;

        this.output_path = output_path;
        this.variables_path = variables_path;
        this.options_path = options_path;
        this.sampling_order_path = sampling_order_path;

        this.overallFitness = -1.0;

        this.earlyStop = new EarlyStop();

        this.fitted = false;

        if(seed == null) {
            this.mt = new MersenneTwister();
            this.seed = mt.nextInt();
        } else {
            this.seed = seed;
            this.mt = new MersenneTwister(seed);
        }

        this.dn = new DependencyNetwork(
                mt, variables_path, options_path, sampling_order_path, this.learning_rate, this.n_generations
        );
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        FitnessCalculator fc = new FitnessCalculator(5, data, null);

        BaselineIndividual bi = new BaselineIndividual(data);
        HashMap<String, String> startPoint = bi.getCharacteristics();

        System.out.println(String.format("Gen\t\t\tnevals\t\tMin\t\t\t\t\tMedian\t\t\t\tMax"));
        for(int c = 0; c < this.n_generations; c++) {
            if(this.earlyStop.isStopping()) {
                break;
            }

            Individual[] population = dn.gibbsSample(startPoint, thining_factor, this.n_individuals, data);

            Double[][] fitnesses = fc.evaluateEnsembles(seed, population);

            ArrayIndexComparator comparator = new ArrayIndexComparator(fitnesses[0]);
            Integer[] sortedIndices = comparator.createIndexArray();
            Arrays.sort(sortedIndices, comparator);

            this.currentGenBest = population[sortedIndices[0]];
            this.currentGenFitness = fitnesses[0][sortedIndices[0]];

            if(this.currentGenFitness > this.overallFitness) {
                this.overallFitness = this.currentGenFitness;
                this.overallBest = this.currentGenBest;
            }

            this.earlyStop.update(c, this.currentGenFitness);

            System.out.println(String.format(
                    "%d\t\t\t%d\t\t\t%.8f\t\t\t%.8f\t\t\t%.8f",
                    c,
                    population.length,
                    fitnesses[0][sortedIndices[sortedIndices.length - 1]],
                    fitnesses[0][sortedIndices[sortedIndices.length / 2]],
                    fitnesses[0][sortedIndices[0]]
            ));

            this.dn.updateStructure(population, sortedIndices, this.selection_share);  // TODO update structure
            this.dn.updateProbabilities(population, sortedIndices, this.selection_share);
        }
        this.fitted = true;
    }

    protected static void createFolder(String path) throws FilerException {
        File file = new File(path);
        boolean successful = file.mkdir();
        if(!successful) {
            throw new FilerException("could not create directory " + path);
        }
    }

    public static void metadata_path_start(String str_time, CommandLine commandLine) throws ParseException, IOException {
        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        String metadata_path = commandLine.getOptionValue("metadata_path");

        // create one folder for each dataset
        EDNEL.createFolder(metadata_path + File.separator + str_time);
        for(String dataset : dataset_names) {
            EDNEL.createFolder(metadata_path + File.separator + str_time + File.separator + dataset);
        }

        JSONObject obj = new JSONObject();
        for(Option parameter : commandLine.getOptions()) {
            obj.put(parameter.getLongOpt(), parameter.getValue());
        }

        FileWriter fw = new FileWriter(metadata_path + File.separator + str_time + File.separator + "parameters.json");
        fw.write(obj.toJSONString());
        fw.flush();
    }

    public Individual getCurrentGenBest() {
        return currentGenBest;
    }

    public Individual getOverallBest() {
        return overallBest;
    }
}

