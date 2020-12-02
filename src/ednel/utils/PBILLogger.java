package ednel.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.network.DependencyNetwork;
import ednel.network.variables.AbstractVariable;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import javax.annotation.processing.FilerException;
import java.io.*;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class PBILLogger {

    private final String dataset_name;
    protected final String dataset_overall_path;
    protected final String this_run_path;
    protected final String dataset_metadata_path;

    protected HashMap<String,  // generation
            HashMap<String, // child variable
                    HashMap<String,  // parents (e.g. "PART=true,J48_pruning=J48_unpruned")
                            Double  // probability of combination
                    >
            >
    > pastDependencyStructures = null;

    protected Individual last;
    protected Individual overall;

    protected final int n_sample;
    protected final int n_fold;
    protected final int n_individuals;
    protected final boolean log;

    protected int curGen;

    protected ArrayList<Double> minFitness;
    protected ArrayList<Double> maxFitness;
    protected ArrayList<Double> medianFitness;
    protected ArrayList<Integer> discardedIndividuals;
    protected ArrayList<Integer> lapTimes;
    protected ArrayList<Integer> nevals;
    protected ArrayList<Integer> dnConnections;

    protected HashMap<String, String> pastPopulations = null;

    private Instances learn_data;
    private Instances val_data;
    private ArrayList<Double> currentGenBestValFitness;

    /** Column names as displayed in stdout during evolution */
    public static final String[] column_names = {"dataset", " gen", "nevals", "min", "median", "max",
            "validation fitness", "lap time (s)", "discarded samples", "dn connections"};

    /** Width of each column displayed in the console */
    public static final int[] column_widths = {30, 4, 6, 10, 10, 10, 20, 12, 20, 14};

    protected static final String[] metricsToCollect = new String[]{
            "avgCost",
            "getClassPriors",
            "confusionMatrix",
            "correct",
            "errorRate",
            "incorrect",
            "kappa",
            "KBInformation",
            "KBMeanInformation",
            "KBRelativeInformation",
            "meanAbsoluteError",
            "meanPriorAbsoluteError",
            "numInstances",
            "pctCorrect",
            "pctIncorrect",
            "pctUnclassified",
            "relativeAbsoluteError",
            "rootMeanPriorSquaredError",
            "rootMeanSquaredError",
            "rootRelativeSquaredError",
            "SFEntropyGain",
            "SFMeanEntropyGain",
            "SFMeanPriorEntropy",
            "SFMeanSchemeEntropy",
            "SFPriorEntropy",
            "sizeOfPredictedRegions",
            "totalCost",
            "unclassified",
            "weightedAreaUnderPRC",
            "weightedAreaUnderROC",
            "weightedFMeasure",
            "weightedFalseNegativeRate",
            "weightedFalsePositiveRate",
            "weightedMatthewsCorrelation",
            "weightedPrecision",
            "weightedRecall",
            "weightedTrueNegativeRate",
            "weightedTruePositiveRate",
            "unweightedMacroFmeasure",
            "unweightedMicroFmeasure"
    };
    private ArrayList<Double> dnMeanHeuristics;
    private ArrayList<String> samplingOrders;

    /**
     *
     * @param dataset_metadata_path
     * @param n_individuals
     * @param n_sample
     * @param n_fold
     * @param log Whether to implement more aggressive logging capabilities. May have a great impact in performance.
     */
    public PBILLogger(
            String dataset_name, String dataset_metadata_path,
            int n_individuals, int n_generations, int n_sample, int n_fold, boolean log
    ) {
        this.dataset_name = dataset_name;
        this.n_sample = n_sample;
        this.n_fold = n_fold;

        this.learn_data = null;
        this.val_data = null;
        this.currentGenBestValFitness = null;

        this.log = log;
        if(this.log) {
            this.pastPopulations = new HashMap<>(n_generations * n_individuals * 50);
            this.pastDependencyStructures = new HashMap<>(n_generations * 50);
        }

        this.dataset_metadata_path = dataset_metadata_path;
        this.dataset_overall_path = String.format(
                "%s%soverall%stest_sample-%02d_fold-%02d.csv",
                dataset_metadata_path, File.separator, File.separator, n_sample, n_fold
        );
        this.this_run_path = String.format(
                "%s%ssample_%02d_fold_%02d",
                this.dataset_metadata_path, File.separator, this.n_sample, this.n_fold
        );

        this.n_individuals = n_individuals;

        this.minFitness = new ArrayList<>();
        this.medianFitness = new ArrayList<>();
        this.maxFitness = new ArrayList<>();
        this.discardedIndividuals = new ArrayList<>();
        this.lapTimes = new ArrayList<>();
        this.nevals = new ArrayList<>();
        this.dnConnections = new ArrayList<>();
        this.dnMeanHeuristics = new ArrayList<>();
        this.samplingOrders = new ArrayList<>();

        this.curGen = 0;
    }

    public PBILLogger(
            String dataset_name, String dataset_metadata_path,
            int n_individuals, int n_generations, int n_sample, int n_fold, boolean log,
            Instances learn_data, Instances val_data
    ) {
        this(dataset_name, dataset_metadata_path, n_individuals, n_generations, n_sample, n_fold, log);
        this.learn_data = learn_data;
        this.val_data = val_data;
        this.currentGenBestValFitness = new ArrayList<>();
    }

    public static double getMedianFitness(Individual[] population, Integer[] sortedIndices) {
        double medianFitness;
        if((population.length % 2) == 0) {
            int ind0 = (population.length / 2) - 1, ind1 = (population.length / 2);
            medianFitness = (
                    population[sortedIndices[ind0]].getFitness().getLearnQuality() +
                            population[sortedIndices[ind1]].getFitness().getLearnQuality())
                    / 2;
        } else {
            medianFitness = population[sortedIndices[population.length / 2]].getFitness().getLearnQuality();
        }
        return medianFitness;
    }

    public void log(Integer[] sortedIndices,
                    Individual[] population, Individual overall, Individual last,
                    DependencyNetwork dn, LocalDateTime t1, LocalDateTime t2
    ) throws Exception {
        this.overall = overall;
        this.last = last;
        this.nevals.add(dn.getCurrentGenEvals());
        this.discardedIndividuals.add(dn.getCurrentGenDiscardedIndividuals());
        this.dnConnections.add(dn.getCurrentGenConnections());
        this.dnMeanHeuristics.add(dn.getCurrentGenMeanHeuristic());

        this.currentGenBestValFitness.add(last.getFitness().getValQuality());

        this.lapTimes.add((int)t1.until(t2, ChronoUnit.SECONDS));

        this.logPopulation(sortedIndices, population);
        this.logDependencyNetworkStructureAndProbabilities(dn);

        ArrayList<String> samplingOrder = dn.getSamplingOrder();
        StringBuilder sb = new StringBuilder("");
        for(String var : samplingOrder) {
            sb.append(var).append(",");
        }

        String so_str = sb.toString();
        so_str = "\"" + so_str.substring(0, so_str.length() - 1) + "\"";
        this.samplingOrders.add(so_str);

        this.curGen += 1;
    }

    /**
     * Convenience method for log and print functions.
     *
     * @param sortedIndices
     * @param population
     * @param overall
     * @param last
     * @param dn
     * @param t1
     * @param t2
     * @throws Exception
     */
    public void log_and_print(Integer[] sortedIndices,
                    Individual[] population, Individual overall, Individual last,
                    DependencyNetwork dn, LocalDateTime t1, LocalDateTime t2
    ) throws Exception {
        this.log(sortedIndices, population, overall, last, dn, t1, t2);
        this.print();
    }


    private void logPopulation(Integer[] sortedIndices, Individual[] population) {
        if(this.log) {
            for(int i = 0; i < population.length; i++) {
                for(String characteristic : population[i].getCharacteristics().keySet()) {
                    this.pastPopulations.put(String.format(
                            "gen_%03d_ind_%03d_%s", this.curGen, i, characteristic
                    ), population[i].getCharacteristics().get(characteristic));
                }
                this.pastPopulations.put(String.format(
                        "gen_%03d_ind_%03d_%s", this.curGen, i, "fitness"
                ), String.valueOf(population[i].getFitness().getLearnQuality()));
            }
        }
        this.minFitness.add(population[sortedIndices[population.length - 1]].getFitness().getLearnQuality());
        this.maxFitness.add(population[sortedIndices[0]].getFitness().getLearnQuality());

        this.medianFitness.add(PBILLogger.getMedianFitness(population, sortedIndices));
    }

    private void logDependencyNetworkStructureAndProbabilities(DependencyNetwork dn) {
        if(this.log) {
            HashMap<String, AbstractVariable> variables = dn.getVariables();
            Object[] variableNames = variables.keySet().toArray();

            HashMap<String, HashMap<String, Double>> thisGeneration = new HashMap<>();

            for(Object variableName : variableNames) {
                thisGeneration.put(
                    (String)variableName,
                    variables.get(variableName).getTablePrettyPrint()
                );
            }
            this.pastDependencyStructures.put(
                String.format(Locale.US, "%03d", this.curGen),
                thisGeneration
            );
        }
    }

    private String getEvaluationLineForClassifier(String name, Evaluation evaluation) throws Exception {

        String line = name;

        Method[] overallMethods = null;
        HashMap<String, Method> overallMethodDict = null;

        if(evaluation != null) {
            overallMethods = evaluation.getClass().getMethods();
            overallMethodDict = new HashMap<>(overallMethods.length);

            for(Method method : overallMethods) {
                overallMethodDict.put(method.getName(), method);
            }
        }

        for(String methodName : metricsToCollect) {
            if(evaluation == null) {
                line += ",";
            } else {
                Object res = overallMethodDict.get(methodName).invoke(evaluation);
                if(res.getClass().isArray()) {
                    try {
                        double[] doublyOverall = ((double[])res);

                        line += ",\"np.array([";
                        for(int k = 0; k < doublyOverall.length; k++) {
                            line += doublyOverall[k] + ",";
                        }
                        line = line.substring(0, line.lastIndexOf(",")) + "], dtype=np.float64)\"";
                    } catch(ClassCastException e) {
                        double[][] doublyOverall = ((double[][])res);

                        line += ",\"np.array([[";
                        for(int j = 0; j < doublyOverall.length; j++) {
                            for(int k = 0; k < doublyOverall[j].length; k++) {
                                line += doublyOverall[j][k] + ",";
                            }
                            line = line.substring(0, line.lastIndexOf(",")) + "],[";
                        }
                        line = line.substring(0, line.lastIndexOf(",[")) + "], dtype=np.float64)\"";
                    } catch(Exception e) {
                        throw(e);
                    }
                } else {
                    line += "," + res;
                }
            }
        }
        return line + "," + (evaluation != null? FitnessCalculator.getUnweightedAreaUnderROC(evaluation) : "") + "\n";
    }

    private Double[] evaluationsToFile(
            HashMap<String, Individual> individuals, Instances train_data, Instances test_data) throws Exception {

        BufferedWriter bw = new BufferedWriter(new FileWriter(this.dataset_overall_path));
        Double[] fitnesses = new Double [individuals.size()];

        // writes header
        String header = "algorithm";
        for(String methodName : metricsToCollect) {
            header += "," + methodName;
        }
        bw.write(header + "," + "unweightedAreaUnderRoc" + "\n");

        Evaluation evaluation;

        Object[] indNames = individuals.keySet().toArray();

        for(int i = 0; i < indNames.length; i++) {
            String indName = (String)indNames[i];
            evaluation = new Evaluation(train_data);
            evaluation.evaluateModel(individuals.get(indName), test_data);

            bw.write(this.getEvaluationLineForClassifier(indName, evaluation));

            HashMap<String, AbstractClassifier> indClassifiers = individuals.get(indName).getClassifiers();

            for(String key: indClassifiers.keySet()) {
                if(!String.valueOf(indClassifiers.get(key)).equals("null")) {
                    evaluation = new Evaluation(train_data);
                    evaluation.evaluateModel(indClassifiers.get(key), test_data);

                    fitnesses[i] = FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
                } else {
                    fitnesses[i] = 0.0;
                    evaluation = null;
                }
                bw.write(this.getEvaluationLineForClassifier(indName + "-" + key, evaluation));
            }
        }
        bw.close();
        return fitnesses;
    }

    public void toFile(
            DependencyNetwork dn, HashMap<String, Individual> individuals, Instances train_data, Instances test_data
    ) throws Exception {

        Double[] fitnesses = evaluationsToFile(individuals, train_data, test_data);
        PBILLogger.createFolder(this_run_path);
        individualsCharacteristicsToFile(individuals, fitnesses);
        individualsClassifiersToFile(individuals);
        loggerDataToFile();
        dependencyNetworkStructureToFile(dn);
    }

    private void loggerDataToFile() throws Exception {
        if(this.log) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    this_run_path + File.separator + "loggerData.csv"
            ));

            // writes header
            bw.write(
                    "gen,nevals,min,median,max," + (this.val_data != null? "currentGenBestValFitness," : "") +
                    "lap time (seconds),discarded individuals (including burn-in),GM connections,GM mean heuristic," +
                    "sampling order\n");

            for(int i = 0; i < this.curGen; i++) {
                bw.write(String.format(
                        Locale.US,
                        "%d,%d,%.8f,%.8f,%.8f,"  + (this.val_data != null? "%.8f," : "%s") +
                                "%04d,%04d,%04d,%.8f,%s\n",
                        i,
                        this.nevals.get(i),
                        this.minFitness.get(i),
                        this.medianFitness.get(i),
                        this.maxFitness.get(i),
                        this.val_data != null? this.currentGenBestValFitness.get(i) : "",
                        this.lapTimes.get(i),
                        this.discardedIndividuals.get(i),
                        this.dnConnections.get(i),
                        this.dnMeanHeuristics.get(i),
                        this.samplingOrders.get(i)
                ));
            }
            bw.close();
        }
    }

    private void dependencyNetworkStructureToFile(DependencyNetwork dn) throws IOException {
        //BufferedWriter bw = new BufferedWriter();
        if(this.log) {
            // writes to file
            String sourceFile = this_run_path + File.separator + "dependency_network_structure.json";

            File inFile = new File(sourceFile);

            FileWriter fw = new FileWriter(inFile);
            Gson converter = new GsonBuilder().setPrettyPrinting().create();
            fw.write(converter.toJson(this.pastDependencyStructures));
            fw.flush();
            fw.close();

            // now zips
            this.zipFile(sourceFile, this_run_path + File.separator + "dependency_network_structure.zip");
            inFile.delete();
        }
    }

    /**
     * Zips any file.
     *
     * @param inFile Path to file as present in operating system.
     * @param outFile Path and name of zip file to be written.
     * @throws IOException
     */
    private void zipFile(String inFile, String outFile) throws IOException {
        FileOutputStream fos = new FileOutputStream(outFile);
        ZipOutputStream zipOut = new ZipOutputStream(fos);
        File fileToZip = new File(inFile);
        FileInputStream fis = new FileInputStream(fileToZip);
        ZipEntry zipEntry = new ZipEntry(fileToZip.getName());
        zipOut.putNextEntry(zipEntry);
        byte[] bytes = new byte[1024];
        int length;
        while((length = fis.read(bytes)) >= 0) {
            zipOut.write(bytes, 0, length);
        }
        zipOut.close();
        fis.close();
        fos.close();
    }

    private void individualsClassifiersToFile(HashMap<String, Individual> individuals) throws Exception {
        for(String indName : individuals.keySet()) {
            String destination_path = this_run_path + File.separator + indName;

            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    destination_path + "_classifiers.md"
            ));

            bw.write(individuals.get(indName).toString());
            individuals.get(indName).treeModelsToFiles(destination_path);
            bw.close();
        }
    }

    private String getCharacteristicsLineForIndividual(
            Object[] order, String indName, HashMap<String, String> characteristics, Double aucs) {

        String line = indName;
        for(Object ch : order) {
            line += "," + characteristics.get(ch);
        }
        line += ",null," + aucs;
        return line + "\n";
    }

    private void individualsCharacteristicsToFile(
            HashMap<String, Individual> individuals, Double[] fitnesses) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(
                this_run_path + File.separator + "characteristics.csv"
        ));

        // writes header, saves order of columns
        Object[] order = individuals.get(individuals.keySet().toArray()[0]).getCharacteristics().keySet().toArray();
        String header = "individual_name";
        for(int i = 0; i < order.length; i++) {
            header += "," + order[i];
        }
        bw.write(header + ",fitness,test_auc\n");

        Object[] indNames = individuals.keySet().toArray();

        for(int i = 0; i < indNames.length; i++) {
            String indName = (String)indNames[i];
            HashMap<String, String> characteristics = individuals.get(indName).getCharacteristics();
            bw.write(this.getCharacteristicsLineForIndividual(order, indName, characteristics, fitnesses[i]));
        }
        if(this.log) {
            for(int i = 0; i < this.curGen; i++) {
                for(int j = 0; j < this.n_individuals; j++) {
                    String line = String.format("gen_%03d_ind_%03d", i, j);
                    for(Object ch : order) {
                        line += "," + pastPopulations.get(String.format("gen_%03d_ind_%03d_%s", i, j, ch));
                    }
                    line += "," + pastPopulations.get(String.format("gen_%03d_ind_%03d_%s", i, j, "fitness")) + ",null";
                    bw.write(line + "\n");
                }
            }
        }
        bw.close();
    }

    public static void metadata_path_start(String str_time, HashMap<String, String> options) throws IOException {
        String[] dataset_names = options.get("datasets_names").split(",");
        String metadata_path = options.get("metadata_path");

        // create one folder for each dataset
        PBILLogger.createFolder(metadata_path + File.separator + str_time);
        for(String dataset : dataset_names) {
            String partial = metadata_path + File.separator + str_time + File.separator + dataset;

            PBILLogger.createFolder(partial);
            PBILLogger.createFolder(partial + File.separator + "overall");
        }

        HashMap<String, String> obj = new HashMap<>();
        for(String parameter : options.keySet()) {
            obj.put(parameter, options.get(parameter));
        }

        FileWriter fw = new FileWriter(
                metadata_path + File.separator + str_time + File.separator + "parameters.json"
        );

        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        fw.write(converter.toJson(obj));
        fw.flush();
        fw.close();
    }

    private static void createFolder(String path) throws FilerException {
        File file = new File(path);
        boolean successful = file.mkdir();
        if(!successful) {
            throw new FilerException("could not create directory " + path);
        }
    }

    public void print() {
        if(this.curGen == 1) {
            for(int i = 0; i < PBILLogger.column_names.length; i++) {
                int n_padding = PBILLogger.column_widths[i] - PBILLogger.column_names[i].length();
                String padding = new String(new char[n_padding]).replace("\0", " ");

                System.out.print(String.format("%s%s  ", PBILLogger.column_names[i], padding));
            }
            System.out.println();
        }
        String[] data = {
                this.dataset_name,
                String.format("%4d", this.curGen - 1),
                String.format("%4d", this.nevals.get(this.curGen - 1)),
                String.format("%01.6f", this.minFitness.get(this.curGen - 1)),
                String.format("%01.6f", this.medianFitness.get(this.curGen - 1)),
                String.format("%01.6f", this.maxFitness.get(this.curGen - 1)),
                String.format("%01.6f", this.currentGenBestValFitness.get(this.curGen - 1)),
                String.format("%6d", this.lapTimes.get(this.curGen - 1)),
                String.format("%6d", this.discardedIndividuals.get(this.curGen - 1)),
                String.format("%3d", this.dnConnections.get(this.curGen - 1))
        };
        for(int i = 0; i < data.length; i++) {
            int n_padding = PBILLogger.column_widths[i] - data[i].length();
            String padding = new String(new char[n_padding]).replace("\0", " ");
            System.out.print(String.format("%s%s  ", data[i], padding));
        }
        System.out.println();
    }

    public void setDatasets(Instances train_data, Instances test_data) {
        this.learn_data = train_data;
        this.val_data = test_data;
        this.currentGenBestValFitness = new ArrayList<>();
    }

    public String getThisRunPath() {
        return this_run_path;
    }

    public String getDatasetMetadataPath() {
        return this.dataset_metadata_path;
    }
}

