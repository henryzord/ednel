package ednel.utils;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import ednel.eda.individual.FitnessCalculator;
import ednel.eda.individual.Individual;
import ednel.network.DependencyNetwork;
import ednel.network.variables.AbstractVariable;
import guru.nidi.graphviz.attribute.MapAttributes;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import javax.annotation.processing.FilerException;
import java.io.*;
import java.lang.reflect.Method;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public class PBILLogger {

    private final String dataset_name;
    protected final String dataset_overall_path;
    protected final String dataset_thisrun_path;
    protected HashMap<String,  // generation
            HashMap<String, // child variable
                    HashMap<String,  // parents (e.g. "PART=true,J48_pruning=J48_unpruned")
                            Double  // probability of combination
                    >
            >
    > pastDependencyStructures = null;
    protected final String dataset_metadata_path;
    protected Individual overall;
    protected Individual last;
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

    private Instances train_data;
    private Instances test_data;
    private ArrayList<Double> currentGenBestTestFitness;

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

        this.train_data = null;
        this.test_data = null;
        this.currentGenBestTestFitness = null;

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
        this.dataset_thisrun_path = String.format(
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
            Instances train_data, Instances test_data
    ) {
        this(dataset_name, dataset_metadata_path, n_individuals, n_generations, n_sample, n_fold, log);
        this.train_data = train_data;
        this.test_data = test_data;
        this.currentGenBestTestFitness = new ArrayList<>();
    }

    public static double getMedianFitness(Double[] fitnesses, Integer[] sortedIndices) {
        double medianFitness;
        if((fitnesses.length % 2) == 0) {
            int ind0 = (fitnesses.length / 2) - 1, ind1 = (fitnesses.length / 2);
            medianFitness = (fitnesses[sortedIndices[ind0]] + fitnesses[sortedIndices[ind1]]) / 2;
        } else {
            medianFitness = fitnesses[sortedIndices[fitnesses.length / 2]];
        }
        return medianFitness;
    }

    public void log(Double[] fitnesses, Integer[] sortedIndices,
                    Individual[] population, Individual overall, Individual last,
                    DependencyNetwork dn, LocalDateTime t1, LocalDateTime t2
    ) throws Exception {
        this.overall = overall;
        this.last = last;
        this.nevals.add(dn.getCurrentGenEvals());
        this.discardedIndividuals.add(dn.getCurrentGenDiscardedIndividuals());
        this.dnConnections.add(dn.getCurrentGenConnections());
        this.dnMeanHeuristics.add(dn.getCurrentGenMeanHeuristic());

        if(this.test_data != null) {
            last.buildClassifier(train_data);
            this.currentGenBestTestFitness.add(FitnessCalculator.getUnweightedAreaUnderROC(train_data, test_data, last));
        }

        this.lapTimes.add((int)t1.until(t2, ChronoUnit.SECONDS));

        this.logPopulation(fitnesses, sortedIndices, population);
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
     * @param fitnesses
     * @param sortedIndices
     * @param population
     * @param overall
     * @param last
     * @param dn
     * @param t1
     * @param t2
     * @throws Exception
     */
    public void log_and_print(Double[] fitnesses, Integer[] sortedIndices,
                    Individual[] population, Individual overall, Individual last,
                    DependencyNetwork dn, LocalDateTime t1, LocalDateTime t2
    ) throws Exception {
        this.log(fitnesses, sortedIndices, population, overall, last, dn, t1, t2);
        this.print();
    }


    private void logPopulation(Double[] fitnesses, Integer[] sortedIndices, Individual[] population) {
        if(this.log) {
            for(int i = 0; i < population.length; i++) {
                for(String characteristic : population[i].getCharacteristics().keySet()) {
                    this.pastPopulations.put(String.format(
                            "gen_%03d_ind_%03d_%s", this.curGen, i, characteristic
                    ), population[i].getCharacteristics().get(characteristic));
                }
                this.pastPopulations.put(String.format(
                        "gen_%03d_ind_%03d_%s", this.curGen, i, "fitness"
                ), String.valueOf(fitnesses[i]));
            }
        }
        this.minFitness.add(fitnesses[sortedIndices[fitnesses.length - 1]]);
        this.maxFitness.add(fitnesses[sortedIndices[0]]);

        this.medianFitness.add(PBILLogger.getMedianFitness(fitnesses, sortedIndices));
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
        PBILLogger.createFolder(dataset_thisrun_path);
        individualsCharacteristicsToFile(individuals, fitnesses);
        individualsClassifiersToFile(individuals);
        loggerDataToFile();
        dependencyNetworkStructureToFile(dn);
    }

    private void loggerDataToFile() throws Exception {
        if(this.log) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    dataset_thisrun_path + File.separator + "loggerData.csv"
            ));

            // writes header
            bw.write(
                    "gen,nevals,min,median,max," + (this.test_data != null? "currentGenBestTestFitness," : "") +
                    "lap time (seconds),discarded individuals (including burn-in),GM connections,GM mean heuristic," +
                    "sampling order\n");

            for(int i = 0; i < this.curGen; i++) {
                bw.write(String.format(
                        Locale.US,
                        "%d,%d,%.8f,%.8f,%.8f,"  + (this.test_data != null? "%.8f," : "%s") +
                                "%04d,%04d,%04d,%.8f,%s\n",
                        i,
                        this.nevals.get(i),
                        this.minFitness.get(i),
                        this.medianFitness.get(i),
                        this.maxFitness.get(i),
                        this.test_data != null? this.currentGenBestTestFitness.get(i) : "",
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
            String sourceFile = dataset_thisrun_path + File.separator + "dependency_network_structure.json";

            File inFile = new File(sourceFile);

            FileWriter fw = new FileWriter(inFile);
            Gson converter = new GsonBuilder().setPrettyPrinting().create();
            fw.write(converter.toJson(this.pastDependencyStructures));
            fw.flush();
            fw.close();

            // now zips
            this.zipFile(sourceFile, dataset_thisrun_path + File.separator + "dependency_network_structure.zip");
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
            String destination_path = dataset_thisrun_path + File.separator + indName;

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
                dataset_thisrun_path + File.separator + "characteristics.csv"
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

    public static void metadata_path_start(String str_time, CommandLine commandLine) throws ParseException, IOException {
        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        String metadata_path = commandLine.getOptionValue("metadata_path");

        // create one folder for each dataset
        PBILLogger.createFolder(metadata_path + File.separator + str_time);
        for(String dataset : dataset_names) {
            String partial = metadata_path + File.separator + str_time + File.separator + dataset;

            PBILLogger.createFolder(partial);
            PBILLogger.createFolder(partial + File.separator + "overall");
        }

        HashMap<String, String> obj = new HashMap<>();
        for(Option parameter : commandLine.getOptions()) {
            obj.put(parameter.getLongOpt(), parameter.getValue());
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
            System.out.println(
                    "Dataset\t\t\tGen\t\tnevals\t\tMin\t\t\tMedian\t\tMax\t\t\t" +
                            (this.test_data != null? "currentGenBest\t" : "") + "Lap time (s)\t" +
                            "Discarded Individuals\tDN Connections"
            );
            System.out.println(
                    (
                            this.test_data != null? String.join("", Collections.nCopies(18, "\t")) +
                                    "TestFitness" + String.join("", Collections.nCopies(6, "\t")) :
                                    String.join("", Collections.nCopies(22, "\t"))
                    ) + "(w/ burn-in)");
        }

        System.out.println(String.format(
                "%s\t\t\t%d\t\t%d\t\t\t%.8f\t%.8f\t%.8f\t" + (this.test_data != null? "%.8f\t\t" : "%s") +
                        "%04d\t\t\t%04d\t\t\t\t\t%04d",
                this.dataset_name,
                this.curGen - 1,
                this.nevals.get(this.curGen - 1),
                this.minFitness.get(this.curGen - 1),
                this.medianFitness.get(this.curGen - 1),
                this.maxFitness.get(this.curGen - 1),
                this.test_data != null? this.currentGenBestTestFitness.get(this.curGen - 1) : "",
                this.lapTimes.get(this.curGen - 1),
                this.discardedIndividuals.get(this.curGen - 1),
                this.dnConnections.get(this.curGen - 1)
        ));
    }

    public void setDatasets(Instances train_data, Instances test_data) {
        this.train_data = train_data;
        this.test_data = test_data;
        this.currentGenBestTestFitness = new ArrayList<>();
    }
}

