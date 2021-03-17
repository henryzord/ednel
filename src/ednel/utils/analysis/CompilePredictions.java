package ednel.utils.analysis;

import ednel.eda.individual.FitnessCalculator;
import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import smile.neighbor.lsh.Hash;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class CompilePredictions {

    protected Instances dummyDataset;
    protected HashMap<String, DummyClassifier> dummyClassifiers;

    /**
     * Compile predictions from classifiers, from a list of files
     *
     * @param files An ArrayList with .preds files, one for each fold (it can have any number of folds)
     * @param path_predictions Path to where .preds file are stored
     * @throws Exception
     */
    public CompilePredictions(ArrayList<String> files, String path_predictions) throws Exception {
        this(CompilePredictions.fromFilesToTable(files, path_predictions));
    }

    public CompilePredictions(ArrayList<String> lines) {
        // someInstances is the dataset that is built to emulate classification based on predictions of classifiers
        ArrayList<Instance> someInstances = new ArrayList<>();
        // set of un-repeated class values
        HashSet<Double> classValues = new HashSet<>();

        HashMap<String, HashMap<Integer, double[]>> linesPerClassifier = new HashMap<>();

        HashMap<Integer, String> clfsNames = new HashMap<>();

        int counter_instance = 0;
        for(String line : lines) {
            String[] splitted = line.replaceAll("\n", "").split(";");

            if(line.contains("classValue")) {
                clfsNames.clear();

                // j = 0 is always "classValue"
                for(int j = 1; j < splitted.length; j++) {
                    clfsNames.put(j, splitted[j]);
                    if(!linesPerClassifier.containsKey(splitted[j])) {
                        linesPerClassifier.put(splitted[j], new HashMap<>());
                    }
                }
            } else {
                double actualClass = Double.parseDouble(splitted[0]);

                classValues.add(actualClass);
                // counter is the predictive value (dummy), actualClass the class value
                someInstances.add(new DenseInstance(1, new double[]{(double)counter_instance, actualClass}));

                for(int j = 1; j < splitted.length; j++) {
                    double[] probs;
                    if(splitted[j].length() > 0) {
                        String[] strProbs = splitted[j].split(",");
                        probs = new double[strProbs.length];
                        for(int k = 0; k < strProbs.length; k++) {
                            probs[k] = Double.parseDouble(strProbs[k]);
                        }
                    } else {
                        // adds null because this classifier is not present for this fold.
                        // statistics for this classifier will not be present
                        probs = null;
                    }

                    HashMap<Integer, double[]> clfLines = linesPerClassifier.get(clfsNames.get(j));
                    clfLines.put(counter_instance, probs);
                    linesPerClassifier.put(clfsNames.get(j), clfLines);
                }
                counter_instance += 1;
            }
        }

        // remove classifiers that are not present in all folds
        ArrayList<String> all_classifiers_in_dict = new ArrayList<>();
        all_classifiers_in_dict.addAll(linesPerClassifier.keySet());
        for(String clfName : all_classifiers_in_dict) {
            int n_lines = linesPerClassifier.get(clfName).size();
            if(n_lines != counter_instance) {
                linesPerClassifier.remove(clfName);
            }
        }

        this.interpretData(someInstances, classValues, linesPerClassifier);
    }

    /**
     *
     * @param files An ArrayList with .preds files, one for each fold (it can have any number of folds)
     * @param path_predictions Path to where .preds file are stored
     * @return
     * @throws IOException
     */
    protected static ArrayList<String> fromFilesToTable(ArrayList<String> files, String path_predictions) throws IOException {
        ArrayList<String> lines = new ArrayList<>();
        for(String some_file : files) {
            BufferedReader br = new BufferedReader(new FileReader(String.format("%s%s%s",path_predictions,File.separator,some_file)));
            String line;
            while((line=br.readLine()) != null) {
                lines.add(line);
            }
        }
        return lines;
    }

    /**
     * Does for only one classifier, instead of multiple.
     *
     * @param probs Probabilities as produced by AbstractClassifier.distributionsForInstances
     * @param y Actual classes of instances in the dataset
     */
    public CompilePredictions(double[][] probs, double[] y, String clfName) {
        // some instances is the dataset that is built to emulate classification based on predictions of classifiers
        ArrayList<Instance> someInstances = new ArrayList<>();
        // set of un-repeated class values
        HashSet<Double> classValues = new HashSet<>();

        HashMap<Integer, double[]> probDistForSingleClassifier = new HashMap<>();

        int counter = 0;
        for(int i = 0; i < probs.length; i++) {
            classValues.add(y[i]);
            someInstances.add(new DenseInstance(1, new double[]{(double) counter, y[i]}));

            for(int j = 0; j < probs[i].length; j++) {
                probDistForSingleClassifier.put(counter, probs[i]);
            }
            counter += 1;
        }
        // list of dictionaries: each list entry is the dictionary for a classifier
        HashMap<String, HashMap<Integer, double[]>> linesPerClassifier = new HashMap<>();
        linesPerClassifier.put(clfName, probDistForSingleClassifier);

        this.interpretData(someInstances, classValues, linesPerClassifier);
    }

    /**
     * Builds DummyDataset and DummyClassifiers
     *
     * @param someInstances An ArrayList of instances, as read from files
     * @param classValues HashSet of class values (that is, unrepeated values)
     * @param linesPerClassifier A HashMap where the first key is the classifier name, and the value another HashMap,
     *                           this time with the instance ID as the key and the probability distribution as value
     */
    protected void interpretData(
            ArrayList<Instance> someInstances, HashSet<Double> classValues,
            HashMap<String, HashMap<Integer, double[]>> linesPerClassifier
    ) {
        // builds dummyDataset
        ArrayList<String> classValuesAL = new ArrayList<>();
        for(Double ob : classValues) {
            classValuesAL.add(ob.toString());
        }
        Collections.sort(classValuesAL);

        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(){{
            add(new Attribute("predictive"));
            add(new Attribute("class", classValuesAL));
        }};

        this.dummyDataset = new Instances("dummy dataset", attrInfo, someInstances.size());
        this.dummyDataset.addAll(someInstances);
        this.dummyDataset.setClassIndex(this.dummyDataset.numAttributes() - 1);

        // builds dummyClassifiers
        this.dummyClassifiers = new HashMap<>();
        for(String clfName : linesPerClassifier.keySet()) {
            this.dummyClassifiers.put(clfName, new DummyClassifier(linesPerClassifier.get(clfName)));
        }
    }

    public double getAUC(String clfName) throws Exception {
        Evaluation eval = new Evaluation(this.dummyDataset);
        eval.evaluateModel(this.dummyClassifiers.get(clfName), this.dummyDataset);
        return FitnessCalculator.getUnweightedAreaUnderROC(eval);
    }

    public HashMap<String, DummyClassifier> getDummyClassifiers() {
        return dummyClassifiers;
    }

    public Instances getDummyDataset() {
        return this.dummyDataset;
    }

    protected static CommandLine parseOptions(String[] args) throws ParseException {
        Options options = new Options();

        options.addOption(Option.builder()
                .required(true)
                .longOpt("path_predictions")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where .preds files (one for each fold) are stored. In this same folder," +
                        "a \"summary_1.csv\" file with all results will be written.")
                .build()
        );

        CommandLineParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    /**
     * Collects files from a folder.
     *
     * Generates a HashMap with the following hierarchy:
     * 1. a HashMap where the key is the sample number and the value another HashMap;
     * 2. A HashMap where the key is the classifier name, per file name, and the value an ArrayList
     * 3. An ArrayList with the file names
     *
     * @param path Path where .preds files are stored
     * @return A HashMap organizing files per sample and classifier name
     * @throws NullPointerException
     */
    protected static HashMap<Integer, HashMap<String, ArrayList<String>>> collectFiles(String path) throws NullPointerException {
        File folder = new File(path);
        String[] files = folder.list();

        if(files == null) {
            throw new NullPointerException("Path does not point to a valid directory.");
        }

        HashMap<Integer, HashMap<String, ArrayList<String>>> filePreds = new HashMap<>();

        for(String file : files) {
            if(file.contains(".preds")) {
                String indName = "someClassifier";
                int n_sample = Integer.parseInt(
                        file.substring(
                            file.indexOf("sample-") + "sample-".length(),
                            file.indexOf("sample-") + "sample-".length() + 2
                    )
                );

                HashMap<String, ArrayList<String>> local;
                if(!filePreds.containsKey(n_sample)) {
                    local = new HashMap<>();
                } else {
                    local = filePreds.get(n_sample);
                }

                ArrayList<String> thisList;
                if(!local.containsKey(indName)) {
                    thisList = new ArrayList<>();
                } else {
                    thisList = local.get(indName);
                }
                thisList.add(file);
                local.put(indName, thisList);
                filePreds.put(n_sample, local);
            }
        }
        return filePreds;
    }

    public static void main(String[] args) throws Exception {
        CommandLine cmd = CompilePredictions.parseOptions(args);

        // sample, individual name (overall, last), arraylist of files

        // 1. a HashMap where the key is the sample number and the value another HashMap;
        // 2. A HashMap where the key is the classifier name (e.g. J48, RandomForest, etc) and the value an ArrayList
        // 3. An ArrayList with the file names
        HashMap<Integer, HashMap<String, ArrayList<String>>> filePreds = CompilePredictions.collectFiles(
                cmd.getOptionValue("path_predictions")
        );

        HashMap<Integer, Instances> dummyDatasets = new HashMap<>();
        HashMap<Integer, HashMap<String, AbstractClassifier>> classifiers_per_sample = new HashMap<>();

        for(Integer n_sample : filePreds.keySet()) {
            HashMap<String, AbstractClassifier> all_classifiers = new HashMap<>();

            for(String indName : filePreds.get(n_sample).keySet()) {
                CompilePredictions fj = new CompilePredictions(filePreds.get(n_sample).get(indName), cmd.getOptionValue("path_predictions"));
                dummyDatasets.put(n_sample, fj.getDummyDataset());  // operation gets repeated but has no effect on subsequent calls
                HashMap<String, DummyClassifier> dummyClassifiers = fj.getDummyClassifiers();

                for(String key : dummyClassifiers.keySet()) {
                    if(!key.contains("ensemble")) {
                        all_classifiers.put(indName + "-" + key + String.format(Locale.US, "-sample-%02d", n_sample), dummyClassifiers.get(key));
                    } else {
                        all_classifiers.put(indName + String.format(Locale.US, "-sample-%02d", n_sample), dummyClassifiers.get(key));
                    }
                }
            }
            classifiers_per_sample.put(n_sample, all_classifiers);
        }
        PBILLogger.newEvaluationsToFile(classifiers_per_sample, dummyDatasets, cmd.getOptionValue("path_predictions"));
    }
}
