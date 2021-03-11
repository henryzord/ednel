package ednel.utils.analysis;

import ednel.eda.individual.FitnessCalculator;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class FoldJoiner {

    protected Instances dummyDataset;
    protected HashMap<String, DummyClassifier> dummyClassifiers;

    /**
     * Does for only one classifier, instead of multiple.
     *
     * @param probs Probabilities as produced by AbstractClassifier.distributionsForInstances
     * @param y Actual classes of instances in the dataset
     */
    public FoldJoiner(double[][] probs, double[] y) {
        // some instances is the dataset that is built to emulate classification based on predictions of classifiers
        ArrayList<Instance> someInstances = new ArrayList<>();
        // set of un-repeated class values
        HashSet<Double> classValues = new HashSet<>();

        // classifier names as presented in header of lines
        ArrayList<String> clfsNames = new ArrayList<String>(){{
            add("classifier");
        }};

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
        ArrayList<HashMap<Integer, double[]>> probabilities = new ArrayList<HashMap<Integer, double[]>>(){{
            add(probDistForSingleClassifier);
        }};

        this.interpretData(someInstances, classValues, clfsNames, probabilities);
    }

    public FoldJoiner(ArrayList<String> lines) {
        // someInstances is the dataset that is built to emulate classification based on predictions of classifiers
        ArrayList<Instance> someInstances = new ArrayList<>();
        // set of un-repeated class values
        HashSet<Double> classValues = new HashSet<>();

        String[] header = lines.get(0).replace("\n", "").split(";");
        // classifier names as presented in header
        ArrayList<String> clfsNames = new ArrayList<>();

        // list of dictionaries: each list entry is the dictionary for a classifier
        ArrayList<HashMap<Integer, double[]>> probabilities = new ArrayList<>();

        for(int i = 1; i < header.length; i++) {
            clfsNames.add(header[i]);
            probabilities.add(new HashMap<>());
        }

        int counter = 0;
        for(int i = 1; i < lines.size(); i++) {
            // each .preds file has a header. So header lines might get repeated
            if(lines.get(i).contains("classValue")) {
                continue;
            }

            String[] splitted = lines.get(i).replace("\n", "").split(";");

            double actualClass = Double.parseDouble(splitted[0]);

            classValues.add(actualClass);
            // counter is the predictive value (dummy), actualClass the class value
            someInstances.add(new DenseInstance(1, new double[]{(double) counter, actualClass}));

            for(int j = 1; j < splitted.length; j++) {
                double[] probs;
                if(splitted[j].length() > 0) {
                    String[] strProbs = splitted[j].split(",");
                    probs = new double[strProbs.length];
                    for (int k = 0; k < strProbs.length; k++) {
                        probs[k] = Double.parseDouble(strProbs[k]);
                    }
                } else {
                    // adds null because this classifier is not present for this fold.
                    // statistics for this classifier will not be present
                    probs = null;
                }
                HashMap<Integer, double[]> thisClfProbabilities = probabilities.remove(j - 1);
                thisClfProbabilities.put(counter, probs);
                probabilities.add(j - 1, thisClfProbabilities);
            }
            counter += 1;
        }
        this.interpretData(someInstances, classValues, clfsNames, probabilities);
    }

    public FoldJoiner(ArrayList<String> files, String path_predictions) throws Exception {
        this(FoldJoiner.fromFilesToTable(files, path_predictions));
    }

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
     * Builds DummyDataset and DummyClassifiers
     *
     * @param someInstances An ArrayList of instances, as read from files
     * @param classValues HashSet of class values (that is, unrepeated values)
     * @param clfNames Name of classifiers in files
     * @param probabilities An ArrayList where in each position there is a HashMap for each classifier; and each HashMap
     *                      is the index of the instance, and the probability distribution for that classifier for that
     *                      instance.
     */
    protected void interpretData(
            ArrayList<Instance> someInstances, HashSet<Double> classValues,
            ArrayList<String> clfNames, ArrayList<HashMap<Integer, double[]>> probabilities
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
        for(int i = 0; i < clfNames.size(); i++) {
            this.dummyClassifiers.put(clfNames.get(i), new DummyClassifier(probabilities.get(i)));
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
}
