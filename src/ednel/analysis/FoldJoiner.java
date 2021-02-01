package ednel.analysis;

import ednel.eda.individual.FitnessCalculator;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class FoldJoiner {

    protected Instances dummyDataset;
    protected DummyClassifier dummyClassifier;

    public FoldJoiner(ArrayList<String> lines) {
        HashMap<Integer, double[]> probabilities = new HashMap<>();
        HashMap<Integer, Double> actualClasses = new HashMap<>();

        ArrayList<Instance> someInstances = new ArrayList<>();
        HashSet<Double> classValues = new HashSet<>();

        int counter = 0;
        for(String line : lines) {
            String[] splitted = line.split(";");
            double actualClass = Double.parseDouble(splitted[0]);
            String[] strProbs = splitted[1].split(",");
            double[] probs = new double[strProbs.length];
            for (int j = 0; j < strProbs.length; j++) {
                probs[j] = Double.parseDouble(strProbs[j]);
            }
            probabilities.put(counter, probs);
            actualClasses.put(counter, actualClass);
            DenseInstance inst = new DenseInstance(1, new double[]{(double) counter, actualClass});
            classValues.add(actualClass);
            someInstances.add(inst);

            counter += 1;
        }
        this.interpretData(someInstances, classValues, probabilities);
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

    protected void interpretData(
            ArrayList<Instance> someInstances, HashSet<Double> classValues,
            HashMap<Integer, double[]> probabilities
    ) {
        this.dummyClassifier = new DummyClassifier(probabilities);

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
    }

    public double getAUC() throws Exception {
        Evaluation eval = new Evaluation(this.dummyDataset);
        eval.evaluateModel(this.dummyClassifier, this.dummyDataset);

        return FitnessCalculator.getUnweightedAreaUnderROC(eval);
    }
}
