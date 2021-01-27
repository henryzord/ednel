package ednel.analysis;

import ednel.Main;
import ednel.eda.individual.FitnessCalculator;
import javassist.bytecode.AttributeInfo;
import org.apache.commons.cli.*;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class CompilePredictions {

    protected static CommandLine parseOptions(String[] args) throws ParseException {
        Options options = new Options();

        options.addOption(Option.builder()
                .required(true)
                .longOpt("path_predictions")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where .json files (one for each fold) are stored. In this same folder," +
                        "a \"summary.csv\" file with all results will be written.")
                .build()
        );

        options.addOption(Option.builder()
                .required(true)
                .longOpt("datasets_path")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where all datasets are stored")
                .build()
        );

        options.addOption(Option.builder()
                .required(true)
                .longOpt("dataset_name")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where all datasets are stored")
                .build()
        );

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);
        return cmd;
    }

    protected static HashMap<String, ArrayList<String>> collectFiles(String path) throws NullPointerException {
        File folder = new File(path);
        String[] files = folder.list();

        HashMap<String, ArrayList<String>> filePreds = new HashMap<>();
        for(String file : files) {
            if(file.contains(".preds")) {
                String[] split1 = file.split("_");
                String indName = (split1[split1.length - 1]).split(".preds")[0];

                ArrayList<String> thisList;
                if(!filePreds.containsKey(indName)) {
                    thisList = new ArrayList<>();
                } else {
                    thisList = filePreds.get(indName);
                }
                thisList.add(file);
                filePreds.put(indName, thisList);
            }
        }
        return filePreds;
    }

    public static void main(String[] args) throws Exception {
        CommandLine cmd = CompilePredictions.parseOptions(args);

        HashMap<String, ArrayList<String>> filePreds = collectFiles(cmd.getOptionValue("path_predictions"));

        HashMap<String, DummyClassifier> dummies = new HashMap<>();
        HashMap<String, Instances> dummyDatasets = new HashMap<>();

        ArrayList<Instance> someInstances = new ArrayList<>();
        HashSet<Double> classValues = new HashSet<>();

        for(String indName : filePreds.keySet()) {
            HashMap<Integer, double[]> probabilities = new HashMap<>();
            HashMap<Integer, Double> actualClasses = new HashMap<>();

            int counter = 0;
            for(String some_file : filePreds.get(indName)) {
                BufferedReader br = new BufferedReader(new FileReader(String.format("%s%s%s", cmd.getOptionValue("path_predictions"), File.separator, some_file)));
                String line;
                while ((line = br.readLine()) != null) {
                    String[] splitted = line.split(";");
                    double actualClass = Double.parseDouble(splitted[0]);
                    String[] strProbs = splitted[1].split(",");
                    double[] probs = new double[strProbs.length];
                    for(int j = 0; j < strProbs.length; j++) {
                        probs[j] = Double.parseDouble(strProbs[j]);
                    }
                    probabilities.put(counter, probs);
                    actualClasses.put(counter, actualClass);
                    DenseInstance inst = new DenseInstance(1, new double[]{(double)counter, actualClass});
                    classValues.add(actualClass);
                    someInstances.add(inst);

                    counter += 1;
                }
            }
            DummyClassifier dummy = new DummyClassifier(probabilities);
            dummies.put(indName, dummy);

            ArrayList<String> classValuesAL = new ArrayList<>();
            for(Double ob : classValues) {
                classValuesAL.add(ob.toString());
            }
            Collections.sort(classValuesAL);

            ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(){{
                add(new Attribute("predictive"));
                add(new Attribute("class", classValuesAL));
            }};
            Instances dummyDataset = new Instances("dummy dataset", attrInfo, someInstances.size());
            dummyDataset.addAll(someInstances);
            dummyDataset.setClassIndex(dummyDataset.numAttributes() - 1);
            dummyDatasets.put(indName, dummyDataset);

            // HashMap<String, Instances> loaded = Main.loadDataset(
            //         cmd.getOptionValue("datasets_path"),
            //         cmd.getOptionValue("dataset_name"),
            //         1
            // );

            // Instances all_dataset = new Instances(loaded.get("train_data"));
            // boolean res = all_dataset.addAll(new Instances(loaded.get("test_data")));
            // if(!res) {
            //     throw new Exception("could not append datasets.");
            // }
            // all_dataset.setClassIndex(all_dataset.numAttributes() - 1);

            Evaluation eval = new Evaluation(dummyDataset);
            eval.evaluateModel(dummy, dummyDataset);

            for(int i = 0; i < dummyDataset.numClasses(); i++) {
                System.out.printf("auc for class %d: %f\n", i, eval.areaUnderROC(i));
            }
            double general_auc = FitnessCalculator.getUnweightedAreaUnderROC(eval);
            System.out.printf("general auc: %f\n", general_auc);

            int z = 0;

            // TODO test
        }


        int z = 0;

    }
}
