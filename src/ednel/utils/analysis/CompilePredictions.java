package ednel.utils.analysis;

import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
import smile.neighbor.lsh.Hash;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.io.File;
import java.util.*;

public class CompilePredictions {

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
                FoldJoiner fj = new FoldJoiner(filePreds.get(n_sample).get(indName), cmd.getOptionValue("path_predictions"));
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
