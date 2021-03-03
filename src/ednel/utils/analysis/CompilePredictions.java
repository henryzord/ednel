package ednel.utils.analysis;

import ednel.utils.PBILLogger;
import org.apache.commons.cli.*;
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
                .desc("Path to folder where .json files (one for each fold) are stored. In this same folder," +
                        "a \"summary.csv\" file with all results will be written.")
                .build()
        );

//        options.addOption(Option.builder()
//                .required(true)
//                .longOpt("datasets_path")
//                .type(String.class)
//                .hasArg()
//                .numberOfArgs(1)
//                .desc("Path to folder where all datasets are stored")
//                .build()
//        );
//
//        options.addOption(Option.builder()
//                .required(true)
//                .longOpt("dataset_name")
//                .type(String.class)
//                .hasArg()
//                .numberOfArgs(1)
//                .desc("Path to folder where all datasets are stored")
//                .build()
//        );

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);
        return cmd;
    }

    /**
     * Collects files from a folder.
     *
     * Generates a HashMap with the following hierarchy: n_sample, indName, list of predictions per folds
     * @param path
     * @return
     * @throws NullPointerException
     */
    protected static HashMap<Integer, HashMap<String, ArrayList<String>>> collectFiles(String path) throws NullPointerException {
        File folder = new File(path);
        String[] files = folder.list();

        HashMap<Integer, HashMap<String, ArrayList<String>>> filePreds = new HashMap<>();

        for(String file : files) {
            if(file.contains(".preds")) {
                // String[] split1 = file.split("_");
                //String indName = (split1[split1.length - 1]).split(".preds")[0];
                String indName = "someClassifier";
                int n_sample = Integer.parseInt(file.substring(file.indexOf("sample-") + "sample-".length(), file.indexOf("sample-") + "sample-".length() + 2));

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
        HashMap<Integer, HashMap<String, ArrayList<String>>> filePreds = collectFiles(cmd.getOptionValue("path_predictions"));

        Instances dummyDataset = null;
        HashMap<String, AbstractClassifier> all_classifiers = new HashMap<>();

        for(Integer n_sample : filePreds.keySet()) {
            for(String indName : filePreds.get(n_sample).keySet()) {
                FoldJoiner fj = new FoldJoiner(filePreds.get(n_sample).get(indName), cmd.getOptionValue("path_predictions"));
                dummyDataset = fj.getDummyDataset();
                HashMap<String, DummyClassifier> dummyClassifiers = fj.getDummyClassifiers();

                for(String key : dummyClassifiers.keySet()) {
                    if(!key.contains("ensemble")) {
                        all_classifiers.put(indName + "-" + key + String.format(Locale.US, "-sample-%02d", n_sample), dummyClassifiers.get(key));
                    } else {
                        all_classifiers.put(indName + String.format(Locale.US, "-sample-%02d", n_sample), dummyClassifiers.get(key));
                    }
                }
            }
        }
        PBILLogger.newEvaluationsToFile(filePreds.size(), all_classifiers, dummyDataset, cmd.getOptionValue("path_predictions"));
    }
}
