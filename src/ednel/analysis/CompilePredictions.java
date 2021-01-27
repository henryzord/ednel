package ednel.analysis;

import ednel.eda.individual.FitnessCalculator;
import org.apache.commons.cli.*;
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

        for(String indName : filePreds.keySet()) {
            FoldJoiner fj = new FoldJoiner(filePreds.get(indName), cmd.getOptionValue("path_predictions"));
            System.out.println("AUC for individual " + indName + " " + fj.getAUC());
        }
    }
}
