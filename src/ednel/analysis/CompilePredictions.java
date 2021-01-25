package ednel.analysis;

import ednel.Main;
import org.apache.commons.cli.*;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;

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

    protected static HashMap<String, ArrayList<String>> collectFiles(String path) {
        File folder = new File(path);
        String[] files = folder.list();

        HashMap<String, ArrayList<String>> filePreds = new HashMap<>();
        for(String file : files) {
            if(file.contains(".json")) {
                String[] split1 = file.split("_");
                String indName = (split1[split1.length - 1]).split(".json")[0];

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

        Main.loadDataset(
                cmd.getOptionValue("datasets_path"),
                cmd.getOptionValue("dataset_name"),
                1  // TODO do until 10
        );


        int z = 0;

    }
}
