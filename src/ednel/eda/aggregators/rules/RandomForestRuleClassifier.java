package ednel.eda.aggregators.rules;

import ednel.Main;
import ednel.TestDatasets;
import ednel.eda.aggregators.RuleExtractorAggregator;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;

import java.util.HashMap;
import java.util.Random;

public class RandomForestRuleClassifier extends RandomForest implements OptionHandler {
    RuleExtractorAggregator aggregator;

    public RandomForestRuleClassifier() {
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
        aggregator = new RuleExtractorAggregator();
        aggregator.setCompetences(new AbstractClassifier[]{this}, data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double index = Double.POSITIVE_INFINITY, max = - Double.POSITIVE_INFINITY;
        double[] dist = distributionForInstance(instance);
        for(int i = 0; i < dist.length; i++) {
            if(dist[i] > max) {
                max = dist[i];
                index = i;
            }
        }
        return index;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.aggregator.aggregateProba(null, instance);
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return this.aggregator.aggregateProba(null, batch);
    }

    public static void main(String[] args) {
        Options options = new Options();

        options.addOption(Option.builder()
                .required(true)
                .longOpt("datasets_path")
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Must lead to a path that contains several subpaths, one for each dataset. " +
                        "Each subpath, in turn, must have the arff files.")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("datasets_names")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Name of the datasets to run experiments. Must be a list separated by a comma\n" +
                        "\tExample: iris,mushroom,adult")
                .build()
        );

        options.addOption(Option.builder()
                .longOpt("output_path")
                .required(true)
                .type(String.class)
                .hasArg()
                .numberOfArgs(1)
                .desc("Path to folder where a markdown file with generated classifier will be written to.")
                .build());

        CommandLineParser parser = new DefaultParser();

        try {
            CommandLine commandLine = parser.parse(options, args);

            String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");

            for(String dataset_name : dataset_names) {

                System.out.print(String.format("on dataset %s... ", dataset_name));

                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = Main.loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );

                Instances train_data = curDatasetFolds.get(1).get("train");
//                Instances test_data = curDatasetFolds.get(1).get("test");

                RandomForestRuleClassifier rfr = new RandomForestRuleClassifier();
                RandomForest rf = new RandomForest();

                Random rnd = new Random(5);


                Evaluation ev = new Evaluation(train_data);
                ev.crossValidateModel(rfr, train_data, 5, rnd);

                double mean = 0;
                for(int i = 0; i < train_data.numClasses(); i++) {
                    mean += ev.areaUnderROC(i);
                }
                System.out.println("For aggregated rules: " + mean / train_data.numClasses());


                ev = new Evaluation(train_data);
                ev.crossValidateModel(rf, train_data, 5, rnd);
                mean = 0;
                for(int i = 0; i < train_data.numClasses(); i++) {
                    mean += ev.areaUnderROC(i);
                }
                System.out.println("For default random forest: " + mean / train_data.numClasses());

            }
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
