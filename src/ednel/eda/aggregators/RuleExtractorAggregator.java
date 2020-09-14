package ednel.eda.aggregators;

import ednel.Main;
import ednel.TestDatasets;
import ednel.classifiers.trees.SimpleCart;
import ednel.eda.aggregators.rules.RuleExtractor;
import ednel.eda.aggregators.rules.SimpleRuleClassifier;
import ednel.eda.rules.ExtractedRule;
import org.apache.commons.cli.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;

public class RuleExtractorAggregator extends Aggregator implements Serializable {

    ArrayList<ExtractedRule> unorderedRules;
    ArrayList<Double> unorderedRulesQualities;
    int n_classes;
    HashMap<String, ExtractedRule[]> orderedRules;
    HashMap<String, Double[]> orderedRuleQualities;

    public RuleExtractorAggregator() {
        this.unorderedRules = new ArrayList<>();
        this.unorderedRulesQualities = new ArrayList<>();
        this.competences = new double[0];
        this.n_classes = 0;
        this.orderedRules = new HashMap<>();
        this.orderedRuleQualities = new HashMap<>();
    }

    /**
     * Set competences for rules (i.e. not classifiers).
     *
     * @param clfs List of classifiers
     * @param train_data Training data
     * @throws Exception
     */
    @Override
    public void setCompetences(AbstractClassifier[] clfs, Instances train_data) throws Exception {
        this.n_classes = train_data.numClasses();

        HashSet<ExtractedRule> unordered_cand_rules = new HashSet<>();

        this.orderedRules = new HashMap<>();
        this.orderedRuleQualities = new HashMap<>();

        final boolean[] all_activated = new boolean[train_data.size()];
        for(int i = 0; i < train_data.size(); i++) {
            all_activated[i] = true;
        }

        for(int i = 0; i < clfs.length; i++) {
            if(clfs[i] == null) {
                continue;
            }
            // algorithm generates unordered rules; proceed
            if(!clfs[i].getClass().equals(JRip.class) && !clfs[i].getClass().equals(PART.class)) {

                unordered_cand_rules.addAll(Arrays.asList(
                        RuleExtractor.fromClassifierToRules(clfs[i], train_data)
                ));
            } else if(clfs[i].getClass().equals(PART.class) || clfs[i].getClass().equals(JRip.class)) {
                String clf_name = clfs[i].getClass().getSimpleName();
                ExtractedRule[] ordered_rules = RuleExtractor.fromClassifierToRules(clfs[i], train_data);
                orderedRules.put(clf_name, ordered_rules);

                boolean[] activated = all_activated.clone();
                Double[] rule_qualities = new Double[ordered_rules.length];

                for(int j = 0; j < ordered_rules.length; j++) {
                    rule_qualities[j] = ordered_rules[j].quality(train_data, activated);

                    boolean[] covered = ordered_rules[j].covers(train_data, activated);
                    for(int n = 0; n < train_data.size(); n++) {
                        activated[n] = activated[n] && !(covered[n] && (ordered_rules[j].getConsequent() == train_data.get(n).classValue()));
                    }
                }
                orderedRuleQualities.put(clf_name, rule_qualities);
            }
        }

        this.selectUnorderedRules(train_data, unordered_cand_rules, all_activated);
    }

    private void selectUnorderedRules(Instances train_data, HashSet<ExtractedRule> candidateRules, boolean[] all_activated) {
        boolean[] activated = all_activated.clone();

        double bestQuality;
        ExtractedRule bestRule;

        int remaining_instances = train_data.size();
        while(remaining_instances > 0) {
            bestRule = null;
            bestQuality = 0.0;
            for(ExtractedRule rule : candidateRules) {
                double quality = rule.quality(train_data, activated);
                if(quality > bestQuality) {
                    bestQuality = quality;
                    bestRule = rule;
                }
            }
            if(bestRule == null) {
                break;  // nothing else to do!
            }

            unorderedRules.add(bestRule);
            unorderedRulesQualities.add(bestRule.quality(train_data, all_activated));

            candidateRules.remove(bestRule);
            remaining_instances = 0;

            boolean[] covered = bestRule.covers(train_data, activated);
            for(int i = 0; i < train_data.size(); i++) {
                activated[i] = activated[i] && !(covered[i] && (bestRule.getConsequent() == train_data.get(i).classValue()));
                remaining_instances += activated[i]? 1 : 0;
            }
        }
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    @Override
    public double[][] aggregateProba(AbstractClassifier[] clfs, Instances batch) throws Exception {
        double[][] classProbs = new double[batch.size()][this.n_classes];

        for(int i = 0; i < batch.size(); i++) {
            double votesSum = 0.0;
            for(int c = 0; c < this.n_classes; c++) {
                classProbs[i][c] = 0.0;
            }
            // ordered classifiers vote first
            for(String classifier : this.orderedRules.keySet()) {
                ExtractedRule[] rules = this.orderedRules.get(classifier);
                for(int j = 0; j < rules.length; j++) {
                    if(rules[j].covers(batch.get(i))) {
                        double voteWeight = this.orderedRuleQualities.get(classifier)[j];
                        classProbs[i][(int)rules[j].getConsequent()] += voteWeight;
                        votesSum += voteWeight;
                    }
                }
            }

            // unordered/grouped classifiers vote next
            for(int j = 0; j < this.unorderedRules.size(); j++) {
                if(this.unorderedRules.get(j).covers(batch.get(i))) {
                    double voteWeight = this.unorderedRulesQualities.get(j);
                    classProbs[i][(int)this.unorderedRules.get(j).getConsequent()] += voteWeight;
                    votesSum += voteWeight;
                }
            }
            for(int c = 0; c < this.n_classes; c++) {
                classProbs[i][c] /= votesSum;
            }
        }
        return classProbs;
    }

    @Override
    public double[] aggregateProba(AbstractClassifier[] clfs, Instance instance) throws Exception {
        double[] classProbs = new double[this.n_classes];
        double votesSum = 0.0;
        for(int c = 0; c < this.n_classes; c++) {
            classProbs[c] = 0.0;
        }
        // ordered classifiers vote first
        for(String classifier : this.orderedRules.keySet()) {
            ExtractedRule[] rules = this.orderedRules.get(classifier);
            for(int j = 0; j < rules.length; j++) {
                if(rules[j].covers(instance)) {
                    double voteWeight = this.orderedRuleQualities.get(classifier)[j];
                    classProbs[(int)rules[j].getConsequent()] += voteWeight;
                    votesSum += voteWeight;
                }
            }
        }

        // unordered/grouped classifiers vote next
        for(int j = 0; j < this.unorderedRules.size(); j++) {
            if(this.unorderedRules.get(j).covers(instance)) {
                double voteWeight = this.unorderedRulesQualities.get(j);
                classProbs[(int)this.unorderedRules.get(j).getConsequent()] += voteWeight;
                votesSum += voteWeight;
            }
        }
        for(int c = 0; c < this.n_classes; c++) {
            classProbs[c] /= votesSum;
        }

        return classProbs;
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

                System.out.print(String.format("on dataset %s...\n", dataset_name));

                HashMap<Integer, HashMap<String, Instances>> curDatasetFolds = Main.loadFoldsOfDatasets(
                        commandLine.getOptionValue("datasets_path"),
                        dataset_name
                );

                Instances train_data = curDatasetFolds.get(1).get("train");
                Instances test_data = curDatasetFolds.get(1).get("test");

                Random rnd = new Random(5);
                AbstractClassifier[] clfs = new AbstractClassifier[]{ new DecisionTable(), new JRip(), new SimpleCart(), new J48(), new PART()};

                for(AbstractClassifier clf : clfs) {
                    System.out.println("at classifier " + clf.getClass());

                    Evaluation ev = new Evaluation(train_data);
                    clf.buildClassifier(train_data);
                    ev.evaluateModel(clf, test_data);

                    System.out.println("Normal behavior: \t\t" + ev.errorRate());

                    SimpleRuleClassifier src = new SimpleRuleClassifier(clf);
                    src.buildClassifier(train_data);
//                    ev = new Evaluation(train_data);
                    ev.evaluateModel(src, test_data);
                    System.out.println("Extracted behavior: \t" + ev.errorRate());
                }
            }
        } catch (Exception pe) {
            pe.printStackTrace();
        }
    }
}
