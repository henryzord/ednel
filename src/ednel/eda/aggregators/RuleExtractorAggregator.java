package ednel.eda.aggregators;

import ednel.eda.rules.ExtractedRule;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

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
        try {
            System.out.println(RuleExtractor.formatNumericDecisionTableCell("(87.5-98.5]", "any_column"));
        } catch (Exception e) {
            e.printStackTrace();
        }
//            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource("C:\\Users\\henry\\Desktop\\play_tennis.arff");
//
//            AbstractClassifier[] clfs = new AbstractClassifier[]{new JRip(), new PART(), new J48(), new DecisionTable(), new SimpleCart()};
//
//            Instances train_data = train_set.getDataSet();
//            train_data.setClassIndex(train_data.numAttributes() - 1);
//
//            RealRule[][] all_rules = new RealRule[clfs.length][];
//
//            for(int i = 0; i < clfs.length; i++) {
//                clfs[i].buildClassifier(train_data);
//                all_rules[i] = RuleExtractorAggregator.fromClassifierToRules(clfs[i], train_data);
//            }
//            for(int c = 0; c < clfs.length; c++) {
//                for(int r = 0; r < all_rules[c].length; r++) {
//                    if(all_rules[c][r].covers(train_data.get(0))) {
//                        System.out.println(String.format("rule %d from classifier %d: %s", r, c, all_rules[c][r]));
//                    }
//                }
//            }
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    }
}
