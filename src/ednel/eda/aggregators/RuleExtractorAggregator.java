package ednel.eda.aggregators;

import ednel.classifiers.trees.SimpleCart;
import ednel.eda.rules.ExtractedRule;
import weka.classifiers.AbstractClassifier;
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
//    double aggregatedRuleCompetence;

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
            // algorithm generates unordered rules; proceed
            if(!clfs[i].getClass().equals(JRip.class) && !clfs[i].getClass().equals(PART.class)) {

                unordered_cand_rules.addAll(Arrays.asList(
                        RuleExtractorAggregator.fromClassifierToRules(clfs[i], train_data)
                ));
            } else if(clfs[i].getClass().equals(PART.class) || clfs[i].getClass().equals(JRip.class)) {
                String clf_name = clfs[i].getClass().getSimpleName();
                ExtractedRule[] ordered_rules = RuleExtractorAggregator.fromClassifierToRules(clfs[i], train_data);
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


    public static ExtractedRule[] fromClassifierToRules(AbstractClassifier clf, Instances train_data) throws Exception {
        if(clf instanceof J48) {
            return RuleExtractorAggregator.fromJ48ToRules((J48)clf, train_data);
        } else if(clf instanceof DecisionTable) {
            return RuleExtractorAggregator.fromDecisionTableToRules((DecisionTable)clf, train_data);
        } else if(clf instanceof SimpleCart) {
            return RuleExtractorAggregator.fromSimpleCartToRules((SimpleCart)clf, train_data);
        } else if(clf instanceof JRip) {
            return RuleExtractorAggregator.fromJRipToRules((JRip)clf, train_data);
        } else if(clf instanceof PART) {
            return RuleExtractorAggregator.fromPARTToRules((PART)clf, train_data);
        }

        throw new ClassNotFoundException(
                "clf must be one of the following classifiers: J48, SimpleCart, JRip, PART, DecisionTable"
        );
    }

    private static ExtractedRule[] fromPARTToRules(PART clf, Instances train_data) throws Exception {
        String str = clf.toString();

        String[] lines = str.substring(
                str.indexOf("------------------") + "------------------".length(),
                str.indexOf("Number of Rules")
        ).trim().split("\n\n");

        ArrayList<String> rule_lines = new ArrayList<>(lines.length);

        for(int i = 0; i < lines.length; i++) {
            rule_lines.add(lines[i].replaceAll("AND\n", "and "));
        }

        ExtractedRule[] rules = new ExtractedRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data);
        }
        return rules;
    }

    private static ExtractedRule[] fromJRipToRules(JRip clf, Instances train_data) throws Exception {
        String str = clf.toString();

        String[] lines = str.substring(str.indexOf("===========") + "===========".length(), str.indexOf("Number of Rules")).trim().split("\n");

        ArrayList<String> rule_lines = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            String priors = lines[i].substring(0, lines[i].lastIndexOf("=>")).trim();
//            if (priors.length() > 0) {
            String posteriori = lines[i].substring(lines[i].lastIndexOf("=>") + "=>".length()).trim();
            priors = priors.replaceAll("\\(", "").replaceAll("\\)", "");
            rule_lines.add(String.format("%s: %s", priors, posteriori.split("=")[1]));
//            }
        }
        ExtractedRule[] rules = new ExtractedRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data);
        }
        return rules;
    }

    private static ExtractedRule[] fromSimpleCartToRules(SimpleCart clf, Instances train_data) throws Exception {
        String str = clf.toString();

        String[] lines = (
                str.substring(
                        str.indexOf("CART Decision Tree") + "CART Decision Tree".length(),
                        str.indexOf("Number of Leaf Nodes")
                )
        ).trim().split("\n");

        // levels has the level that each rule is within
        ArrayList<Integer> levels = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            int count = 0, fromIndex = 0;
            // while there are still sub-levels within this rule
            while ((fromIndex = lines[i].indexOf("| ", fromIndex)) != -1) {
                count++;
                fromIndex += "| ".length();
            }
            levels.add(count);
        }
        ArrayList<String> new_lines = new ArrayList<>(lines.length);
        ArrayList<Integer> new_levels = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replaceAll("\\| ", "").trim();
            if(lines[i].contains("|")) {
                String[] separated0 = lines[i].split(":");  // separated0[0] = priors, separated0[1] = posteriori
                boolean hasPosteriori = separated0.length > 1;
                String symbol = separated0[0].contains("!=")? "!=" : "=";
                String[] separated1 = separated0[0].split("!=|="); // separated1[0] = attribute name,
                // separated1[1] = values (if more than one)
                String[] separated2 = separated1[1].split("\\|");  // separated2 = attribute values
                if(separated2.length > 1) {

                    for(String attrValue : separated2) {
                        String new_line = String.format(
                                "%s %s %s%s",
                                separated1[0],
                                symbol,
                                attrValue.substring(1, attrValue.length() - 1),
                                hasPosteriori? ":" + separated0[1] : ""
                        );
                        new_lines.add(new_line);
                        new_levels.add(levels.get(i));
                    }
                }
            } else {
//                String new_line = String.format("%s=%s%s", separated1[0], separated2[0].substring(1, separated2[0].length() - 1), hasPosteriori? ":" + separated0[1] : "");
                new_lines.add(lines[i]);
                new_levels.add(levels.get(i));
            }
        }

        int deepest_level = Collections.max(new_levels);
        for(int lvl = deepest_level; lvl > 0; lvl--) {
            int index_last_minus = -1;
            for(int i = 0; i < new_levels.size(); i++) {
                if(new_levels.get(i) == (lvl - 1)) {
                    index_last_minus = i;
                } else if(new_levels.get(i) == lvl) {
                    new_lines.set(i, new_lines.get(index_last_minus) + " and " + new_lines.get(i));
                    new_levels.set(i, new_levels.get(i) - 1);
                }
            }
        }
        ArrayList<String> rule_lines = new ArrayList<>(new_lines.size());
        for(int i = 0; i < new_lines.size(); i++) {
            if(new_lines.get(i).contains(":")) {
                // checks if there are pre-conditions
                if(new_lines.get(i).split(":")[0].trim().length() > 0) {
                    rule_lines.add(new_lines.get(i));
                }
            }
        }

        ExtractedRule[] rules = new ExtractedRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data);
        }
        return rules;
    }

    private static ExtractedRule[] fromJ48ToRules(J48 clf, Instances train_data) throws Exception {
        String str = clf.toString();
        String[] lines = (
                str.substring(
                        str.indexOf("------------------") + "------------------".length(),
                        str.indexOf("Number of Leaves")
                )
        ).trim().split("\n");

        // levels has the level that each rule is within
        ArrayList<Integer> levels = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            int count = 0, fromIndex = 0;
            // while there are still sub-levels within this rule
            while ((fromIndex = lines[i].indexOf("|", fromIndex)) != -1) {
                count++;
                fromIndex++;
            }
            levels.add(count);

            lines[i] = lines[i].replaceAll("\\|", "").trim();
        }

        int deepest_level = Collections.max(levels);
        for(int j = deepest_level; j > 0; j--) {
            int index_last_minus = -1;
            for(int i = 0; i < levels.size(); i++) {
                if(levels.get(i) == (j - 1)) {
                    index_last_minus = i;
                } else if(levels.get(i) == j) {
                    lines[i] = lines[index_last_minus] + " and " + lines[i];
                    levels.set(i, levels.get(i) - 1);
                }
            }
        }
        ArrayList<String> rule_lines = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            if(lines[i].contains(":")) {
                // checks if there are pre-conditions
                if(lines[i].split(":")[0].trim().length() > 0) {
                    rule_lines.add(lines[i]);
                }
            }
        }

        ExtractedRule[] rules = new ExtractedRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data);
        }
        return rules;
    }

    private static String formatNumericDecisionTableCell(String pre, String column_name) throws Exception {
        // TODO method breaks when interval does not involve inf values! (e.g. '(87.5-98.5]')

        pre = pre.replaceAll("\'", "");

        String[] parts = pre.substring(1, pre.length() - 1).split("-");
        if(parts.length > 2) {
            String[] new_parts = new String[2];
            new_parts[0] = "-" + parts[1];
            new_parts[1] = parts[2];
            parts = new_parts;
        }

        char opening_char = pre.charAt(0);
        char closing_char = pre.charAt(pre.length() - 1);

        String post_process = "";

        if(!parts[0].equals("-inf")) {
            if(opening_char == '(') {
                post_process += String.format("%s > " + parts[0], column_name);
            } else if(opening_char == '[') {
                post_process += String.format("%s >= " + parts[0], column_name);
            } else {
                throw new Exception("pre must be opened either by a ( or a [ character!");
            }
            if(!parts[1].equals("inf")) {
                post_process += " and ";
            }
        }
        if(!parts[1].equals("inf")) {
            if(closing_char == ')') {
                post_process += String.format("%s < " + parts[1], column_name);
            } else if (closing_char == ']') {
                post_process += String.format("%s <= " + parts[1], column_name);
            } else {
                throw new Exception("pre must be closed either by a ) or a ] character!");
            }
        }
        return post_process;
    }

    public static ExtractedRule[] fromDecisionTableToRules(DecisionTable decisionTable, Instances train_data) throws Exception {
        decisionTable.setDisplayRules(true);
        String str = decisionTable.toString();

        String[] rules = str.split("Rules:")[1].trim().split("\n");

        ArrayList<String> headerColumns = new ArrayList<>();
        for (String header : rules[1].split(" ")) {
            if (header.length() > 0) {
                headerColumns.add(header);
            }
        }

        ExtractedRule[] extractedRules = new ExtractedRule[rules.length - 4];

        StringBuffer prior = new StringBuffer("");
        if (headerColumns.size() > 1) {
            // why start at 3 and finish at -1? because rules[0] and rules[2] are only table delimiters
            // (i.e. ========), as well as rules[-1]. rules[1] contains the column headers of the table
            int counter = 0;
            for (int i = 3; i < rules.length - 1; i++) {
                String[] priors = rules[i].split(" +");
                String posterior = priors[priors.length - 1];
                StringBuffer newline = new StringBuffer("");
                for(int j = 0; j < priors.length - 1; j++) {
                    String post = priors[j];
                    if(priors[j].contains("\'")) {
                        post = RuleExtractorAggregator.formatNumericDecisionTableCell(priors[j], headerColumns.get(j));
                    } else {
                        post = String.format("%s = %s", headerColumns.get(j), post);
                    }
                    newline.append(post);
                    if(j != priors.length - 2) {
                        newline.append(" and ");
                    }
                }
                newline.append(": ").append(posterior);

                extractedRules[counter] = new ExtractedRule(newline.toString(), train_data);
                counter += 1;
            }
        } else {  // has only default rule!
            return new ExtractedRule[0];
        }
        return extractedRules;
    }

    public static void main(String[] args) {
        try {
            System.out.println(RuleExtractorAggregator.formatNumericDecisionTableCell("(87.5-98.5]", "any_column"));
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
