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
import java.lang.reflect.Array;
import java.util.*;

public class RuleExtractorAggregator extends Aggregator implements Serializable {

    ArrayList<ExtractedRule> rules;
    int n_classes;

    public RuleExtractorAggregator() {
        rules = new ArrayList<>();
        this.competences = new double[0];
        n_classes = 0;
    }

    /**
     * Set competences for rules (i.e. not classifiers).
     * @param clfs List of classifiers
     * @param train_data Training data
     * @throws Exception
     */
    @Override
    public void setCompetences(AbstractClassifier[] clfs, Instances train_data) throws Exception {
        HashSet<ExtractedRule> candidate_rules = new HashSet<>();

        for(AbstractClassifier clf : clfs) {
            // algorithm generates unordered rules; proceed
            if(!clf.getClass().equals(JRip.class) && !clf.getClass().equals(PART.class)) {
                candidate_rules.addAll(Arrays.asList(RuleExtractorAggregator.fromClassifierToRules(clf, train_data)));
            }
        }

        this.rules = new ArrayList<>();
        this.n_classes = train_data.numClasses();
        ArrayList<Double> qualities = new ArrayList<>();

        boolean activated[] = new boolean[train_data.size()];
        for(int i = 0; i < train_data.size(); i++) {
            activated[i] = true;
        }

        double bestQuality;
        ExtractedRule bestRule;

        int remaining_instances = train_data.size();
        while(remaining_instances > 0) {
            bestRule = null;
            bestQuality = 0.0;
            for(ExtractedRule rule : candidate_rules) {
                double quality = rule.quality(train_data, activated);
                if(quality > bestQuality) {
                    bestQuality = quality;
                    bestRule = rule;
                }
            }
            if(bestRule == null) {
                break;  // nothing else to do!
            }

            // TODO using quality on instances that this rule could see during training!
            rules.add(bestRule);
            qualities.add(bestQuality);

            candidate_rules.remove(bestRule);
            remaining_instances = 0;

            boolean[] covered = bestRule.covers(train_data, activated);
            for(int i = 0; i < train_data.size(); i++) {
                activated[i] = activated[i] && !(covered[i] && (bestRule.getConsequent() == train_data.get(i).classValue()));
                remaining_instances += activated[i]? 1 : 0;
            }
        }

        this.competences = new double[qualities.size()];
        for(int i = 0; i < qualities.size(); i++) {
            this.competences[i] = qualities.get(i);
        }
        qualities = null;
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    @Override
    public double[][] aggregateProba(AbstractClassifier[] clfs, Instances batch) throws Exception {
        double[][] classProbs = new double[batch.size()][this.n_classes];
        
        for(int i = 0; i < batch.size(); i++) {
            int votesSum = 0;
            for(int c = 0; c < this.n_classes; c++) {
                classProbs[i][c] = 0.0;
            }

            for(int j = 0; j < this.rules.size(); j++) {
                if(this.rules.get(j).covers(batch.get(i))) {
                    classProbs[i][(int)this.rules.get(j).getConsequent()] += 1;
                    votesSum += 1;
                }
            }
            for(int c = 0; c < this.n_classes; c++) {
                classProbs[i][c] /= (double)votesSum;
            }
        }
        return classProbs;
    }

    @Override
    public double[] aggregateProba(AbstractClassifier[] clfs, Instance instance) throws Exception {
        double[] classProbs = new double[this.n_classes];
        int votesSum = 0;
        for(int c = 0; c < this.n_classes; c++) {
            classProbs[c] = 0.0;
        }

        for(int j = 0; j < this.rules.size(); j++) {
            if(this.rules.get(j).covers(instance)) {
                classProbs[(int)this.rules.get(j).getConsequent()] += 1;
                votesSum += 1;
            }
        }
        for(int c = 0; c < this.n_classes; c++) {
            classProbs[c] /= (double)votesSum;
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
        ArrayList<ExtractedRule> cumulativeRules = new ArrayList<>();
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, i == 0? null : cumulativeRules);
            cumulativeRules.add(rules[i]);
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
        ArrayList<ExtractedRule> cumulativeRules = new ArrayList<>();
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, i == 0? null : cumulativeRules);
            cumulativeRules.add(rules[i]);
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
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, null);
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
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, null);
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

                extractedRules[counter] = new ExtractedRule(newline.toString(), train_data, null);
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
