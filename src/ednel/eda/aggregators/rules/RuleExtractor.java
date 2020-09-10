package ednel.eda.aggregators.rules;

import ednel.classifiers.trees.SimpleCart;
import ednel.eda.rules.ExtractedRule;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.experiment.Stats;

import java.util.ArrayList;
import java.util.Collections;

public class RuleExtractor {
    public static ExtractedRule[] fromClassifierToRules(AbstractClassifier clf, Instances train_data) throws Exception {
        double[] mostCommonValueIndices = new double [train_data.numAttributes()];

        for(int i = 0; i < train_data.numAttributes(); i++) {
            if(train_data.attribute(i).isNominal()) {
                int[] nominalCounts = train_data.attributeStats(i).nominalCounts;
                double index = -1, max = Double.NEGATIVE_INFINITY;
                for(int j = 0; j < nominalCounts.length; j++) {
                    if(nominalCounts[j] > max) {
                        index = j;
                        max = nominalCounts[j];
                    }
                }
                mostCommonValueIndices[i] = index;
            } else {
                mostCommonValueIndices[i] = train_data.attributeStats(i).numericStats.mean;
            }
        }
        if(clf instanceof J48) {
            return RuleExtractor.fromJ48ToRules((J48)clf, train_data, mostCommonValueIndices);
        } else if(clf instanceof DecisionTable) {
            return RuleExtractor.fromDecisionTableToRules((DecisionTable)clf, train_data, mostCommonValueIndices);
        } else if(clf instanceof SimpleCart) {
            return RuleExtractor.fromSimpleCartToRules((SimpleCart)clf, train_data, mostCommonValueIndices);
        } else if(clf instanceof JRip) {
            return RuleExtractor.fromJRipToRules((JRip)clf, train_data, mostCommonValueIndices);
        } else if(clf instanceof PART) {
            return RuleExtractor.fromPARTToRules((PART)clf, train_data, mostCommonValueIndices);
        } else if(clf instanceof RandomForest) {
            return RuleExtractor.fromRandomForestToRules((RandomForest)clf, train_data, mostCommonValueIndices);
        }

        throw new ClassNotFoundException(
                "clf must be one of the following classifiers: J48, SimpleCart, JRip, PART, DecisionTable"
        );
    }

    private static ExtractedRule[] fromPARTToRules(PART clf, Instances train_data, double[] mostCommonValueIndices) throws Exception {
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
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, mostCommonValueIndices);
        }
        return rules;
    }

    private static ExtractedRule[] fromJRipToRules(JRip clf, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        String str = clf.toString();

        String[] lines = str.substring(str.indexOf("===========") + "===========".length(), str.indexOf("Number of Rules")).trim().split("\n");

        String splitStr = "=> " + train_data.attribute(train_data.classIndex()).name() + "=";

        ArrayList<String> rule_lines = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            String priors = lines[i].substring(0, lines[i].lastIndexOf(splitStr)).trim();
            String posteriori = lines[i].substring(lines[i].lastIndexOf(splitStr) + splitStr.length()).trim();
            priors = priors.replaceAll("\\(", "").replaceAll("\\)", "");
            rule_lines.add(String.format("%s: %s", priors, posteriori));
        }
        ExtractedRule[] rules = new ExtractedRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, mostCommonValueIndices);
        }
        return rules;
    }

    private static ExtractedRule[] fromSimpleCartToRules(String str, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        if(str.contains("CART Decision Tree")) {
             str = str.substring(
                str.indexOf("CART Decision Tree") + "CART Decision Tree".length(),
                str.indexOf("Number of Leaf Nodes")
            );
        }
        String[] lines = str.trim().split("\n");

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
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, mostCommonValueIndices);
        }
        return rules;
    }
    private static ExtractedRule[] fromSimpleCartToRules(SimpleCart clf, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        String str = clf.toString();
        return fromSimpleCartToRules(str, train_data, mostCommonValueIndices);
    }

    private static ExtractedRule[] fromJ48ToRules(String str, Instances train_data, double[] mostCommonValueIndices) throws Exception {
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
            rules[i] = new ExtractedRule(rule_lines.get(i), train_data, mostCommonValueIndices);
        }
        return rules;
    }
    private static ExtractedRule[] fromJ48ToRules(J48 clf, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        String str = clf.toString();
        return fromJ48ToRules(str, train_data, mostCommonValueIndices);
    }

    public static String formatNumericDecisionTableCell(String pre, String column_name) throws Exception {
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
                // TODO exception here!
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
                // TODO exception here!
                throw new Exception("pre must be closed either by a ) or a ] character!");
            }
        }
        return post_process;
    }

    public static ExtractedRule[] fromDecisionTableToRules(DecisionTable decisionTable, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        decisionTable.setDisplayRules(true);
        String str = decisionTable.toString();

        boolean useIbk = decisionTable.getUseIBk();

        String[] rules = str.split("Rules:")[1].trim().split("\n");

        ArrayList<String> headerColumns = new ArrayList<>();
        for (String header : rules[1].split(" ")) {
            if (header.length() > 0) {
                headerColumns.add(header);
            }
        }

        // - 4 for toprule, midrule and bottom rule + header
        // + 1 for default rule, that is not included in table
        ExtractedRule[] extractedRules = new ExtractedRule[rules.length - 4 + (useIbk? 0 : 1)];
        int counter = 0;

        StringBuffer prior = new StringBuffer("");
        if (headerColumns.size() > 1) {
            // why start at 3 and finish at -1? because rules[0] and rules[2] are only table delimiters
            // (i.e. ========), as well as rules[-1]. rules[1] contains the column headers of the table
            for (int i = 3; i < rules.length - 1; i++) {
                String[] priors = rules[i].split(" +");
                String posterior = priors[priors.length - 1];
                StringBuffer newline = new StringBuffer("");
                for(int j = 0; j < priors.length - 1; j++) {
                    String post = priors[j];
                    if(priors[j].contains("\'")) {
                        post = RuleExtractor.formatNumericDecisionTableCell(priors[j], headerColumns.get(j));
                    } else {
                        post = String.format("%s = %s", headerColumns.get(j), post);
                    }
                    newline.append(post);
                    if(j != priors.length - 2) {
                        newline.append(" and ");
                    }
                }
                newline.append(": ").append(posterior);

                extractedRules[counter] = new ExtractedRule(newline.toString(), train_data, mostCommonValueIndices);
                counter += 1;
            }
        }
        if(!useIbk) {
            int[] dist = train_data.attributeStats(train_data.classIndex()).nominalCounts;
            double max_index = -1, max = Double.NEGATIVE_INFINITY;
            for(int i = 0; i < dist.length; i++) {
                if(dist[i] > max) {
                    max = dist[i];
                    max_index = i;
                }
            }
            ExtractedRule rule = new ExtractedRule(String.format(": %s", train_data.attribute(train_data.classIndex()).value((int)max_index)), train_data, mostCommonValueIndices);

            if(counter > 0) {
                extractedRules[counter] = rule;
            } else {
                extractedRules = new ExtractedRule[]{rule};
            }
        } else if(counter == 0) {
            extractedRules = new ExtractedRule[0];
        }
        return extractedRules;
    }

    public static ExtractedRule[] fromRandomForestToRules(RandomForest rf, Instances train_data, double[] mostCommonValueIndices) throws Exception {
        rf.setPrintClassifiers(true);
        String str = rf.toString();

        String start = "RandomTree\n==========";
        String end = "Size of the tree";

        String trees_str = str.substring(
                str.indexOf(start),
                str.lastIndexOf(end) + end.length()
        ).trim();

        ArrayList<String> trees = new ArrayList<>();
        int begin_index = trees_str.indexOf(start) + start.length();
        int end_index = trees_str.indexOf(end);
        while(begin_index != -1) {
            String current = trees_str.substring(begin_index, end_index).trim();
            trees.add(current);
            begin_index = trees_str.indexOf(start, begin_index);
            if(begin_index != -1) {
                begin_index += start.length();
            } else {
                break;
            }
            end_index = trees_str.indexOf(end, begin_index);
        }

        ExtractedRule[][] unmerged_rules = new ExtractedRule[trees.size()][];
        int count_rules = 0;
        for(int i = 0; i < trees.size(); i++) {
            unmerged_rules[i] = RuleExtractor.fromSimpleCartToRules(trees.get(i), train_data, mostCommonValueIndices);
            count_rules += unmerged_rules[i].length;
        }

        ExtractedRule[] rules = new ExtractedRule[count_rules];
        int counter = 0;
        for(int i = 0; i < unmerged_rules.length; i++) {
            for(int j = 0; j < unmerged_rules[i].length; j++) {
                rules[counter] = unmerged_rules[i][j];
                counter += 1;
            }
        }
        return rules;
    }
}
