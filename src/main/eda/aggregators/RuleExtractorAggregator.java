package eda.aggregators;

import eda.classifiers.trees.SimpleCart;
import eda.rules.RealRule;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

public class RuleExtractorAggregator extends Aggregator implements Serializable {

    @Override
    protected void setOptions(Object... args) {
        
    }

    @Override
    public double[][] aggregateProba(double[][][] distributions) {
        return new double[0][];
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }


    public static RealRule[] fromClassifierToRules(AbstractClassifier clf, Instances train_data) throws Exception {
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

    private static RealRule[] fromPARTToRules(PART clf, Instances train_data) throws Exception {
        String str = clf.toString();

        String[] lines = str.substring(
                str.indexOf("------------------") + "------------------".length(),
                str.indexOf("Number of Rules")
        ).trim().split("\n\n");

        ArrayList<String> rule_lines = new ArrayList<>(lines.length);

        for(int i = 0; i < lines.length; i++) {
            rule_lines.add(lines[i].replaceAll("AND\n", "and "));
        }

        RealRule[] rules = new RealRule[rule_lines.size()];
        ArrayList<RealRule> cumulativeRules = new ArrayList<>();
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new RealRule(rule_lines.get(i), train_data, i == 0? null : cumulativeRules);
            cumulativeRules.add(rules[i]);
        }
        return rules;
    }

    private static RealRule[] fromJRipToRules(JRip clf, Instances train_data) throws Exception {
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
        RealRule[] rules = new RealRule[rule_lines.size()];
        ArrayList<RealRule> cumulativeRules = new ArrayList<>();
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new RealRule(rule_lines.get(i), train_data, i == 0? null : cumulativeRules);
            cumulativeRules.add(rules[i]);
        }
        return rules;
    }

    private static RealRule[] fromSimpleCartToRules(SimpleCart clf, Instances train_data) throws Exception {
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

        RealRule[] rules = new RealRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new RealRule(rule_lines.get(i), train_data, null);
        }
        return rules;
    }

    private static RealRule[] fromJ48ToRules(J48 clf, Instances train_data) throws Exception {
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

        RealRule[] rules = new RealRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new RealRule(rule_lines.get(i), train_data, null);
        }
        return rules;
    }

    private static String formatNumericDecisionTableCell(String pre) throws Exception {
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
                post_process += "%s > " + parts[0];
            } else if(opening_char == '[') {
                post_process += "%s >= " + parts[0];
            } else {
                throw new Exception("pre must be opened either by a ( or a [ character!");
            }
            if(!parts[1].equals("inf")) {
                post_process += " and ";
            }
        }
        if(!parts[1].equals("inf")) {
            if(closing_char == ')') {
                post_process += "%s < " + parts[1];
            } else if (closing_char == ']') {
                post_process += "%s <= " + parts[1];
            } else {
                throw new Exception("pre must be closed either by a ) or a ] character!");
            }
        }
        return post_process;
    }

    public static RealRule[] fromDecisionTableToRules(DecisionTable decisionTable, Instances train_data) throws Exception {
        decisionTable.setDisplayRules(true);
        String str = decisionTable.toString();

        String[] rules = str.split("Rules:")[1].trim().split("\n");

        ArrayList<String> headerColumns = new ArrayList<>();
        for (String header : rules[1].split(" ")) {
            if (header.length() > 0) {
                headerColumns.add(header);
            }
        }

        RealRule[] realRules = new RealRule[rules.length - 4];

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
                        post = RuleExtractorAggregator.formatNumericDecisionTableCell(priors[j]);
                        post = String.format(post, headerColumns.get(j));
                    } else {
                        post = String.format("%s = %s", headerColumns.get(j), post);
                    }
                    newline.append(post);
                    if(j != priors.length - 2) {
                        newline.append(" and ");
                    }
                }
                newline.append(": ").append(posterior);

                realRules[counter] = new RealRule(newline.toString(), train_data, null);
                counter += 1;
            }
        } else {  // has only default rule!
            return new RealRule[0];
        }
        return realRules;
    }

    public static void main(String[] args) {
        try {
//            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource("D:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv\\german\\german-10-1tra.arff");

            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource("C:\\Users\\henry\\Desktop\\play_tennis.arff");

            AbstractClassifier[] clfs = new AbstractClassifier[]{new JRip(), new PART(), new J48(), new DecisionTable(), new SimpleCart()};

            Instances train_data = train_set.getDataSet();
            train_data.setClassIndex(train_data.numAttributes() - 1);

            RealRule[][] all_rules = new RealRule[clfs.length][];

            for(int i = 0; i < clfs.length; i++) {
                clfs[i].buildClassifier(train_data);
                all_rules[i] = RuleExtractorAggregator.fromClassifierToRules(clfs[i], train_data);
            }
            for(int c = 0; c < clfs.length; c++) {
                for(int r = 0; r < all_rules[c].length; r++) {
                    if(all_rules[c][r].covers(train_data.get(0))) {
                        System.out.println(String.format("rule %d from classifier %d: %s", r, c, all_rules[c][r]));
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
