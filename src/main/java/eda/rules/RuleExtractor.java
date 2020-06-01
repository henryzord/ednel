package eda.rules;

import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Collections;

public class RuleExtractor {

    public RealRule[] fromJ48ToRules(J48 j48) {
        String str = j48.toString();
        String[] lines = (
                str.substring(
                    str.indexOf("------------------") + "------------------".length(),
                        str.indexOf("Number of Leaves")
                        )
        ).trim().split("\n");

        // levels has the level that each rule is within
        Condition[] conditions = new Condition[lines.length];
        ArrayList<Integer> levels = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            conditions[i] = Condition.fromJ48String(lines[i]);

            int count = 0, fromIndex = 0;
            // while there are still sub-levels within this rule
            while ((fromIndex = lines[i].indexOf("|", fromIndex)) != -1) {
                count++;
                fromIndex++;
            }
            levels.add(count);
//            lines[i] = lines[i].replaceAll("\\|", "").trim();
        }
//        int[][] conditionChains = new int[]

        int deepest_level = Collections.max(levels);
        for(int j = deepest_level; j > 0; j--) {
            int index_last_minus = -1;
            for(int i = 0; i < levels.size(); i++) {
                if(levels.get(i) == (j - 1)) {
                    index_last_minus = i;
                } else if(levels.get(i) == j) {
                    // TODo this process can be optimized
                    lines[i] = lines[index_last_minus] + " and " + lines[i];
                    levels.set(i, levels.get(i) - 1);
                }
            }
        }
        ArrayList<String> rule_lines = new ArrayList<>(lines.length);
        for(int i = 0; i < lines.length; i++) {
            if(lines[i].contains(":")) {
                rule_lines.add(lines[i]);
            }
        }

        RealRule[] rules = new RealRule[rule_lines.size()];
        for(int i = 0; i < rule_lines.size(); i++) {
            rules[i] = new RealRule(rule_lines.get(i));
        }
        return rules;
    }

    public ArrayList<RealRule> toRealRules(DecisionTable decisionTable) throws Exception {
        String str = this.toString();

        boolean usesIBk = decisionTable.getUseIBk();
        String[] rules = str.split("Rules:")[1].trim().split("\n");

        ArrayList<String> headerColumns = new ArrayList<>();
        for (String header : rules[1].split(" ")) {
            if (header.length() > 0) {
                headerColumns.add(header);
            }
        }

        ArrayList<RealRule> realRules = new ArrayList<>(rules.length - 3);

        StringBuffer prior = new StringBuffer("");
        if (headerColumns.size() > 1) {
            for (int i = 3; i < rules.length - 1; i++) {
                String line = rules[i].replaceAll("\'", "");
                String[] priors = line.split(" ");
                // TODO must have additional module for dealing with inf values!
                StringBuffer newline = new StringBuffer(priors[0]);
                String posterior = priors[priors.length - 1];
                for(int j = 1; j < priors.length - 1; j++) {
                    if(priors[j].length() > 0) {
                        newline.append(" and ").append(priors[j]);
                    }
                }
                newline.append(":").append(posterior);
                realRules.add(new RealRule(newline.toString()));
            }
            int z = 0;
        } else {
            throw new Exception("not implemented yet!");
        }
        // TODO append majority rule!
        throw new Exception("not implemented yet!");
    }

    public static void main(String[] args) {
        try {
            ConverterUtils.DataSource train_set = new ConverterUtils.DataSource("D:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv\\german\\german-10-1tra.arff");
            ConverterUtils.DataSource test_set = new ConverterUtils.DataSource("D:\\Users\\henry\\Projects\\ednel\\keel_datasets_10fcv\\german\\german-10-1tst.arff");

            RuleExtractor re = new RuleExtractor();

            Instances train_data = train_set.getDataSet(), test_data = test_set.getDataSet();
            train_data.setClassIndex(train_data.numAttributes() - 1);
            test_data.setClassIndex(test_data.numAttributes() - 1);

            J48 j48 = new J48();
            j48.buildClassifier(train_data);
            RealRule[] rules = re.fromJ48ToRules(j48);
            int[][] activated = new int[train_data.size()][rules.length];
            for(int i = 0; i < train_data.size(); i++) {
                for(int j = 0; j < rules.length; j++) {
                    activated[i][j] = rules[j].covers(train_data.get(i))? 1 : 0;
                }
            }
            int z = 0;

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
