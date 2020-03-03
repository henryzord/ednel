package eda.rules;

import weka.classifiers.rules.DecisionTable;
import eda.rules.RealRule;
import weka.classifiers.trees.J48;

import java.util.ArrayList;
import java.util.Collections;

public class RuleExtractor {

    public ArrayList<RealRule> fromJ48ToRules(J48 j48) {
        String str = j48.toString();
        String[] lines = (
                str.substring(
                    str.indexOf("------------------") + "------------------".length(),
                        str.indexOf("Number of Leaves")
                        )
        ).trim().split("\n");

        ArrayList<Integer> levels = new ArrayList<>(lines.length);

        StringBuffer prior = new StringBuffer("");
        for(int i = 0; i < lines.length; i++) {
            int count = 0, fromIndex = 0;
            while ((fromIndex = lines[i].indexOf("|", fromIndex)) != -1) {
                count++;
                fromIndex++;
            }
            levels.add(count);
            lines[i] = lines[i].replaceAll("\\|", "").trim();
        }
        int max_value = Collections.max(levels);
        for(int j = max_value; j > 0; j--) {
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
                rule_lines.add(lines[i]);
            }
        }

        ArrayList<RealRule> rules = new ArrayList<>(rule_lines.size());
        for(String line : rule_lines) {
            RealRule thisRule = new RealRule(line);
            rules.add(thisRule);
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
}
