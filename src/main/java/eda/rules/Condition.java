package eda.rules;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class Condition extends AbstractClassifier {
    private final String attribute;
    private final String value;
//    private final String cls;
//    private final Double arrived;
//    private final Double incorrect;

    public Condition(String attribute, String value) {
        this.attribute = attribute;
        this.value = value;
//        this.cls = cls;

//        this.arrived = arrived != null? Double.parseDouble(arrived) : null;
//        this.incorrect = incorrect != null? Double.parseDouble(incorrect) : null;
    }

    public static Condition fromJ48String(String line) {
        int barIndex = line.lastIndexOf("|");
        int colonIndex = line.indexOf(":");

        String core = line.substring(barIndex == -1? 0 : barIndex, colonIndex == -1? line.length() : colonIndex).trim();
        String[] comparatives = {"=", "<=", ">", "<"};

        boolean finished = false;
        String[] pieces = null;
        for(String comparative : comparatives) {
            if(core.contains(comparative)) {
                pieces = core.split(comparative);
                finished = true;
                break;
            }
        }
//        String cls = null;
//        String arrived = null;
//        String incorrect = null;

//        if(colonIndex != -1) {
//            String consequence = line.split(":")[1].trim();
//            String[] subpieces = consequence.split(" ");
//            cls = subpieces[0].trim();
//
//            if(subpieces.length > 1) {
//                String[] subsubpieces = subpieces[1].trim().replace("(", "").replace(")", "").split("/");
//
//                arrived = subsubpieces[0].trim();
//                if(subsubpieces.length > 1) {
//                    incorrect = subsubpieces[1].trim();
//                }
//            }
//        }
        return new Condition(pieces[0].trim(), pieces[1].trim());
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

    }
}
