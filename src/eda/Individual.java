package eda;

import eda.aggregators.Aggregator;
import eda.aggregators.CompetenceBasedAggregator;
import eda.aggregators.MajorityVotingAggregator;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import eda.trees.SimpleCart;
import weka.core.*;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Individual extends AbstractClassifier implements OptionHandler, Summarizable, TechnicalInformationHandler {

    public J48 j48;
    public SimpleCart simpleCart;
    public REPTree repTree;
    public PART part;
    public JRip jrip;
    public DecisionTable decisionTable;
    public Aggregator aggregator;

    public String aggregatorName;

    protected int n_active_classifiers;
    protected Instances train_data;

    protected HashMap<String, String> characteristics = null;


    public Individual() throws Exception {
        this.j48 = new J48();
        this.part = new PART();
        this.repTree = new REPTree();
        this.jrip = new JRip();
        this.decisionTable = new DecisionTable();
        this.simpleCart = new SimpleCart();

        this.characteristics = new HashMap<>(51);  // approximate number of variables in the GM
    }

    public Individual(String[] options, HashMap<String, String> characteristics, Instances train_data) throws Exception {
        this.j48 = new J48();
        this.part = new PART();
        this.repTree = new REPTree();
        this.jrip = new JRip();
        this.decisionTable = new DecisionTable();
        this.simpleCart = new SimpleCart();

        this.setOptions(options);
        this.buildClassifier(train_data);

        this.characteristics = (HashMap<String, String>)characteristics.clone();  // approximate number of variables in the GM
    }

//    public String[][] getClassifiersNames() {
//        return new String[][]{
//            {"j48", "J48", "Lweka/classifiers/trees/J48;"},
//            {"simpleCart", "SimpleCart", "Lweka/classifiers/trees/SimpleCart;"},
//            {"repTree", "REPTree", "Lweka/classifiers/trees/REPTree;"},
//            {"part", "PART", "Lweka/classifiers/rules/PART;"},
//            {"jrip", "JRip", "Lweka/classifiers/rules/JRip;"},
//            {"decisionTable", "DecisionTable", "Lweka/classifiers/rules/DecisionTable;"}
//        };
//    }

//    protected String[] fromCharacteristicsToOptions() throws Exception {
//        Set<String> all_options = classifiersResources.keySet();
//
//        HashMap<String, ArrayList<String>> optionTable = new HashMap<>(classifiersResources.size());
//        optionTable.put("J48", new ArrayList<String>());
//        optionTable.put("SimpleCart", new ArrayList<String>());
//        optionTable.put("PART", new ArrayList<String>());
//        optionTable.put("JRip", new ArrayList<String>());
//        optionTable.put("DecisionTable", new ArrayList<String>());
//        optionTable.put("BestFirst", new ArrayList<String>());
//        optionTable.put("GreedyStepwise", new ArrayList<String>());
//        optionTable.put("Aggregator", new ArrayList<String>());
//
//        // characteristics has more items than options, because classifiers are not options in Weka
//        for (Iterator<String> it = all_options.iterator(); it.hasNext(); ) {
//            String option = it.next();
//            String algorithmName = option.split("_")[0];
//
//            // if is null, then option was not set
//            if(characteristics.get(option) != null) {
//                if(dtype.equals("np.bool")) {
//                    if(Boolean.valueOf(characteristics.get(option))) {
//                        if(presenceMeans) {
//                            optionTable.get(algorithmName).add(optionName);
//                        }
//                    } else {
//                        if(!presenceMeans) {
//                            optionTable.get(algorithmName).add(optionName);
//                        }
//                    }
//                } else {
//                    optionTable.get(algorithmName).add(optionName + " " + characteristics.get(option));
//                }
//            }
//        }
//
//        String[] algorithms = new String[] {"J48", "SimpleCart", "PART", "JRip", "DecisionTable"};
//
//        String[] options = new String [(algorithms.length + 1) * 2];  // +1 for aggregator
//        ArrayList<String> theseOptions;
//        String optionsString = "";
//
//        int counter = 0;
//        for (String algorithm : algorithms) {
//            options[counter] = "-" + algorithm;
//            counter++;
//            optionsString = "";
//            theseOptions = optionTable.get(algorithm);
//            for(int j = 0; j < theseOptions.size(); j++) {
//                optionsString += " " + theseOptions.get(j);
//            }
//            options[counter] = optionsString;
//            counter++;
//        }
//        counter--;
//
//
//        // TODO probably will break if decision table is not present
//        optionsString += " -S weka.attributeSelection.";
//        if(optionTable.get("BestFirst").size() > 0) {
//            optionsString= optionsString.replace("-S BestFirst", "");
//            optionsString += "BestFirst";
//            theseOptions = optionTable.get("BestFirst");
//        } else {
//            optionsString= optionsString.replace("-S GreedyStepwise", "");
//            optionsString += "GreedyStepwise";
//            theseOptions = optionTable.get("GreedyStepwise");
//        }
//        for(int i = 0; i < theseOptions.size(); i++) {
//            optionsString += " " + theseOptions.get(i);
//        }
//        options[counter] = optionsString;
//        options[counter + 1] = "-Aggregator";
//        options[counter + 2] = characteristics.get("Aggregator");
//
//        return options;
//    }

    public HashMap<String, String> getCharacteristics() {
        return characteristics;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String[] oldOptions = options.clone();  // TODO remove

        String[] j48Parameters = Utils.getOption("J48", options).split(" ");
        String[] simpleCartParameters = Utils.getOption("SimpleCart", options).split(" ");
        String[] reptreeParameters = Utils.getOption("REPTree", options).split(" ");
        String[] partParameters = Utils.getOption("PART", options).split(" ");
        String[] jripParameters = Utils.getOption("JRip", options).split(" ");
        String[] decisionTableParameters = Utils.getOption("DecisionTable", options).split(" ");
        String[] bestFirstParameters = Utils.getOption("BestFirst", options).split(" ");
        String[] greedyStepwise = Utils.getOption("GreedyStepwise", options).split(" ");
        String[] aggregatorParameters = Utils.getOption("Aggregator", options).split(" ");

        aggregatorName = aggregatorParameters[0];
        if(aggregatorName.equals("MajorityVotingAggregator")) {
            this.aggregator = new MajorityVotingAggregator();
        } else if (aggregatorName.equals("CompetenceBasedAggregator")) {
            this.aggregator = new CompetenceBasedAggregator();
        } else {
            throw new Exception("Aggregator " + aggregatorParameters[0] + " not currently supported!");
        }

        if(j48Parameters.length > 1) {
            j48.setOptions(j48Parameters);
        } else {
            j48 = null;
        }
        if(simpleCartParameters.length > 1) {
            simpleCart.setOptions(simpleCartParameters);
        } else {
            simpleCart = null;
        }
        if(reptreeParameters.length > 1) {
            repTree.setOptions(reptreeParameters);
        } else {
            repTree = null;
        }
        if(partParameters.length > 1) {
            part.setOptions(partParameters);
        } else {
            part = null;
        }
        if(jripParameters.length > 1) {
            jrip.setOptions(jripParameters);
        } else {
            jrip = null;
        }
        if(decisionTableParameters.length > 1) {
            String dtSearch = Utils.getOption("S", decisionTableParameters);
            String dtSearchName = dtSearch.substring(dtSearch.lastIndexOf(".") + 1);
            String[] selectedSubParams = null;
            if(dtSearchName.equals("BestFirst")) {
                selectedSubParams = bestFirstParameters;
            } else if(dtSearchName.equals("GreedyStepwise")) { ;
                selectedSubParams = greedyStepwise;
            } else {
                throw new Exception("Search procedure for DecisionTable not found!");
            }
            String[] newDtParams = new String [decisionTableParameters.length + 2];
            int counter = 0;
            for(int i = 0; i < decisionTableParameters.length; i++) {
                newDtParams[counter] = decisionTableParameters[i];
                counter += 1;
            }
            newDtParams[counter] = "-S";
            newDtParams[counter + 1] = dtSearch;
            counter += 1;
            for(int i = 0; i < selectedSubParams.length; i++) {
                newDtParams[counter] += " " + selectedSubParams[i];
            }

            decisionTable.setOptions(newDtParams);
//            String[] newDtParameters = new String [decisionTableParameters.length + ]

//            decisionTableParameters = decisionTableParameters + " -S " + dtSearch;


//            StringBuffer dtSearchParameters = new StringBuffer("");
//            int n_itens = decisionTableParameters.length;
//            boolean concating = false;
//            ArrayList<String> buffer = new ArrayList<>();
//            for(int i = 0; i < n_itens; i++) {
//                if(decisionTableParameters[i].equals("-S")) {
//                    concating = true;
//                    buffer.add(decisionTableParameters[i]);
//                    continue;
//                }
//                if(concating) {
//                    dtSearchParameters.append(" " + decisionTableParameters[i]);
//                } else {
//                    buffer.add(decisionTableParameters[i]);
//                }
//            }
//            buffer.add(dtSearchParameters.toString());
//            String[] newDecisionTableParameters = new String[buffer.size()];
//            System.arraycopy(buffer.toArray(), 0, newDecisionTableParameters, 0, buffer.size());

        } else {
            decisionTable = null;
        }

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train_data = data;

        n_active_classifiers = 0;
        Classifier[] clfs = new Classifier[]{j48, simpleCart, repTree, part, jrip, decisionTable};
        for(Classifier clf : clfs) {
            if(clf != null) {
                clf.buildClassifier(data);
                n_active_classifiers += 1;
            }
        }
        double[] competences = null;
        if(aggregatorName.equals("CompetenceBasedAggregator")) {
            competences = new double[n_active_classifiers];
            int i = 0, counter = 0;
            while(counter < n_active_classifiers) {
                if(clfs[i] != null) {
                    Evaluation evaluation = new Evaluation(train_data);
                    evaluation.evaluateModel(clfs[i], train_data);
                    competences[counter] = FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
                    counter += 1;

                }
                i += 1;
            }
        }
        aggregator.setCompetences(competences);
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
        Classifier[] clfs = new Classifier[]{j48, simpleCart, repTree, part, jrip, decisionTable};

        double[][][] dists = new double[this.n_active_classifiers][][];
        int i = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[i] != null) {
                dists[counter] = new double[][]{clfs[i].distributionForInstance(instance)};
                counter += 1;
            }
            i += 1;
        }
        return this.aggregator.aggregateProba(dists)[0];
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        Classifier[] clfs = new Classifier[]{j48, simpleCart, repTree, part, jrip, decisionTable};

        double[][][] dists = new double[this.n_active_classifiers][][];
        int i = 0, counter = 0;
        while(counter < n_active_classifiers) {
            if(clfs[i] != null) {
                dists[counter] = ((AbstractClassifier) clfs[i]).distributionsForInstances(batch);
                counter += 1;
            }
            i += 1;
        }
        return this.aggregator.aggregateProba(dists);
    }

    @Override
    public String[] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        options.add("-J48");
        options.add(String.join(" ", j48 == null? new String[]{""} : j48.getOptions()));
        options.add("-SimpleCart");
        options.add(String.join(" ", simpleCart == null? new String[]{""} : simpleCart.getOptions()));
        options.add("-REPTree");
        options.add(String.join(" ", repTree == null? new String[]{""} : repTree.getOptions()));
        options.add("-PART");
        options.add(String.join(" ", part == null? new String[]{""} : part.getOptions()));
        options.add("-JRip");
        options.add(String.join(" ", jrip == null? new String[]{""} : jrip.getOptions()));
        options.add("-DecisionTable");
        options.add(String.join(" ", decisionTable == null? new String[]{""} : decisionTable.getOptions()));
        options.add("-Aggregator");
        options.add(this.aggregator.getClass().getName());
        options.add(String.join(" ", aggregator.getOptions()));

        String[] strOptions = options.toArray(new String[options.size()]);

        return strOptions;
    }

    @Override
    public String toString() {
        return "method not implemented yet";
    }

    @Override
    public String toSummaryString() {
        // TODO implement to return tables/trees
        return null;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        // TODO implement once published
        return null;
    }
}
