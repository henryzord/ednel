package eda.individual;

import eda.aggregators.Aggregator;
import eda.aggregators.CompetenceBasedAggregator;
import eda.aggregators.MajorityVotingAggregator;
import eda.classifiers.trees.SimpleCart;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.util.ArrayList;
import java.util.HashMap;

public class Individual extends AbstractClassifier implements OptionHandler, Summarizable, TechnicalInformationHandler {

    public J48 j48;
    public SimpleCart simpleCart;
    public PART part;
    public JRip jrip;
    public DecisionTable decisionTable;
    public Aggregator aggregator;

    public String aggregatorName;

    protected int n_active_classifiers;
    protected Instances train_data;

    protected HashMap<String, String> characteristics = null;

    protected HashMap<String, AbstractClassifier> classifiers;

    public Individual() throws Exception {
        this.j48 = null;
        this.part = null;
        this.jrip = null;
        this.decisionTable = null;
        this.simpleCart = null;

        this.characteristics = new HashMap<>(51);  // approximate number of variables in the GM

        this.classifiers = new HashMap<>(6);
        this.classifiers.put("J48", null);
        this.classifiers.put("SimpleCart", null);
        this.classifiers.put("PART", null);
        this.classifiers.put("JRip", null);
        this.classifiers.put("DecisionTable", null);
    }

    public Individual(String[] options, HashMap<String, String> characteristics, Instances train_data) throws Exception {
        this();

        this.characteristics = (HashMap<String, String>)characteristics.clone();  // approximate number of variables in the GM
        if(Boolean.parseBoolean(this.characteristics.get("DecisionTable"))) {
            this.decisionTable = new DecisionTable();
            this.classifiers.put("DecisionTable", this.decisionTable);
        }
        if(Boolean.parseBoolean(this.characteristics.get("J48"))) {
            this.j48 = new J48();
            this.classifiers.put("J48", this.j48);
        }
        if(Boolean.parseBoolean(this.characteristics.get("SimpleCart"))) {
            this.simpleCart = new SimpleCart();
            this.classifiers.put("SimpleCart", this.simpleCart);
        }
        if(Boolean.parseBoolean(this.characteristics.get("PART"))) {
            this.part = new PART();
            this.classifiers.put("PART", this.part);
        }
        if(Boolean.parseBoolean(this.characteristics.get("JRip"))) {
            this.jrip = new JRip();
            this.classifiers.put("JRip", this.jrip);
        }
        this.setOptions(options);
        this.buildClassifier(train_data);
    }

//    public String[][] getClassifiersNames() {
//        return new String[][]{
//            {"j48", "J48", "Lweka/eda.classifiers/trees/J48;"},
//            {"simpleCart", "SimpleCart", "Lweka/eda.classifiers/trees/SimpleCart;"},
//            {"part", "PART", "Lweka/eda.classifiers/rules/PART;"},
//            {"jrip", "JRip", "Lweka/eda.classifiers/rules/JRip;"},
//            {"decisionTable", "DecisionTable", "Lweka/eda.classifiers/rules/DecisionTable;"}
//        };
//    }


    public HashMap<String, String> getCharacteristics() {
        return characteristics;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String[] j48Parameters = Utils.getOption("J48", options).split(" ");
        String[] simpleCartParameters = Utils.getOption("SimpleCart", options).split(" ");
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
            try {
                j48.setOptions(j48Parameters);
            } catch(Exception e) {
                throw new Exception("Exception found in Classifier J48: " + e.getMessage());
            }
        } else {
            j48 = null;
        }
        if(simpleCartParameters.length > 1) {
            try {
                simpleCart.setOptions(simpleCartParameters);
            } catch(Exception e) {
                throw new Exception("Exception found in Classifier SimpleCart: " + e.getMessage());
            }
        } else {
            simpleCart = null;
        }
        if(partParameters.length > 1) {
            try {
                part.setOptions(partParameters);
            } catch(Exception e) {
                throw new Exception("Exception found in Classifier PART: " + e.getMessage());
            }
        } else {
            part = null;
        }
        if(jripParameters.length > 1) {
            try {
                jrip.setOptions(jripParameters);
            } catch(Exception e) {
                throw new Exception("Exception found in Classifier JRip: " + e.getMessage());
            }
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

            try {
                decisionTable.setOptions(newDtParams);
            } catch(Exception e) {
                throw new Exception("Exception found in Classifier DecisionTable: " + e.getMessage());
            }
        } else {
            decisionTable = null;
        }

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train_data = data;

        n_active_classifiers = 0;
        Classifier[] clfs = new Classifier[]{j48, simpleCart, part, jrip, decisionTable};
        for(Classifier clf : clfs) {
            if(clf != null) {
                clf.buildClassifier(data);
                n_active_classifiers += 1;
            }
        }
        if(n_active_classifiers <= 0) {
            throw new Exception("Ensemble must contain at least one classifier!");
        }
        if(this.aggregator == null) {
            throw new Exception("Ensemble must have an aggregation policy!");
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
        Classifier[] clfs = new Classifier[]{j48, simpleCart, part, jrip, decisionTable};

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
        Classifier[] clfs = new Classifier[]{j48, simpleCart, part, jrip, decisionTable};

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

    public HashMap<String, AbstractClassifier> getClassifiers() {
        return classifiers;
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

