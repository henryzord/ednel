package ednel.eda.individual;

import ednel.classifiers.trees.SimpleCart;
import ednel.eda.aggregators.Aggregator;
import org.reflections.Reflections;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.security.InvalidParameterException;
import java.util.*;

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
    protected AbstractClassifier[] orderedClassifiers = null;

    private static HashMap<String, Class<? extends Aggregator>> aggregatorClasses;

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

        this.orderedClassifiers = new AbstractClassifier[]{j48, simpleCart, part, jrip, decisionTable};
    }

    public Individual(HashMap<String, String> optionTable, HashMap<String, String> characteristics) throws Exception {
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

        String[] options = new String [optionTable.size() * 2];
        HashSet<String> algNames = new HashSet<>();
        algNames.addAll(optionTable.keySet());
//        algNames.remove("BestFirst");
//        algNames.remove("GreedyStepwise");
//        algNames.remove("DecisionTable");

        int counter = 0;
        for(String algName : algNames) {
            options[counter] = "-" + algName;
            String curVal = optionTable.get(algName);
            options[counter + 1] =  String.valueOf(curVal).equals("null")? "" : curVal;
            counter += 2;
        }
        // process decision table
//        options[counter] = "-" + "DecisionTable";
//        if(characteristics.get("DecisionTable").equals("true")) {
//            counter += 1;
//            String[] dtOptions = optionTable.get("DecisionTable").split(" ");
//            String searchAlg_full = Utils.getOption("S", dtOptions);
//            String searchAlg_short = searchAlg_full.split("\\.")[2];
//
//            options[counter] = (" " + String.join(" ", dtOptions).trim() +
//                    " -S " + searchAlg_full + " " + optionTable.getOrDefault(searchAlg_short, "")).trim();
//        }

        this.setOptions(options);
//        this.buildClassifier(train_data);
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

    public static HashMap<String, Class<? extends Aggregator>> getAggregatorClasses() {
        if(Individual.aggregatorClasses == null) {
            Individual.aggregatorClasses = new HashMap<>();

            Reflections reflections = new Reflections("ednel.eda.aggregators");
            Set<Class<? extends Aggregator>> allClasses = reflections.getSubTypesOf(Aggregator.class);
            for (Iterator<Class<? extends Aggregator>> it = allClasses.iterator(); it.hasNext(); ) {
                Class<? extends Aggregator> agg = it.next();
                String[] splitted = agg.getName().split("\\.");
                String agg_name = splitted[splitted.length - 1];
                Individual.aggregatorClasses.put(agg_name, agg);
            }
        }
        return Individual.aggregatorClasses;

    }

    public HashMap<String, String> getCharacteristics() {
        return characteristics;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String[] debug = options.clone();

        String[] j48Parameters = Utils.getOption("J48", options).split(" ");
        String[] simpleCartParameters = Utils.getOption("SimpleCart", options).split(" ");
        String[] partParameters = Utils.getOption("PART", options).split(" ");
        String[] jripParameters = Utils.getOption("JRip", options).split(" ");
        String[] decisionTableParameters = Utils.getOption("DecisionTable", options).split(" ");
        String[] bestFirstParameters = Utils.getOption("BestFirst", options).split(" ");
        String[] greedyStepwise = Utils.getOption("GreedyStepwise", options).split(" ");
        String[] aggregatorParameters = Utils.getOption("Aggregator", options).split(" ");

        this.aggregatorName = aggregatorParameters[0];
        HashMap<String, Class<? extends Aggregator>> aggrClass = Individual.getAggregatorClasses();
        Class<? extends Aggregator> cls = aggrClass.getOrDefault(aggregatorName, null);

        if(cls != null) {
            this.aggregator = (Aggregator)cls.getConstructor().newInstance();
        } else {
            throw new InvalidParameterException("Aggregator " + aggregatorParameters[0] + " not currently supported!");
        }

        if(j48Parameters.length > 1) {
            try {
                j48.setOptions(j48Parameters);
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier J48: " + e.getMessage());
            }
        } else {
            j48 = null;
        }
        if(simpleCartParameters.length > 1) {
            try {
                simpleCart.setOptions(simpleCartParameters);
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier SimpleCart: " + e.getMessage());
            }
        } else {
            simpleCart = null;
        }
        if(partParameters.length > 1) {
            try {
                part.setOptions(partParameters);
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier PART: " + e.getMessage());
            }
        } else {
            part = null;
        }
        if(jripParameters.length > 1) {
            try {
                jrip.setOptions(jripParameters);
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier JRip: " + e.getMessage());
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
                throw new InvalidParameterException("Search procedure for DecisionTable not found!");
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
                throw new InvalidParameterException("Exception found in Classifier DecisionTable: " + e.getMessage());
            }
        } else {
            decisionTable = null;
        }

        this.orderedClassifiers = new AbstractClassifier[]{j48, simpleCart, part, jrip, decisionTable};
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        train_data = data;

        this.n_active_classifiers = 0;
        for(int i = 0; i < this.orderedClassifiers.length; i++) {
            if(this.orderedClassifiers[i] != null) {
                try {
                    this.orderedClassifiers[i].buildClassifier(data);
                    n_active_classifiers += 1;
                }  catch(Exception e) {
                    this.orderedClassifiers[i] = null;
                }
            }
        }
        if(n_active_classifiers <= 0) {
            throw new EmptyEnsembleException("Ensemble must contain at least one classifier!");
        }
        if(this.aggregator == null) {
            throw new NoAggregationPolicyException("Ensemble must have an aggregation policy!");
        }
        this.aggregator.setCompetences(this.orderedClassifiers, data);
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
        return this.aggregator.aggregateProba(this.orderedClassifiers, instance);
    }

    @Override
    public double[][] distributionsForInstances(Instances batch) throws Exception {
        return this.aggregator.aggregateProba(this.orderedClassifiers, batch);
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

