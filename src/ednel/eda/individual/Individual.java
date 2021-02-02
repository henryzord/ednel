package ednel.eda.individual;

import ednel.classifiers.trees.SimpleCart;
import ednel.eda.aggregators.Aggregator;
import ednel.eda.aggregators.RuleExtractorAggregator;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.reflections.Reflections;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.InvalidParameterException;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.TimeoutException;

public class Individual extends AbstractClassifier implements OptionHandler, Summarizable, TechnicalInformationHandler, Comparable<Individual> {

    protected J48 j48;
    protected SimpleCart simpleCart;
    protected PART part;
    protected JRip jrip;
    protected DecisionTable decisionTable;
    protected Aggregator aggregator;

    /** Fitness as measured when this individual was generated */
    protected Fitness fitness;

    /** How many seconds it took to train this individual */
    protected int timeToTrain;

    /** How many rules this individual has. */
    protected int n_rules;

    protected String aggregatorName;

    /** Number of classifiers currently being used by this ensemble. */
    protected int n_active_classifiers;
    protected Instances train_data;

    /** Array of characteristics of this individual. May contain null values. */
    protected HashMap<String, String> characteristics = null;

    protected HashMap<String, AbstractClassifier> classifiers;
    protected String[] orderedClassifiersNames = null;
    protected AbstractClassifier[] orderedClassifiers = null;

    protected HashMap<String, String> optionTable;

    protected Integer timeout_individual;

    private static HashMap<String, Class<? extends Aggregator>> aggregatorClasses;

    static {
        aggregatorClasses = Individual.getAggregatorClasses();
    }

    public Individual(HashMap<String, String> optionTable, HashMap<String, String> characteristics) throws
            EmptyEnsembleException, InvalidParameterException, NoAggregationPolicyException
    {

        this.classifiers = new HashMap<>(6);
        this.characteristics = new HashMap<>();
        for(String key : characteristics.keySet()) {
            this.characteristics.put(key, characteristics.get(key));
        }

        this.optionTable = new HashMap<>();
        for(String key: optionTable.keySet()) {
            this.optionTable.put(key, optionTable.get(key));
        }

        this.timeout_individual = null;

        String[] options = new String [optionTable.size() * 2];
        HashSet<String> algNames = new HashSet<>(optionTable.keySet());

        int counter = 0;
        for(String algName : algNames) {
            options[counter] = "-" + algName;
            String curVal = optionTable.get(algName);
            options[counter + 1] =  String.valueOf(curVal).equals("null")? "" : curVal;
            counter += 2;
        }

//        this.orderedClassifiers = new AbstractClassifier[]{j48, simpleCart, part, jrip, decisionTable};
        this.orderedClassifiersNames = new String[]{"JRip", "DecisionTable", "J48", "PART", "SimpleCart"};
        this.orderedClassifiers = new AbstractClassifier[]{jrip, decisionTable, j48, part, simpleCart};
        this.setOptions(options);
    }

    public Individual(HashMap<String, String> optionTable, HashMap<String, String> characteristics, Integer timeout_individual) throws
            EmptyEnsembleException, InvalidParameterException, NoAggregationPolicyException {
        this(optionTable, characteristics);
        this.timeout_individual = timeout_individual;
    }

    public Individual(Individual other) throws
            EmptyEnsembleException, InvalidParameterException, NoAggregationPolicyException {
        this(other.getOptionTable(), other.getCharacteristics());
    }

    public Individual(Individual other, Integer timeout_individual) throws
            EmptyEnsembleException, InvalidParameterException, NoAggregationPolicyException {
        this(other.getOptionTable(), other.getCharacteristics(), timeout_individual);
    }

    @Override
    public void setOptions(String[] options) throws EmptyEnsembleException, InvalidParameterException,
            NoAggregationPolicyException {
        String[] debug = options.clone();

        String[] j48Parameters;
        String[] simpleCartParameters;
        String[] partParameters;
        String[] jripParameters;
        String[] decisionTableParameters;
        String[] bestFirstParameters;
        String[] greedyStepwise;
        String[] aggregatorParameters;

        try {
            j48Parameters = Utils.getOption("J48", options).split(" ");
            simpleCartParameters = Utils.getOption("SimpleCart", options).split(" ");
            partParameters = Utils.getOption("PART", options).split(" ");
            jripParameters = Utils.getOption("JRip", options).split(" ");
            decisionTableParameters = Utils.getOption("DecisionTable", options).split(" ");
            bestFirstParameters = Utils.getOption("BestFirst", options).split(" ");
            greedyStepwise = Utils.getOption("GreedyStepwise", options).split(" ");
            aggregatorParameters = Utils.getOption("Aggregator", options).split(" ");
        } catch(Exception e) {
            throw new InvalidParameterException("error while processing hyper-parameters for ensemble classifiers.");
        }

        int n_queried_classifiers = 0;

        this.aggregatorName = aggregatorParameters[0];
        Class<? extends Aggregator> cls = Individual.aggregatorClasses.getOrDefault(aggregatorName, null);

        try {
            if(cls != null) {
                this.aggregator = (Aggregator)cls.getConstructor().newInstance();
            } else {
                throw new InvalidParameterException("Aggregator " + aggregatorParameters[0] + " not currently supported!");
            }
        } catch(NoSuchMethodException e) {
            throw new NoAggregationPolicyException("Aggregator " + aggregatorParameters[0] + " not currently supported!");
        } catch (IllegalAccessException | InstantiationException | InvocationTargetException e){
            throw new InvalidParameterException("An error occurred while trying to instantiate an aggregator for this ensemble.");
        }

        if(j48Parameters.length > 1) {
            try {
                j48 = new J48();
                j48.setOptions(j48Parameters);
                n_queried_classifiers += 1;
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier J48: " + e.getMessage());
            }
        } else {
            j48 = null;
        }
        if(simpleCartParameters.length > 1) {
            try {
                simpleCart = new SimpleCart();
                simpleCart.setOptions(simpleCartParameters);
                n_queried_classifiers += 1;
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier SimpleCart: " + e.getMessage());
            }
        } else {
            simpleCart = null;
        }
        if(partParameters.length > 1) {
            try {
                part = new PART();
                part.setOptions(partParameters);
                n_queried_classifiers += 1;
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier PART: " + e.getMessage());
            }
        } else {
            part = null;
        }
        if(jripParameters.length > 1) {
            try {
                jrip = new JRip();
                jrip.setOptions(jripParameters);
                n_queried_classifiers += 1;
            } catch(Exception e) {
                throw new InvalidParameterException("Exception found in Classifier JRip: " + e.getMessage());
            }
        } else {
            jrip = null;
        }
        try {
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
                    decisionTable = new DecisionTable();
                    decisionTable.setOptions(newDtParams);
                    n_queried_classifiers += 1;
                } catch(Exception e) {
                    throw new InvalidParameterException("Exception found in Classifier DecisionTable: " + e.getMessage());
                }
            } else {
                decisionTable = null;
            }
        } catch(Exception e) {
            throw new InvalidParameterException("an error occurred while trying to capture hyper-parameters for the " +
                    "Decision Table classifier of this ensemble.");
        }


        if(n_queried_classifiers == 0) {
            throw new EmptyEnsembleException("no classifier present in this ensemble!");
        }

        this.orderedClassifiersNames = new String[]{"JRip", "DecisionTable", "J48", "PART", "SimpleCart"};
        this.orderedClassifiers = new AbstractClassifier[]{jrip, decisionTable, j48, part, simpleCart};
        this.classifiers.put("J48", this.j48);
        this.classifiers.put("SimpleCart", this.simpleCart);
        this.classifiers.put("PART", this.part);
        this.classifiers.put("JRip", this.jrip);
        this.classifiers.put("DecisionTable", this.decisionTable);

        this.n_active_classifiers = 0;
    }

    @Override
    public void buildClassifier(Instances data) throws EmptyEnsembleException, NoAggregationPolicyException, TimeoutException {
        this.train_data = data;

        LocalDateTime start = LocalDateTime.now();

        this.n_active_classifiers = 0;
        for(int i = 0; i < this.orderedClassifiers.length; i++) {
            if(this.orderedClassifiers[i] != null) {
                try {
                    this.orderedClassifiers[i].buildClassifier(data);
                    n_active_classifiers += 1;
                } catch(Exception e) {
                    this.orderedClassifiers[i] = null;
                }
                if(this.isOvertime(start)) {
                    throw new TimeoutException("individual building is taking more than allowed time.");
                }
            }
        }
        if(this.n_active_classifiers <= 0) {
            throw new EmptyEnsembleException("Ensemble must contain at least one classifier!");
        }
        if(this.aggregator == null) {
            throw new NoAggregationPolicyException("Ensemble must have an aggregation policy!");
        }
        try {
            this.aggregator.setCompetences(this.orderedClassifiers, data);
        } catch(Exception e) {
            throw new NoAggregationPolicyException("Error while setting competences for aggregator.");
        }

        this.timeToTrain = (int)start.until(LocalDateTime.now(), ChronoUnit.SECONDS);
        if(this.aggregator instanceof RuleExtractorAggregator) {
            this.n_rules = ((RuleExtractorAggregator)this.aggregator).getNumberOfRules();
        } else {
            this.n_rules = 0;
            if(this.j48 != null) {
                String j48_summary = this.j48.toSummaryString();
                this.n_rules += Integer.parseInt(j48_summary.substring(
                    j48_summary.indexOf("Number of leaves: ") + "Number of leaves: ".length(),
                    j48_summary.indexOf("Size of the tree: ")
                ).replaceAll("\n", ""));
            }
            if(this.simpleCart != null) {
                this.n_rules += this.simpleCart.numLeaves();
            }
            if(this.jrip != null) {
                this.n_rules += this.jrip.getRuleset().size();
            }
            if(this.part != null) {
                this.n_rules += (int)this.part.measureNumRules();
            }
            if(this.decisionTable != null) {
                this.n_rules += (int)this.decisionTable.measureNumRules();
            }
        }
    }

    /**
     * Defines how many seconds an individual has to train all its base classifiers, or set to NULL to allow infinite
     * time.
     * @param seconds Either a valid Natural number (> 0) or NULL for unlimited time.
     */
    public void setTimeoutIndividual(Integer seconds) {
        if(seconds != null && seconds > 0) {
            this.timeout_individual = seconds;
        } else {
            this.timeout_individual = null;
        }
    }

    public Integer getTimeoutIndividual() {
        return this.timeout_individual;
    }


    private boolean isOvertime(LocalDateTime start) {
        return (this.timeout_individual != null) &&
                ((int)start.until(LocalDateTime.now(), ChronoUnit.SECONDS) > this.timeout_individual);
    }

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

    /**
     * Returns a copy of this individual's characteristics.
     */
    public HashMap<String, String> getCharacteristics() {
        HashMap<String, String> copy = new HashMap<>();
        for(String key : this.characteristics.keySet()) {
            copy.put(key, this.characteristics.get(key));
        }
        return copy;
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
        StringBuilder res = new StringBuilder();

        res.append("# Extracted rules\n\n");

        if(this.aggregator instanceof RuleExtractorAggregator) {
            res.append(this.aggregator.toString());
        } else {
            RuleExtractorAggregator agg = new RuleExtractorAggregator();
            try {
                agg.setCompetences(this.orderedClassifiers, this.train_data);
                res.append(agg.toString());
            } catch(Exception e) {
                // could not collect aggregated rules
            }
        }

        res.append("# Text representation of classifiers as-is\n\n");

        HashMap<String, AbstractClassifier> classifiers = this.getClassifiers();

        for(String clfName : classifiers.keySet()) {
            AbstractClassifier clf = classifiers.get(clfName);
            if(clf != null) {
                try {
                    res.append(Individual.formatClassifierString(clf));
                    res.append("\n\n");
                } catch (Exception ex) {
                    res.append(clf.toString());
                    res.append("\n\n");
                }
            }
        }
        return res.toString();
    }

    public void treeModelsToFiles(String file_name) {
        HashMap<String, AbstractClassifier> classifiers = this.getClassifiers();
        for(String clfName : classifiers.keySet()) {
            AbstractClassifier clf = classifiers.get(clfName);

            if(clf != null) {
                try {
                    Method graphMethod = clf.getClass().getMethod("graph");
                    String dotText = (String)graphMethod.invoke(clf);

                    String imageFilename = String.format("%s_%s_graph.png", file_name, clfName);
                    Graphviz.fromString(dotText).render(Format.PNG).toFile(new File(imageFilename));
                    // bw.write(String.format("# %s\n![](%s)\n", clfName, imageFilename));
//                } catch (NoSuchMethodException e) {
//                    return Individual.formatClassifierString(clf);
//                }
                } catch(Exception e) {
                    // nothing happens
                }
            }
        }
    }


    public static String formatClassifierString(AbstractClassifier clf) throws Exception {
        return (String)Individual.class.getMethod("format" + clf.getClass().getSimpleName() + "String", AbstractClassifier.class).invoke(Individual.class, clf);
    }

    @SuppressWarnings("unused")
    public static String formatJ48String(AbstractClassifier clf) throws Exception {
        try {
            String txt = clf.toString().split("------------------")[1].split("Number of Leaves")[0].trim();
            String[] branches = txt.split("\n");
            String body = "";
            for(int i = 0; i < branches.length; i++) {
                int depth = branches[i].split("\\|").length - 1;
                for(int j = 0; j < depth; j++) {
                    body += "\t";
                }
                body += "* " + branches[i].replaceAll("\\|  ", "").trim() + "\n";
            }
            String header = "## J48 Decision Tree";
            return String.format("%s\n\n%s", header, body);
        } catch(Exception e) {
            return clf.toString();
        }
    }
    @SuppressWarnings("unused")
    public static String formatSimpleCartString(AbstractClassifier clf) throws Exception  {
        try {
            String txt = clf.toString().split("CART Decision Tree")[1].split("Number of Leaf Nodes")[0].trim();
            String[] branches = txt.split("\n");
            String body = "";
            for(int i = 0; i < branches.length; i++) {
                int depth = branches[i].split("\\|").length - 1;
                for(int j = 0; j < depth; j++) {
                    body += "\t";
                }
                body += "* " + branches[i].replaceAll("\\|  ", "").trim() + "\n";
            }
            String header = "## SimpleCart Decision Tree";
            return String.format("%s\n\n%s", header, body);
        } catch(Exception e) {
            return clf.toString();
        }
    }
    @SuppressWarnings("unused")
    public static String formatJRipString(AbstractClassifier clf) throws Exception {
        try {
            String rulesStr = clf.toString().split("===========")[1].split("Number of Rules")[0].trim();
            String classAttrName = rulesStr.substring(rulesStr.lastIndexOf("=>") + 2, rulesStr.lastIndexOf("=")).trim();
            String[] rules = rulesStr.split("\n");
            String newRuleStr = "rules | predicted class\n---|---\n";
            for(int i = 0; i < rules.length; i++) {
                String[] partials = rules[i].split(String.format(" => %s=", classAttrName));
                for(String partial : partials) {
                    newRuleStr += partial.trim() + "|";
                }
                newRuleStr = newRuleStr.substring(0, newRuleStr.length() - 1) + "\n";
            }
            String r_str = String.format("## JRip\n\nDecision list:\n\n%s", newRuleStr);
            return r_str;
        } catch(Exception e) {
            return clf.toString();
        }
    }
    @SuppressWarnings("unused")
    public static String formatPARTString(AbstractClassifier clf) throws Exception {
        try {
            String defaultStr = clf.toString().split("------------------\\n\\n")[1];
            defaultStr = defaultStr.substring(0, defaultStr.lastIndexOf("Number of Rules"));
            String[] rules = defaultStr.split("\n\n");
            String newRuleStr = "rules | predicted class\n---|---\n";
            for(int i = 0; i < rules.length; i++) {
                String[] partials = rules[i].replace("\n", " ").split(":");
                for(String partial : partials) {
                    newRuleStr += partial.trim() + "|";
                }
                newRuleStr = newRuleStr.substring(0, newRuleStr.length() - 1) + "\n";
            }
            String r_str = String.format("## PART\n\nDecision list:\n\n%s", newRuleStr);
            return r_str;

        } catch (Exception e) {
            return clf.toString();
        }
    }
    @SuppressWarnings("unused")
    public static String formatDecisionTableString(AbstractClassifier clf) throws Exception {
        try {
            DecisionTable.class.getMethod("setDisplayRules", Boolean.TYPE).invoke(clf, true);

            Boolean usesIbk = (Boolean)DecisionTable.class.getMethod("getUseIBk").invoke(clf);

            String defaultString = "Non matches covered by " + (usesIbk? "IB1" : "Majority class");
            String[] lines = clf.toString().toLowerCase().replaceAll("\'", "").split("rules:")[1].split("\n");

            ArrayList<String> sanitized_lines = new ArrayList<String>(lines.length);

            int count_columns = 0;
            for(String line : lines) {
                if(line.contains("=")) {
                    if(sanitized_lines.size() == 1) {
                        String delimiter = "---";
                        for(int k = 1; k < count_columns; k++) {
                            delimiter += "|---";
                        }
                        sanitized_lines.add(delimiter);
                    }
                } else if ((line.length() > 0)) {
                    String[] columns = line.trim().split(" ");
                    ArrayList<String> sanitized_columns = new ArrayList<String>(columns.length);
                    count_columns = 0;
                    for(String column : columns) {
                        if (column.length() > 0) {
                            sanitized_columns.add(column);
                            count_columns += 1;
                        }
                    }
                    sanitized_lines.add(String.join("|", sanitized_columns));
                }
            }

            String table_str  = String.join("\n", sanitized_lines);

            String r_str = String.format("## Decision Table\n\n%s\n\n%s", defaultString, table_str);
            return r_str;
        } catch(Exception e) {
            return clf.toString();
        }
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

    public HashMap<String, String> getOptionTable() {
        return this.optionTable;
    }

    public int getNumberOfRules() {
        return this.n_rules;
    }

    public Fitness getFitness() {
        return fitness;
    }

    public void setFitness(Fitness fitness) {
        this.fitness = fitness;
    }

    @Override
    public int compareTo(Individual o) {
        double a = Math.round(this.getFitness().getLearnQuality() * 1e4) / 1e4;
        double b = Math.round(o.getFitness().getLearnQuality() * 1e4) / 1e4;

        int res = Double.compare(a, b);
        if(res == 0) {
            return Integer.compare(o.getFitness().getSize(), this.getFitness().getSize());
        }
        return res;
    }

    public String[] getOrderedClassifiersNames() {
        return orderedClassifiersNames;
    }

    public AbstractClassifier[] getOrderedClassifiers() {
        return this.orderedClassifiers;
    }
}

