package utils;

import dn.DependencyNetwork;
import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math3.genetics.Fitness;
import org.json.simple.JSONObject;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.core.Instances;

import javax.annotation.processing.FilerException;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;

public class PBILLogger {

    protected final String dataset_overall_path;
    protected final String dataset_thisrun_path;
    protected String dataset_metadata_path;
    protected Individual overall;
    protected Individual last;
    protected int n_sample;
    protected int n_fold;
    protected int n_individuals;
    protected boolean log;

    protected int curGen;

    protected ArrayList<Double> minFitness;
    protected ArrayList<Double> maxFitness;
    protected ArrayList<Double> medianFitness;

    protected HashMap<String, String> pastPopulations = null;

    protected static final String[] metricsToCollect = new String[]{
            "avgCost",
            "getClassPriors",
            "confusionMatrix",
            "correct",
            "errorRate",
            "incorrect",
            "kappa",
            "KBInformation",
            "KBMeanInformation",
            "KBRelativeInformation",
            "meanAbsoluteError",
            "meanPriorAbsoluteError",
            "numInstances",
            "pctCorrect",
            "pctIncorrect",
            "pctUnclassified",
            "relativeAbsoluteError",
            "rootMeanPriorSquaredError",
            "rootMeanSquaredError",
            "rootRelativeSquaredError",
            "SFEntropyGain",
            "SFMeanEntropyGain",
            "SFMeanPriorEntropy",
            "SFMeanSchemeEntropy",
            "SFPriorEntropy",
            "sizeOfPredictedRegions",
            "totalCost",
            "unclassified",
            "weightedAreaUnderPRC",
            "weightedAreaUnderROC",
            "weightedFMeasure",
            "weightedFalseNegativeRate",
            "weightedFalsePositiveRate",
            "weightedMatthewsCorrelation",
            "weightedPrecision",
            "weightedRecall",
            "weightedTrueNegativeRate",
            "weightedTruePositiveRate",
            "unweightedMacroFmeasure",
            "unweightedMicroFmeasure"
    };

    /**
     *
     * @param dataset_metadata_path
     * @param n_individuals
     * @param n_sample
     * @param n_fold
     * @param log Whether to implement more aggressive logging capabilities. May have a great impact in performance.
     */
    public PBILLogger(String dataset_metadata_path, int n_individuals, int n_generations, int n_sample, int n_fold, boolean log) {
        this.n_sample = n_sample;
        this.n_fold = n_fold;
        this.log = log;
        if(this.log) {
            this.pastPopulations = new HashMap<>(n_generations * n_individuals * 50);
        }

        this.dataset_metadata_path = dataset_metadata_path;
        this.dataset_overall_path = String.format(
                "%s%soverall%stest_sample-%02d_fold-%02d.csv",
                dataset_metadata_path, File.separator, File.separator, n_sample, n_fold
        );
        this.dataset_thisrun_path = String.format(
                "%s%ssample_%02d_fold_%02d",
                this.dataset_metadata_path, File.separator, this.n_sample, this.n_fold
        );

        this.n_individuals = n_individuals;

        this.minFitness = new ArrayList<>();
        this.medianFitness = new ArrayList<>();
        this.maxFitness = new ArrayList<>();

        this.curGen = 0;
    }

    public void logPopulation(Double[] fitnesses, Integer[] sortedIndices,
                              Individual[] population, Individual overall, Individual last
    ) {
        this.overall = overall;
        this.last = last;

        if(this.log) {
            for(int i = 0; i < population.length; i++) {
                for(String characteristic : population[i].getCharacteristics().keySet()) {
                    this.pastPopulations.put(String.format("gen_%03d_ind_%03d_%s", this.curGen, i, characteristic), population[i].getCharacteristics().get(characteristic));
                }
                this.pastPopulations.put(String.format("gen_%03d_ind_%03d_%s", this.curGen, i, "fitness"), String.valueOf(fitnesses[i]));
            }
        }
        this.minFitness.add(fitnesses[sortedIndices[fitnesses.length - 1]]);
        this.maxFitness.add(fitnesses[sortedIndices[0]]);

        if((fitnesses.length % 2) == 0) {
            int ind0 = (fitnesses.length / 2) - 1, ind1 = (fitnesses.length / 2);
            this.medianFitness.add((fitnesses[sortedIndices[ind0]] + fitnesses[sortedIndices[ind1]]) / 2);
        } else {
            this.medianFitness.add(fitnesses[sortedIndices[fitnesses.length / 2]]);
        }
        this.curGen += 1;
    }

    public void logProbabilities(DependencyNetwork dn) {

    }

    private String getEvaluationLineForClassifier(String name, Evaluation evaluation)
            throws InvocationTargetException, IllegalAccessException {

        Method[] overallMethods = evaluation.getClass().getMethods();
        HashMap<String, Method> overallMethodDict = new HashMap<>(overallMethods.length);

        for(Method method : overallMethods) {
            overallMethodDict.put(method.getName(), method);
        }

        String line = name;

        for(String methodName : metricsToCollect) {
            Object res = overallMethodDict.get(methodName).invoke(evaluation);
            if(res.getClass().isArray()) {
                try {
                    double[] doublyOverall = ((double[])res);

                    line += ",\"np.array([";
                    for(int k = 0; k < doublyOverall.length; k++) {
                        line += doublyOverall[k] + ",";
                    }
                    line = line.substring(0, line.lastIndexOf(",")) + "], dtype=np.float64)\"";
                } catch(ClassCastException e) {
                    double[][] doublyOverall = ((double[][])res);

                    line += ",\"np.array([[";
                    for(int j = 0; j < doublyOverall.length; j++) {
                        for(int k = 0; k < doublyOverall[j].length; k++) {
                            line += doublyOverall[j][k] + ",";
                        }
                        line = line.substring(0, line.lastIndexOf(",")) + "],[";
                    }
                    line = line.substring(0, line.lastIndexOf(",[")) + "], dtype=np.float64)\"";
                } catch(Exception e) {
                    throw(e);
                }

            } else {
                line += "," + res;
            }
        }
        return line + "," + FitnessCalculator.getUnweightedAreaUnderROC(evaluation) + "\n";
    }

    private Double[] evaluationsToFile(HashMap<String, Individual> individuals, Instances train_data, Instances test_data) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(this.dataset_overall_path));

//        HashMap<String, Evaluation> clfEvaluations = new HashMap<>(14);
        Double[] fitnesses = new Double [individuals.size()];

        // writes header
        String header = "algorithm";
        for(String methodName : metricsToCollect) {
            header += "," + methodName;
        }
        bw.write(header + "," + "unweightedAreaUnderRoc" + "\n");

        Evaluation evaluation;

        Object[] indNames = individuals.keySet().toArray();

        for(int i = 0; i < indNames.length; i++) {
            String indName = (String)indNames[i];
            evaluation = new Evaluation(train_data);
            evaluation.evaluateModel(individuals.get(indName), test_data);

            bw.write(this.getEvaluationLineForClassifier(indName, evaluation));

            HashMap<String, AbstractClassifier> overallClassifiers = individuals.get(indName).getClassifiers();

            for(String key: overallClassifiers.keySet()) {
                if(overallClassifiers.get(key) != null) {
                    evaluation = new Evaluation(train_data);
                    evaluation.evaluateModel(overallClassifiers.get(key), test_data);
                }
                fitnesses[i] = FitnessCalculator.getUnweightedAreaUnderROC(evaluation);
                bw.write(this.getEvaluationLineForClassifier(indName + "-" + key, evaluation));
            }
        }
        bw.close();
        return fitnesses;
    }

    public void toFile(HashMap<String, Individual> individuals, Instances train_data, Instances test_data) throws Exception {
        // TODo capture fitnesses here!
        Double[] fitnesses = evaluationsToFile(individuals, train_data, test_data);
        individualsCharacteristicsToFile(dataset_thisrun_path, individuals, fitnesses);
        individualsClassifiersToFile(dataset_thisrun_path, individuals);

        String variablesFolder = dataset_thisrun_path + File.separator + "variables";

//        this.createFolder(variablesFolder);
    }

    private void individualsClassifiersToFile(String thisRunFolder, HashMap<String, Individual> individuals) throws Exception {
        for(String indName : individuals.keySet()) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    thisRunFolder + File.separator + indName  + "_classifiers.md"
            ));

            HashMap<String, AbstractClassifier> classifiers = individuals.get(indName).getClassifiers();
            for(String clfName : classifiers.keySet()) {
                AbstractClassifier clf = classifiers.get(clfName);

                if(clf != null) {
                    // return (String)PBILLogger.class.getMethod("format" + clf.getClass().getSimpleName() + "String", AbstractClassifier.class).invoke(PBILLogger.class, clf);
                    try {
                        Method graphMethod = clf.getClass().getMethod("graph");
                        String dotText = (String)graphMethod.invoke(clf);

                        String imageFilename = String.format("%s_%s_graph.png", indName, clfName);
                        Graphviz.fromString(dotText).render(Format.PNG).toFile(new File(dataset_thisrun_path + File.separator + imageFilename));
                        bw.write(String.format("# %s\n![](%s)\n", clfName, imageFilename));
                    } catch (NoSuchMethodException e) {
                        String clfString = PBILLogger.formatClassifierString(clf);
                        bw.write(clfString + "\n\n\n");
                    }
                }
            }
            bw.close();
        }
    }

    private String getCharacteristicsLineForIndividual(Object[] order, String indName, HashMap<String, String> characteristics, Double aucs) {
        String line = indName;
        for(Object ch : order) {
            line += "," + characteristics.get(ch);
        }
        line += ",null," + aucs;
        return line + "\n";
    }

    private void individualsCharacteristicsToFile(String thisRunFolder, HashMap<String, Individual> individuals, Double[] fitnesses) throws IOException {
        PBILLogger.createFolder(thisRunFolder);
        BufferedWriter bw = new BufferedWriter(new FileWriter(thisRunFolder + File.separator + "characteristics.csv"));

        // writes header, saves order of columns
        Object[] order = individuals.get(individuals.keySet().toArray()[0]).getCharacteristics().keySet().toArray();
        String header = "individual_name";
        for(int i = 0; i < order.length; i++) {
            header += "," + order[i];
        }
        bw.write(header + ",fitness,test_auc\n");

        Object[] indNames = individuals.keySet().toArray();

        for(int i = 0; i < indNames.length; i++) {
            String indName = (String)indNames[i];
            HashMap<String, String> characteristics = individuals.get(indName).getCharacteristics();
            bw.write(this.getCharacteristicsLineForIndividual(order, indName, characteristics, fitnesses[i]));
        }
        if(this.log) {
            for(int i = 0; i < this.curGen; i++) {
                for(int j = 0; j < this.n_individuals; j++) {
                    String line = String.format("gen_%03d_ind_%03d", i, j);
                    for(Object ch : order) {
                        line += "," + pastPopulations.get(String.format("gen_%03d_ind_%03d_%s", i, j, ch));
                    }
                    line += "," + pastPopulations.get(String.format("gen_%03d_ind_%03d_%s", i, j, "fitness")) + ",null";
                    bw.write(line + "\n");
                }
            }
        }
        bw.close();
    }

    public static void metadata_path_start(String str_time, CommandLine commandLine) throws ParseException, IOException {
        String[] dataset_names = commandLine.getOptionValue("datasets_names").split(",");
        String metadata_path = commandLine.getOptionValue("metadata_path");

        // create one folder for each dataset
        PBILLogger.createFolder(metadata_path + File.separator + str_time);
        for(String dataset : dataset_names) {
            PBILLogger.createFolder(metadata_path + File.separator + str_time + File.separator + dataset);
            PBILLogger.createFolder(metadata_path + File.separator + str_time + File.separator + dataset + File.separator + "overall");
        }

        JSONObject obj = new JSONObject();
        for(Option parameter : commandLine.getOptions()) {
            obj.put(parameter.getLongOpt(), parameter.getValue());
        }

        FileWriter fw = new FileWriter(metadata_path + File.separator + str_time + File.separator + "parameters.json");
        fw.write(obj.toJSONString());
        fw.flush();
    }

    private static void createFolder(String path) throws FilerException {
        File file = new File(path);
        boolean successful = file.mkdir();
        if(!successful) {
            throw new FilerException("could not create directory " + path);
        }
    }

    public void print() {
        System.out.println(String.format(
                "%d\t\t\t%d\t\t\t%.8f\t\t\t%.8f\t\t\t%.8f",
                this.curGen,
                this.n_individuals,
                this.minFitness.get(this.curGen - 1),
                this.medianFitness.get(this.curGen - 1),
                this.maxFitness.get(this.curGen - 1)
        ));
    }

    public static String formatClassifierString(AbstractClassifier clf) throws Exception {
        return (String)PBILLogger.class.getMethod("format" + clf.getClass().getSimpleName() + "String", AbstractClassifier.class).invoke(PBILLogger.class, clf);
    }

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
            String header = "# J48 Decision Tree";
            return String.format("%s\n\n%s", header, body);
        } catch(Exception e) {
            return clf.toString();
        }
    }
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
            String header = "# SimpleCart Decision Tree";
            return String.format("%s\n\n%s", header, body);
        } catch(Exception e) {
            return clf.toString();
        }
    }
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
            String r_str = String.format("# JRip\n\nDecision list:\n\n%s", newRuleStr);
            return r_str;
      } catch(Exception e) {
            return clf.toString();
        }
    }
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
            String r_str = String.format("# PART\n\nDecision list:\n\n%s", newRuleStr);
            return r_str;

        } catch (Exception e) {
            return clf.toString();
        }
    }
    public static String formatDecisionTableString(AbstractClassifier clf) throws Exception {
        try {
            Boolean usesIbk = (Boolean) DecisionTable.class.getMethod("getUseIBk").invoke(clf);

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

            String r_str = String.format("# Decision Table\n\n%s\n\n%s", defaultString, table_str);
            return r_str;
        } catch(Exception e) {
            return clf.toString();
        }
    }

}
