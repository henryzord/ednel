package utils;

import dn.DependencyNetwork;
import eda.classifiers.trees.SimpleCart;
import eda.individual.FitnessCalculator;
import eda.individual.Individual;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.ParseException;
import org.json.simple.JSONObject;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.PART;
import weka.core.Instances;

import weka.classifiers.trees.J48;

import javax.annotation.processing.FilerException;
import javax.xml.soap.Detail;
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
    protected String dataset_metadata_path;
    protected Individual overall;
    protected Individual last;
    protected int n_sample;
    protected int n_fold;
    protected int n_individuals;
    protected int curGen;
    protected ArrayList<Double> minFitness;
    protected ArrayList<Double> maxFitness;
    protected ArrayList<Double> medianFitness;



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

    public PBILLogger(String dataset_metadata_path, int n_individuals, int n_sample, int n_fold) {
        this.dataset_metadata_path = dataset_metadata_path;
        this.dataset_overall_path = String.format(
                "%s%soverall%stest_sample-%02d_fold_%02d.csv",
                dataset_metadata_path, File.separator, File.separator, n_sample, n_fold
        );
        this.n_sample = n_sample;
        this.n_fold = n_fold;
        this.n_individuals = n_individuals;

        this.minFitness = new ArrayList<>();;
        this.medianFitness = new ArrayList<>();;
        this.maxFitness = new ArrayList<>();;

        this.curGen = 0;
    }

    public void logPopulation(double min, double median, double max, Individual overall, Individual last) {
        this.overall = overall;
        this.last = last;
        this.minFitness.add(min);
        this.medianFitness.add(median);
        this.maxFitness.add(max);

        this.curGen += 1;
    }

    public void logProbabilities(DependencyNetwork dn) {

    }

    private String getEvaluationLineForClassifier(
            String name, Evaluation evaluation) throws InvocationTargetException, IllegalAccessException {
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

                    line += ", \"np.array([";
                    for(int k = 0; k < doublyOverall.length; k++) {
                        line += doublyOverall[k] + ",";
                    }
                    line = line.substring(0, line.lastIndexOf(",")) + "], dtype=np.float64)\"";
                } catch(ClassCastException e) {
                    double[][] doublyOverall = ((double[][])res);

                    line += ", \"np.array([[";
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

    private void evaluationsToFile(HashMap<String, Individual> individuals, Instances train_data, Instances test_data) throws Exception {
        BufferedWriter bw = new BufferedWriter(new FileWriter(this.dataset_overall_path));

        HashMap<String, Evaluation> clfEvaluations = new HashMap<>(14);

        for(String indName : individuals.keySet()) {
            Evaluation evaluation = new Evaluation(train_data);
            HashMap<String, AbstractClassifier> overallClassifiers = individuals.get(indName).getClassifiers();
            evaluation.evaluateModel(individuals.get(indName), test_data);

            clfEvaluations.put(indName, evaluation);

            for(String key: overallClassifiers.keySet()) {
                if(overallClassifiers.get(key) != null) {
                    evaluation.evaluateModel(overallClassifiers.get(key), test_data);
                }
                clfEvaluations.put(indName + "-" + key, evaluation);
            }
        }

        // writes header
        String header = "algorithm";
        for(String methodName : metricsToCollect) {
            header += "," + methodName;
        }
        bw.write(header + "," + "unweightedAreaUnderRoc" + "\n");

        for(String clf: clfEvaluations.keySet()) {
            bw.write(this.getEvaluationLineForClassifier(clf, clfEvaluations.get(clf)));
        }
        bw.close();
    }

    public void toFile(HashMap<String, Individual> individuals, Instances train_data, Instances test_data) throws Exception {
        String thisRunFolder = String.format(
                "%s%ssample_%02d_fold_%02d",
                this.dataset_metadata_path, File.separator, this.n_sample, this.n_fold
        );

        evaluationsToFile(individuals, train_data, test_data);
        individualsCharacteristicsToFile(thisRunFolder, individuals);
        individualsClassifiersToFile(thisRunFolder, individuals);
    }

    private void individualsClassifiersToFile(String thisRunFolder, HashMap<String, Individual> individuals) throws Exception {
        for(String indName : individuals.keySet()) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    thisRunFolder + File.separator + indName  + "_classifiers.csv"
            ));

            // TODO optimize this. each classifier must be better written
            HashMap<String, AbstractClassifier> classifiers = individuals.get(indName).getClassifiers();
            for(String clfName : classifiers.keySet()) {
                AbstractClassifier clf = classifiers.get(clfName);

                String clfString = PBILLogger.formatClassifierString(clf);
                bw.write(clfName + "\n");
                bw.write(clfString + "\n\n\n");
            }
            bw.close();
        }
    }

    private String getCharacteristicsLineForIndividual(Object[] order, String indName, HashMap<String, String> characteristics) {
        String line = indName;
        for(Object ch : order) {
            line += "," + characteristics.get(ch);
        }
        return line + "\n";
    }

    private void individualsCharacteristicsToFile(String thisRunFolder, HashMap<String, Individual> individuals) throws IOException {
        PBILLogger.createFolder(thisRunFolder);

        BufferedWriter bw = new BufferedWriter(new FileWriter(thisRunFolder + File.separator + "characteristics.md"));

        Object[] characteristicsNames = null;
        for(String indName : individuals.keySet()) {
            HashMap<String, String> characteristics = individuals.get(indName).getCharacteristics();
            if(characteristicsNames == null) {
                characteristicsNames = characteristics.keySet().toArray();
                String header = "individual_name";
                for(int i = 0; i < characteristicsNames.length; i++) {
                    header += "," + characteristicsNames[i];
                }
                bw.write(header + "\n");
            }
            bw.write(this.getCharacteristicsLineForIndividual(characteristicsNames, indName, characteristics));
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
                this.maxFitness.get(this.curGen - 1),
                this.medianFitness.get(this.curGen - 1)
        ));
    }

    public static String formatClassifierString(AbstractClassifier clf) throws Exception {
        return (String)PBILLogger.class.getMethod("format" + clf.getClass().getSimpleName() + "String", AbstractClassifier.class).invoke(PBILLogger.class, clf);
    }

    public static String formatJ48String(AbstractClassifier clf) throws Exception {
        try {
            String header = "# J48 Decision Tree";
            //        String rawBody = ;

//            body = '\n\n'.join(map(lambda x: x.strip(), txt.split('------------------')[-1].split('Number of Leaves')[0].strip().replace('|   ', '  * ').split('\n')))

//            return '%s\n\n%s' % (header, body)
            return clf.toString();
        } catch(Exception e) {
            return clf.toString();
        }
    }
    public static String formatSimpleCartString(AbstractClassifier clf) throws Exception  {
        return clf.toString();
    }
    public static String formatJRipString(AbstractClassifier clf) throws Exception {
        return clf.toString();
    }
    public static String formatPARTString(AbstractClassifier clf) throws Exception {
        return clf.toString();
    }
    public static String formatDecisionTableString(AbstractClassifier clf) throws Exception {
        Boolean usesIbk = (Boolean) DecisionTable.class.getMethod("getUseIBk").invoke(clf);

        String defaultString = "Non matches covered by " + (usesIbk? "IB1" : "Majority class");
        String[] lines = clf.toString().toLowerCase().replaceAll("\'", "").split("rules:")[1].split("\n");

        // TODO do not join null values!
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

//        fmt = ['---' for i in range(len(df.columns))]
//        df_fmt = pd.DataFrame([fmt], columns=df.columns)
//        df_formatted = pd.concat([df_fmt, df])
//        table_str = df_formatted.to_csv(sep="|", index=False)
        String table_str  = String.join("\n", sanitized_lines);

        String r_str = String.format("# Decision Table\n\n%s\n\n%s", defaultString, table_str);
        return r_str;
    }

}
