package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import eda.BaselineIndividual;
import org.apache.commons.math3.random.MersenneTwister;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.*;

public class DependencyNetwork {
    private HashMap<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private ArrayList<String> variable_names;
    private ArrayList<String> sampling_order;

    public DependencyNetwork(MersenneTwister mt, String variables_path, String sampling_order_path) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.mt = mt;
        this.variables = new HashMap<>();
        this.variable_names = new ArrayList<>((int)Math.pow(algorithms.length, 2));

        this.sampling_order = new ArrayList<>();

        // reads sampling order
        String row;
        BufferedReader csvReader = new BufferedReader(new FileReader(sampling_order_path));
        while ((row = csvReader.readLine()) != null) {
            sampling_order.add(row);
        }

        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables_names  = Files.list(new File(algorithms[i].toString()).toPath()).toArray();
            for(int j = 0; j < variables_names.length; j++) {
                csvReader = new BufferedReader(new FileReader(variables_names[j].toString()));

                row = csvReader.readLine();

                String[] header = row.split(",(?![^(]*\\))");

                HashMap<String, Boolean> isContinuous = new HashMap<>();

                String variableName = header[header.length - 2];
                isContinuous.put(variableName, false);
                this.variable_names.add(variableName);
                int n_variables_table = 1;
                String[] parentNames = new String [0];

                HashMap<String, HashMap<String, ArrayList<Integer>>> table = new HashMap<>(header.length);
                table.put(variableName, new HashMap<String, ArrayList<Integer>>());

                if(header.length > 2) {
                    parentNames = new String [header.length - 2];
                    n_variables_table = n_variables_table + parentNames.length;
                    for(int k = 0; k < header.length - 2; k++) {
                        parentNames[k] = header[k];
                        isContinuous.put(parentNames[k], false);
                        table.put(parentNames[k], new HashMap<String, ArrayList<Integer>>((int)Math.pow(2, n_variables_table)));
                    }
                }

                ArrayList<Float> probabilities = new ArrayList<>((int)Math.pow(2, n_variables_table));

                int index = 0;

                while ((row = csvReader.readLine()) != null) {
                    String[] this_data = row.split(",(?![^(]*\\))");
                    probabilities.add(Float.valueOf(this_data[this_data.length - 1]));
                    for(int k = 0; k < this_data.length - 1; k++) {
                        isContinuous.replace(
                                header[k],
                                isContinuous.get(header[k]) ||
                                        (this_data[k].contains("loc") && this_data[k].contains("scale"))
                        );

                        if(!table.get(header[k]).containsKey(this_data[k])) {  // if this variable does not have this value
                            table.get(header[k]).put(this_data[k], new ArrayList<>());
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();

                if(isContinuous.get(variableName)) {
                    variables.put(variableName, new ContinuousVariable(variableName, parentNames, table, probabilities, mt));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parentNames, table, probabilities, mt));
                }
            }
        }
    }

    public ArrayList<HashMap> gibbsSample(HashMap<String, String> lastStart, int thinning_factor, int sampleSize) {
        ArrayList<HashMap> sampled = new ArrayList<>(sampleSize);

        HashMap<String, String> current = new HashMap<>(lastStart.size());
        // TODO use laplace correction
        for(int i = 0; i < sampleSize * thinning_factor; i++) {
            System.out.println("at sample " + i);  // TODO remove
            for(String variable : this.sampling_order) {
                System.out.println(variable);  // TODO remove

                AbstractVariable curVariable = this.variables.get(variable);
                String[] parentsNames = curVariable.getParents();
                String[] parentValues = new String [parentsNames.length];

                for(int k = 0; k < parentsNames.length; k++) {
                    parentValues[k] = lastStart.get(parentsNames[k]);
                }

                current.put(
                        variable,
                        this.variables.get(variable).conditionalSampling(parentsNames, parentValues)
                );
            }
            if((i % thinning_factor) == 0) {
                sampled.add(current);
            }
            lastStart = current;
        }
        return sampled;
    }

    public static void main(String[] args) throws Exception {
        Instances train_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff")));
        train_data.setClassIndex(train_data.numAttributes() - 1);

        String variables_path = "/home/henry/Projects/ednel/resources/distributions";
        String sampling_order_path = "/home/henry/Projects/ednel/resources/sampling_order.csv";

        MersenneTwister mt = new MersenneTwister();

        DependencyNetwork dn = new DependencyNetwork(mt, variables_path, sampling_order_path);
        BaselineIndividual bi = new BaselineIndividual(train_data);
        HashMap<String, String> startPoint = bi.getCharacteristics();
        dn.gibbsSample(startPoint,10, 50);
    }
}
