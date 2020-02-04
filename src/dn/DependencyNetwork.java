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
    private Hashtable<String, AbstractVariable> variables;
    private MersenneTwister mt;
    private ArrayList<String> variable_names;

    public DependencyNetwork(MersenneTwister mt, String variables_path) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.mt = mt;
        this.variables = new Hashtable<>();
        this.variable_names = new ArrayList<>((int)Math.pow(algorithms.length, 2));

        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables_names  = Files.list(new File(algorithms[i].toString()).toPath()).toArray();
            for(int j = 0; j < variables_names.length; j++) {
                BufferedReader csvReader = new BufferedReader(new FileReader(variables_names[j].toString()));

                String row = csvReader.readLine();

                String[] header = row.split(",(?![^(]*\\))");

                String variableName = header[header.length - 2];
                this.variable_names.add(variableName);
                int n_variables_table = 1;
                String[] parentNames = new String [0];

                Hashtable<String, Hashtable<String, ArrayList<Integer>>> table = new Hashtable<>(header.length);
                table.put(variableName, new Hashtable<String, ArrayList<Integer>>());

                if(header.length > 2) {
                    parentNames = new String [header.length - 2];
                    n_variables_table = n_variables_table + parentNames.length;
                    for(int k = 0; k < header.length - 2; k++) {
                        parentNames[k] = header[k];
                        table.put(parentNames[k], new Hashtable<String, ArrayList<Integer>>((int)Math.pow(2, n_variables_table)));
                    }
                }

                ArrayList<Float> probabilities = new ArrayList<>((int)Math.pow(2, n_variables_table));

                int index = 0;
                while ((row = csvReader.readLine()) != null) {
                    String[] this_data = row.split(",(?![^(]*\\))");
                    probabilities.add(Float.valueOf(this_data[this_data.length - 1]));
                    for(int k = 0; k < this_data.length - 1; k++) {
                        if(!table.get(header[k]).containsKey(this_data[k])) {  // if this variable does not have this value
                            table.get(header[k]).put(this_data[k], new ArrayList<>());
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();
                
                boolean isContinuous = (table.get(variableName).contains("loc") && table.get(variableName).contains("scale"));
                boolean[] continuousParents;

                continuousParents = new boolean [parentNames.length];
                for(int k = 0; k < parentNames.length; k++) {
                    continuousParents[k] = (table.get(parentNames[k]).contains("loc") && table.get(parentNames[k]).contains("scale"));
                    if(continuousParents[k] && isContinuous) {
                        throw new Exception("currently this is an unsupported case");
                    }
                }


                if(isContinuous) {
                    variables.put(variableName, new ContinuousVariable(variableName, parentNames, table, probabilities, mt));
                } else {
                    variables.put(variableName, new DiscreteVariable(variableName, parentNames, table, probabilities, mt));
                }
            }
        }
    }

    public void gibbsSample(Hashtable<String, String> lastStart, int thinning_factor) {
        for(String variable : this.variable_names) {
            AbstractVariable curVariable = this.variables.get(variable);
            String[] parentsNames = curVariable.getParents();
            String[] parentValues = new String [parentsNames.length];

            for(int i = 0; i < parentsNames.length; i++) {
                parentValues[i] = lastStart.get(parentsNames[i]);
            }

            this.variables.get(variable).conditionalSampling(parentsNames, parentValues);
        }

        // String startVariable = (String)sset[mt.nextInt(lastStart.size())];
        // String initialValue = lastStart.get(startVariable);

        int z = 0;

        // TODO use laplace correction
        // TODO research thinning
    }

    public static void main(String[] args) throws Exception {
        Instances train_data = new Instances(new BufferedReader(new FileReader("/home/henry/Projects/eacomp/keel_datasets_10fcv/mammographic/mammographic-10-3tra.arff")));
        train_data.setClassIndex(train_data.numAttributes() - 1);

        String variables_path = "/home/henry/Projects/ednel/resources/distributions";

        MersenneTwister mt = new MersenneTwister();

        DependencyNetwork dn = new DependencyNetwork(mt, variables_path);
        BaselineIndividual bi = new BaselineIndividual(train_data);
        Hashtable<String, String> startPoint = bi.getCharacteristics();
        dn.gibbsSample(startPoint,10);
    }
}
