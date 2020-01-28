package dn;

import dn.variables.AbstractVariable;
import dn.variables.ContinuousVariable;
import dn.variables.DiscreteVariable;
import org.apache.commons.math3.random.MersenneTwister;

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
                String[] parentNames = null;

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
                        if(!table.get(header[k]).contains(this_data[k])) {
                            table.get(header[k]).put(this_data[k], new ArrayList<>((int)Math.pow(2, n_variables_table)));
                        }
                        table.get(header[k]).get(this_data[k]).add(index);
                    }
                    index++;
                }
                csvReader.close();
                
                boolean isContinuous = (table.get(variableName).contains("loc") && table.get(variableName).contains("scale"));
                boolean[] continuousParents;
                if(parentNames != null) {
                    continuousParents = new boolean [parentNames.length];
                    for(int k = 0; k < parentNames.length; k++) {
                        continuousParents[k] = (table.get(parentNames[k]).contains("loc") && table.get(parentNames[k]).contains("scale"));
                        if(continuousParents[k] && isContinuous) {
                            throw new Exception("currently this is an unsupported case");
                        }
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

    public void gibbsSample(String lastStart) {
        AbstractVariable startVariable = this.variables.get(lastStart);
        startVariable.unconditionalSampling(1);
        int z = 0;

        // TODO use laplace correction
        // TODO research burning time
        // TODO research latency
    }

    public static void main(String[] args) throws Exception {
        String variables_path = "/home/henry/Projects/ednel/resources/distributions";

        MersenneTwister mt = new MersenneTwister();

        DependencyNetwork dn = new DependencyNetwork(mt, variables_path);
        dn.gibbsSample("J48_pruning");
    }
}
