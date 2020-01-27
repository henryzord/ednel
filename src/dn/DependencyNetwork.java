package dn;

import javafx.scene.transform.MatrixType;
import org.apache.commons.math3.random.MersenneTwister;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.stream.Stream;

public class DependencyNetwork {
    private Hashtable<String, Variable> variables;
    private MersenneTwister mt;

    public DependencyNetwork(MersenneTwister mt, String variables_path) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();

        this.mt = mt;

        variables = new Hashtable<>();

        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables_names  = Files.list(new File(algorithms[i].toString()).toPath()).toArray();
            for(int j = 0; j < variables_names.length; j++) {
                BufferedReader csvReader = new BufferedReader(new FileReader(variables_names[j].toString()));

                String row = csvReader.readLine();

                String[] header = row.split(",(?![^(]*\\))");

                String variableName = header[header.length - 2];
                int n_variables_table = 1;
                String[] parents = null;

                Hashtable<String, Hashtable<String, ArrayList<Integer>>> table = new Hashtable<>(header.length);
                table.put(variableName, new Hashtable<String, ArrayList<Integer>>());

                if(header.length > 2) {
                    parents = new String [header.length - 2];
                    n_variables_table = n_variables_table + parents.length;
                    for(int k = 0; k < header.length - 2; k++) {
                        parents[k] = header[k];
                        table.put(parents[k], new Hashtable<String, ArrayList<Integer>>((int)Math.pow(2, n_variables_table)));
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
                variables.put(variableName, new Variable(variableName, parents, table, probabilities, mt));
            }
        }
    }

    public void gibbsSample() {
        Enumeration<String> variable_names = this.variables.keys();
        int z = 0;
    }

    public static void main(String[] args) throws Exception {
        String variables_path = "/home/henry/Projects/ednel/resources/distributions";

        MersenneTwister mt = new MersenneTwister();

        DependencyNetwork dn = new DependencyNetwork(mt, variables_path);
        dn.gibbsSample();
    }
}
