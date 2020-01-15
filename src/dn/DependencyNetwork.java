package dn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Stream;

public class DependencyNetwork {
    public DependencyNetwork(String variables_path) throws Exception {
        Object[] algorithms = Files.list(new File(variables_path).toPath()).toArray();
        for(int i = 0; i < algorithms.length; i++) {
            Object[] variables = Files.list(new File(((Path)algorithms[i]).toString()).toPath()).toArray();
            for(int j = 0; j < variables.length; j++) {
                BufferedReader csvReader = new BufferedReader(new FileReader(((Path)variables[j]).toString()));
                String row;
                int n_columns = -1;
                ArrayList<String> data = new ArrayList<>();
                while ((row = csvReader.readLine()) != null) {
                    String[] this_data = row.split(",");
                    n_columns = this_data.length;
                    data.addAll(Arrays.asList(this_data));
                }
                csvReader.close();
                ConditionalTable ct = new ConditionalTable(data, n_columns);
            }

            int z = 0;
        }
    }

    public static void main(String[] args) throws Exception {
        String variables_path = "/home/henry/Projects/WekaCustom/distributions";

        DependencyNetwork dn = new DependencyNetwork(variables_path);
    }
}
