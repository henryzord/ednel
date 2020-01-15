package dn;

// import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GibbsSampler {

    public GibbsSampler(int[][] marginal) {

    }

    // private int[][][] marginalToConditionals(int[][] marginal) {
    //     int[][][] conditionals = new int[marginal.length][][];
    //
    //     for(int i = 0; i < marginal.length; i++) {
    //
    //     }
    //
    //     return null;
    //
    //     // indices = marginal.index  # type: pd.MultiIndex
    //     // variables = np.array(indices.names)
    //     //
    //     // conditionals = dict()
    //     //
    //     // for variable in variables:
    //     //     # creates matrix
    //     //     neighbors = variables[adjacency_matrix[variable].nonzero()].tolist()
    //     //     n_columns = len(neighbors) + 1
    //     //
    //     //     conditionals[variable] = pd.DataFrame(
    //     //         index=pd.MultiIndex.from_product(iterables=[[0, 1] for x in range(n_columns)], names=neighbors + [variable]),
    //     //         columns=['p'],
    //     //         dtype=np.float32
    //     //     )
    //     //
    //     //     # writing values
    //     //
    //     //     for combs in it.product(*[[0, 1] for x in neighbors]):
    //     //         index = {}
    //     //         for some_var in variables:
    //     //             if (some_var == variable) or (some_var not in neighbors):
    //     //                 index[some_var] = slice(None)
    //     //             else:
    //     //                 index[some_var] = combs[neighbors.index(some_var)]
    //     //
    //     //         n_index = tuple([index[v] for v in marginal.index.names])
    //     //         submatrix = marginal.loc(axis=0)[n_index]
    //     //         # submatrix = submatrix / submatrix.sum()  # normalizes
    //     //
    //     //         # puts values in conditional matrix
    //     //         index[variable] = slice(None)
    //     //         _sum = submatrix.loc(axis=0)[tuple([index[y] for y in variables])].sum()
    //     //
    //     //         for c in [0, 1]:
    //     //             index[variable] = c
    //     //
    //     //             conditionals[variable].loc(axis=0)[tuple([index[y] for y in neighbors] + [index[variable]])] = \
    //     //                 submatrix.loc(axis=0)[tuple([index[y] for y in variables])].sum() / _sum
    //     //
    //     // return conditionals
    // }
    //
    // // public static void readCsv(String path) throws Exception {
    // //     List<List<String>> records = new ArrayList<List<String>>();
    // //     try (CSVReader csvReader = new CSVReader(new FileReader("book.csv"));) {
    // //         String[] values = null;
    // //         while ((values = csvReader.readNext()) != null) {
    //             records.add(Arrays.asList(values));
    //         }
    //     }
    // }

    public static void main(String[] args) throws Exception {
        String cfValue = "/home/henry/Projects/WekaCustom/distributions/J48/J48_confidenceFactorValue.csv";
        // GibbsSampler.readCsv(cfValue);
    }
}
