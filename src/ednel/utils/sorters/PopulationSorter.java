package ednel.utils.sorters;

import ednel.eda.individual.Fitness;
import ednel.eda.individual.Individual;

import java.util.ArrayList;
import java.util.HashMap;

public class PopulationSorter {
    /**
     * Checks whether a dominates b (i.e. a is a better solution, in all criteria, than b).
     *
     * @param a first solution
     * @param b second solution
     * @return -1 if b dominates a, +1 if a dominates b, and 0 if there is no dominance
     */
    private static int a_dominates_b(Fitness a, Fitness b) {
        boolean a_dominates = ((a.getLearnQuality() >= b.getLearnQuality()) && (a.getSize() <= b.getSize())) &&
                ((a.getLearnQuality() > b.getLearnQuality()) || (a.getSize() < b.getSize()));
        boolean b_dominates = ((b.getLearnQuality() >= a.getLearnQuality()) && (b.getSize() <= a.getSize())) &&
                ((b.getLearnQuality() > a.getLearnQuality()) || (b.getSize() < a.getSize()));

        if(a_dominates) {
            if(b_dominates) {
                return 0;
            } else {
                return 1;
            }
        } else if(b_dominates) {
            return -1;
        } else {
            return 0;
        }
    }

    public static Integer[] paretoArgsort(Individual[] population) {
        HashMap<Integer, ArrayList<Integer>> dominates = new HashMap<>();
        HashMap<Integer, Integer> dominated = new HashMap<>();

        ArrayList<Integer> cur_front = new ArrayList<>();

        Integer[] sortedIndices = new Integer[population.length];
        int counter = 0;

        ArrayList<ArrayList<Integer>> fronts = new ArrayList<>();

        for(int i = 0; i < population.length; i++) {
            dominated.put(i, 0);
            dominates.put(i, new ArrayList<>());
        }

        for(int i = 0; i < population.length; i++) {
            for(int j = i + 1; j < population.length; j++) {
                int res = a_dominates_b(population[i].getFitness(), population[j].getFitness());
                if(res == 1) {
                    dominated.put(j, dominated.getOrDefault(j, 0) + 1);  // signals that j is dominated by one solution

                    ArrayList<Integer> thisDominates = dominates.getOrDefault(i, new ArrayList<Integer>());
                    thisDominates.add(j);
                    dominates.put(i, thisDominates);  // add j to the list of dominated solutions by i
                } else if(res == -1) {
                    dominated.put(i, dominated.getOrDefault(i, 0) + 1);  // signals that i is dominated by one solution

                    ArrayList<Integer> thisDominates = dominates.getOrDefault(j, new ArrayList<Integer>());
                    thisDominates.add(i);
                    dominates.put(j, thisDominates);  // add i to the list of dominated solutions by j
                }
            }
            if(dominated.get(i) == 0) {
                cur_front.add(i);
            }
        }

        while(cur_front.size() != 0) {
            ArrayList<Integer> some_set = new ArrayList<>();

            for(Integer master : cur_front) {
                for(Integer slave : dominates.get(master)) {
                    dominated.put(slave, dominated.get(slave) - 1);
                    if(dominated.get(slave) == 0) {
                        some_set.add(slave);
                    }
                }
            }
            Double[] cur_front_double = new Double[cur_front.size()];
            for(int i = 0; i < cur_front.size(); i++) {
                cur_front_double[i] = population[cur_front.get(i)].getFitness().getLearnQuality();
            }
            Integer[] local_indices = Argsorter.decrescent_argsort(cur_front_double);

            for(Integer item : local_indices) {
                sortedIndices[counter] = cur_front.get(item);
                counter += 1;
            }

            fronts.add((ArrayList<Integer>)cur_front.clone());
            cur_front = some_set;
        }
        return sortedIndices;
    }

    public static Integer[] simpleArgsort(Individual[] population) {
        Double[] fitness_values = new Double[population.length];
        for(int i = 0; i < population.length; i++) {
            fitness_values[i] = population[i].getFitness().getLearnQuality();
        }
        return Argsorter.decrescent_argsort(fitness_values);
    }

    public static Integer[] lexicographicArgsort(Individual[] population) {
        return Argsorter.decrescent_argsort(population);
    }
}
