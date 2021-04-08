import argparse
import json
import os
import sys

import javabridge
from mPBIL.pbil.model import PBIL
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances

# private static ArrayList<HashMap<String, Object>> getPBILCombinations(String script_path) {
#         ArrayList<HashMap<String, Object>> combinations = new ArrayList<>();
#
#         float[] learning_rate_values = {0.13f, 0.26f, 0.52f};
#         float[] selection_share_values = {0.3f, 0.5f};
#         int n_individuals = 10;  // TODO change from 10 to 100 individuals!
#         int n_generations = 2; // TODO change from 10 to 100 generations!
#
#         System.out.println("TODO change from 10 to 100 individuals!");
#         System.out.println("TODO change from 10 to 100 generations!");
#
#         String heap_size = "4g";
#         int n_jobs = 1;
#         int n_samples = 1;
#
#         for(float learning_rate : learning_rate_values) {
#             for(float selection_share : selection_share_values) {
#                 HashMap<String, Object> comb = new HashMap<>();
#
#                 comb.put("learning_rate", learning_rate);
#                 comb.put("selection_share", selection_share);
#                 comb.put("n_individuals", n_individuals);
#                 comb.put("n_generations", n_generations);
#                 comb.put("heap_size", heap_size);
# //                comb.put("n_jobs", n_jobs);
# //                comb.put("n_samples", n_samples);
#                 comb.put("script_path", script_path);
#
#                 combinations.add(comb);
#             }
#         }
#
#         return combinations;
#     }


def read_dataset(path: str) -> Instances:
    loader = Loader("weka.core.converters.ArffLoader")  # type: weka.core.converters.Loader

    data = loader.load_file(path)
    data.class_is_last()

    filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
    javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
    javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', data.jobject)
    jtrain_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        data.jobject, filter_obj
    )
    data = Instances(jtrain_data)
    return data


def main(args):
    e = None

    try:
        jvm.start(max_heap_size=args.heap_size)  # using 4GB of heap size for JVM

        train_data = read_dataset(args.train_data)
        test_data = read_dataset(args.test_data)

        pbil = PBIL(
            resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
            train_data=train_data,
            lr=args.learning_rate, selection_share=args.selection_share,
            n_generations=args.n_generations, n_individuals=args.n_individuals
        )

        actual_class = []
        for i in range(test_data.num_instances):
            actual_class += [test_data.get_instance(i).values[test_data.class_index]]

        overall, last = pbil.run(1)

        to_report = {'overall': overall, 'last': last}
        dict_res = dict(classValue=actual_class)
        for name, clf in to_report.items():
            preds = clf.predict_proba(test_data)

            dict_res[name] = list(map(list, map(list, preds)))

        res = json.dumps(dict_res)
        print('<json>')
        print(res)
        print('</json>')
    except Exception as some:
        e = some
    finally:
        jvm.stop()
        # os.unlink(train_name)  # TODO testing
        # os.unlink(test_name)  # TODO testing

    if e is not None:
        raise e

    exit(0)


if __name__ == '__main__':
    # TODO here
    # --learning-rate 0.13 --selection-share 0.3 --n-individuals 10 --n-generations 100 --heap-size 4g --n-jobs 1 --n-samples 1

    parser = argparse.ArgumentParser(
        description='Support script for calling PBIL from a Java context.'
    )

    # parser.add_argument(
    #     '--dataset-path', action='store', required=True,
    #     help='Must lead to a path that contains several subpaths, one for each dataset. Each subpath, in turn, must '
    #          'have the arff files.'
    # )
    #
    # parser.add_argument(
    #     '--dataset-name', action='store', required=True,
    #     help='Name of the dataset to be run.'
    # )

    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--learning-rate', action='store', required=True,
        help='Learning rate of PBIL', type=float
    )

    parser.add_argument(
        '--selection-share', action='store', required=True,
        help='Fraction of fittest population to use to update graphical model', type=float
    )

    parser.add_argument(
        '--n-generations', action='store', required=True,
        help='Maximum number of generations to run the algorithm', type=int
    )

    parser.add_argument(
        '--n-individuals', action='store', required=True,
        help='Number of individuals in the population', type=int
    )

    parser.add_argument(
        '--train-data', action='store', required=True,
        help='Path to temporary file with training data'
    )

    parser.add_argument(
        '--test-data', action='store', required=True,
        help='Path to temporary file with test data'
    )

    some_args = parser.parse_args()

    main(args=some_args)



