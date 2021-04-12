import argparse
import json
import os
import sys

import javabridge
from mPBIL.pbil.model import PBIL
from weka.classifiers import Evaluation
from weka.core import jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.core.dataset import Instances
from sklearn.metrics import roc_auc_score
import numpy as np
from collections import Counter
from copy import deepcopy
from datetime import datetime as dt


def get_pbil_combinations():
    combinations = []

    learning_rate_values = [0.13, 0.26, 0.52]
    selection_share_values = [0.3, 0.5]
    n_individuals = 10  # TODO change from 10 to 100 individuals!
    n_generations = 2  # TODO change from 10 to 100 generations!

     # TODO change from 10 to 100 individuals!
     # TODO change from 10 to 100 generations!

    for learning_rate in learning_rate_values:
        for selection_share in selection_share_values:
            comb = {
                "learning_rate": learning_rate,
                "selection_share": selection_share,
                "n_individuals": n_individuals,
                "n_generations": n_generations
            }
            combinations += [comb]

    return combinations


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


def get_params(args: argparse.Namespace) -> dict:
    return deepcopy(args.__dict__)


def main(args):
    e = None

    try:
        experiment_folder = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

        os.mkdir(os.path.join(args.metadata_path, experiment_folder))
        os.mkdir(os.path.join(args.metadata_path, experiment_folder, args.dataset_name))
        os.mkdir(os.path.join(args.metadata_path, experiment_folder, args.dataset_name, 'overall'))

        with open(os.path.join(args.metadata_path, experiment_folder, 'parameters.json'), 'w') as write_file:
            dict_params = get_params(some_args)
            json.dump(dict_params, write_file, indent=2)

        jvm.start(max_heap_size=args.heap_size)  # using 4GB of heap size for JVM

        n_external_folds = 10  # TODO do not change this
        n_internal_folds = args.n_internal_folds

        for n_external_fold in range(1, n_external_folds + 1):
            seed = Random(1)

            external_train_data = read_dataset(
                os.path.join(
                    args.datasets_path,
                    args.dataset_name,
                    '%s-10-%dtra.arff' % (args.dataset_name, n_external_fold)
                )
            )  # type: Instances

            external_test_data = read_dataset(
                os.path.join(
                    args.datasets_path,
                    args.dataset_name,
                    '%s-10-%dtst.arff' % (args.dataset_name, n_external_fold)
                )
            )  # type: Instances

            external_train_data.stratify(n_internal_folds)

            class_unique_values = np.array(external_train_data.attribute(external_train_data.class_index).values)

            combinations = get_pbil_combinations()  # type: list

            overall_aucs = []  # type: list
            last_aucs = []  # type: list

            for comb in combinations:
                internal_actual_classes = []
                overall_preds = []
                last_preds = []

                for n_internal_fold in range(n_internal_folds):
                    internal_train_data = external_train_data.train_cv(n_internal_folds, n_internal_fold, seed)
                    internal_test_data = external_train_data.test_cv(n_internal_folds, n_internal_fold)

                    internal_actual_classes.extend(list(internal_test_data.values(internal_test_data.class_index)))

                    pbil = PBIL(
                        resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
                        train_data=internal_train_data,
                        lr=comb['learning_rate'], selection_share=comb['selection_share'],
                        n_generations=comb['n_generations'], n_individuals=comb['n_individuals']
                    )

                    overall, last = pbil.run(1)

                    overall_preds.extend(list(map(list, overall.predict_proba(internal_test_data))))
                    last_preds.extend(list(map(list, last.predict_proba(internal_test_data))))

                internal_actual_classes = np.array(internal_actual_classes, dtype=np.int)
                overall_preds = np.array(overall_preds)
                last_preds = np.array(last_preds)

                overall_auc = 0.
                last_auc = 0.
                for i, c in enumerate(class_unique_values):
                    actual_binary_class = (internal_actual_classes == i).astype(np.int)
                    overall_auc += roc_auc_score(y_true=actual_binary_class, y_score=overall_preds[:, i])
                    last_auc += roc_auc_score(y_true=actual_binary_class, y_score=last_preds[:, i])

                overall_aucs += [overall_auc / len(class_unique_values)]
                last_aucs += [last_auc / len(class_unique_values)]

            best_overall = np.argmax(overall_aucs)  # type: int
            best_last = np.argmax(last_aucs)  # type: int

            uses_overall = overall_aucs[best_overall] > last_aucs[best_last]
            best_index = best_overall if uses_overall else best_last  # type: int

            pbil = PBIL(
                resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
                train_data=external_train_data,
                lr=combinations[best_index]['learning_rate'], selection_share=combinations[best_index]['selection_share'],
                n_generations=combinations[best_index]['n_generations'], n_individuals=combinations[best_index]['n_individuals']
            )

            overall, last = pbil.run(1)

            clf = overall if uses_overall else last
            external_preds = list(map(list, clf.predict_proba(external_test_data)))
            external_actual_classes = list(external_test_data.values(external_test_data.class_index).astype(np.int))

            with open(
                    os.path.join(args.metadata_path, experiment_folder, args.dataset_name, 'test_sample-01_fold-%02d_parameters.json' % n_external_fold),
                    'w'
            ) as write_file:
                dict_best_params = deepcopy(combinations[best_index])
                dict_best_params['individual'] = 'overall' if uses_overall else 'last'
                for k in dict_best_params.keys():
                    dict_best_params[k] = str(dict_best_params[k])

                json.dump(dict_best_params, write_file, indent=2)

            with open(
                os.path.join(args.metadata_path, experiment_folder, args.dataset_name, 'overall', 'test_sample-01_fold-%02d_overall.preds' % n_external_fold)
            , 'w') as write_file:
                write_file.write('classValue;Individual\n')
                for i in range(len(external_actual_classes)):
                    write_file.write('%r;%s\n' % (external_actual_classes[i], ','.join(map(str, external_preds[i]))))

    except Exception as some:
        e = some
    finally:
        jvm.stop()

    if e is not None:
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs a nested cross validation for PBIL.'
    )

    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--dataset-name', action='store', required=True,
        help='Name of dataset to run nested cross validation'
    )

    parser.add_argument(
        '--n-internal-folds', action='store', required=True,
        help='Number of folds to use to perform an internal cross-validation for each combination of hyper-parameters', type=int
    )

    some_args = parser.parse_args()

    main(args=some_args)



