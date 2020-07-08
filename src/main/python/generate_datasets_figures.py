import os

import javabridge
import numpy as np
import pandas as pd
from scipy.io import arff
from weka.core.converters import Loader
from weka.core.dataset import Instances
from collections import Counter
from matplotlib import pyplot as plt
import argparse


def __dataframe_preprocess__(dataset_path):
    value, metadata = path_to_arff(dataset_path)

    df = pd.DataFrame(value, columns=metadata._attributes)

    attributes = metadata._attributes
    for attr_name, attr_dict in attributes.items():
        if attr_dict.type_name in ('nominal', 'string'):
            df[attr_name] = df[attr_name].apply(lambda x: x.decode('utf-8'))

            df[attr_name] = df[attr_name].astype('category')
        elif attr_dict.type_name == 'date':
            raise TypeError('unsupported attribute type!')
        else:
            df[attr_name] = df[attr_name].astype(np.float32)

    return df


def path_to_dataframes(dataset_path: str, n_fold: int = None):
    """
    Reads dataframes from an .arff file, casts categorical attributes to categorical type of pandas.

    :param dataset_path:
    :type dataset_path: str
    :param n_fold:
    :type n_fold: int
    :rtype: pandas.DataFrame
    :return:
    """

    dataset_name = dataset_path.split('/')[-1]

    if n_fold is not None:
        train_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtra.arff' % n_fold]))
        test_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtst.arff' % n_fold]))

    train_df = __dataframe_preprocess__(train_path)
    test_df = __dataframe_preprocess__(test_path)

    return train_df, test_df


def path_to_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e., "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.loadarff(dataset_path)
    return af


def read_datasets(dataset_path, n_fold):
    """

    :param dataset_path:
    :param n_fold:
    :return: A tuple (train-data, test_data), where each object is an Instances object
    """

    dataset_name = dataset_path.split('/')[-1]

    train_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtra.arff' % n_fold]))
    test_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtst.arff' % n_fold]))

    loader = Loader("weka.core.converters.ArffLoader")
    train_data = loader.load_file(train_path)
    train_data.class_is_last()

    test_data = loader.load_file(test_path)
    test_data.class_is_last()

    filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
    javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
    javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', train_data.jobject)
    jtrain_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        train_data.jobject, filter_obj
    )
    jtest_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        test_data.jobject, filter_obj
    )

    train_data = Instances(jtrain_data)
    test_data = Instances(jtest_data)

    return train_data, test_data


def __get_numeric_count__(df):
    count = 0
    for column in df.columns:
        if str(df[column].dtype) != 'category':
            count += 1

    return count


def __get_nominal_count__(df):
    count = 0
    for column in df.columns:
        if str(df[column].dtype) == 'category':
            count += 1

    return count


def __get_missing_count__(joined_dataset):
    isnull_matrix = joined_dataset.isnull()
    isnull_array = isnull_matrix.apply(lambda x: np.any(x), axis=1).values.astype(np.int32)
    return round(np.sum(isnull_array)/len(joined_dataset), 2)


def __write_markdown__(df, markdown_path):
    header = "## KEEL Datasets\n\n" \
             "Most of the datasets were extracted from KEEL repository. Some of them present rows with missing " \
             "values (which is the data present at the column missing), All datasets are in .arff format and " \
             "separated in 20 files, a train and test set for each one of the iterations of a " \
             "10-fold stratified cross validation.\n\n"

    columns = df.columns

    with open(os.path.join(markdown_path, "DATASETS.md"), 'w') as _f:
        _f.write(header)

        _f.write('|' + ' | '.join(columns) + ' |\n')

        _f.write('|' + ' | '.join([' -----:' for c in columns]) + ' |\n')

        for i, row in df.iterrows():
            _f.write('|' + ' | '.join([str(row[c]) for c in columns]) + ' |\n')


def main(datasets_path, markdown_path, figures_path):
    datasets_metadata = []

    for dataset_name in os.listdir(datasets_path):
        print('on dataset', dataset_name)

        try:
            train_set, test_set = path_to_dataframes(
                os.path.join(datasets_path, dataset_name, dataset_name),
                1
            )

            joined_dataset = pd.concat([train_set, test_set], ignore_index=True)

            class_labels_count = Counter(list(joined_dataset[joined_dataset.columns[-1]]))

            datasets_metadata += [[
                dataset_name,
                joined_dataset.shape[0],
                joined_dataset.shape[1],
                __get_nominal_count__(joined_dataset),
                __get_numeric_count__(joined_dataset),
                len(class_labels_count),
                __get_missing_count__(joined_dataset),
                "![%s](figures/%s.png)" % (dataset_name, dataset_name)
            ]]

            counted = [(name, count) for name, count in class_labels_count.items()]
            counted = sorted(counted, key=lambda x: x[0])
            x_ticks = np.arange(len(counted))

            plt.clf()
            plt.bar(x_ticks, list(zip(*counted))[1])
            plt.axis('off')
            plt.savefig(
                os.path.join(figures_path, dataset_name) + ".png",
                format="png"
            )

        except:
            print('\tfailed')

    df_metadata = pd.DataFrame(
        datasets_metadata,
        columns=['name', 'instances', 'attributes', 'categorical', 'numeric', 'classes', 'missing', 'class distribution']
    )
    __write_markdown__(df_metadata, markdown_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating figures of distribution of instances among classes, '
                    'for all datasets used in an experiment. As well as updating the DATASETS.md file, '
                    'with description of all datasets.'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path to a folder of folders: each subfolder contains the subsets for each dataset.'
    )
    parser.add_argument(
        '--markdown-path', action='store', required=True,
        help='Path where a new DATASETS.md will be write. Will overwrite previous file, '
             'or create a new one if not present.'
    )
    parser.add_argument(
        '--figures-path', action='store', required=True,
        help='Path where .png files (one for each dataset) will be written. Will overwrite previous files if any.'
    )

    args = parser.parse_args()
    main(datasets_path=args.datasets_path, markdown_path=args.markdown_path, figures_path=args.figures_path)


