"""
Use this script to collapse metrics after running an experiment.

export java_home before using it:

export JAVA_HOME="/usr/lib/jvm/jdk1.8.0_221
"""

import argparse
import itertools as it
import os
from functools import reduce

import numpy as np
import pandas as pd


class EDAEvaluation(object):
    metrics = [
        ("avg_cost", np.mean),
        ("class_priors", np.sum),
        ("confusion_matrix", np.sum),
        ("correct", np.sum),
        ("error_rate", np.mean),
        ("incorrect", np.sum),
        ("kappa", np.mean),
        ("kb_information", np.mean),
        ("kb_mean_information", np.mean),
        ("kb_relative_information", np.mean),
        ("mean_absolute_error", np.mean),
        ("mean_prior_absolute_error", np.mean),
        # ("n_classes", np.mean),
        ("num_instances", np.sum),
        ("percent_correct", np.mean),
        ("percent_incorrect", np.mean),
        ("percent_unclassified", np.mean),
        ("relative_absolute_error", np.mean),
        ("root_mean_prior_squared_error", np.mean),
        ("root_mean_squared_error", np.mean),
        ("root_relative_squared_error", np.mean),
        ("sf_entropy_gain", np.mean),
        ("sf_mean_entropy_gain", np.mean),
        ("sf_mean_prior_entropy", np.mean),
        ("sf_mean_scheme_entropy", np.mean),
        ("sf_prior_entropy", np.mean),
        ("size_of_predicted_regions", np.mean),
        ("total_cost", np.sum),
        ("unclassified", np.sum),
        ("unweighted_area_under_roc", np.mean),
        ("unweighted_macro_f_measure", np.mean),
        ("unweighted_micro_f_measure", np.mean),
        ("weighted_area_under_prc", np.mean),
        ("weighted_area_under_roc", np.mean),
        ("weighted_f_measure", np.mean),
        ("weighted_false_negative_rate", np.mean),
        ("weighted_false_positive_rate", np.mean),
        ("weighted_matthews_correlation", np.mean),
        ("weighted_precision", np.mean),
        ("weighted_recall", np.mean),
        ("weighted_true_negative_rate", np.mean),
        ("weighted_true_positive_rate", np.mean),
    ]

    # converts names of metrics from Weka (from Java) to Weka (from Python wrapper)
    metrics_dict = {
        "avg_cost": "avgCost",
        "class_priors": "getClassPriors",
        "confusion_matrix": "confusionMatrix",
        "correct": "correct",
        "error_rate": "errorRate",
        "incorrect": "incorrect",
        "kappa": "kappa",
        "kb_information": "KBInformation",
        "kb_mean_information": "KBMeanInformation",
        "kb_relative_information": "KBRelativeInformation",
        "mean_absolute_error": "meanAbsoluteError",
        "mean_prior_absolute_error": "meanPriorAbsoluteError",
        # "n_classes": "n_classes",
        "num_instances": "numInstances",
        "percent_correct": "pctCorrect",
        "percent_incorrect": "pctIncorrect",
        "percent_unclassified": "pctUnclassified",
        "unclassified": "unclassified",
        "relative_absolute_error": "relativeAbsoluteError",
        "root_mean_prior_squared_error": "rootMeanPriorSquaredError",
        "root_mean_squared_error": "rootMeanSquaredError",
        "root_relative_squared_error": "rootRelativeSquaredError",
        "sf_entropy_gain": "SFEntropyGain",
        "sf_mean_entropy_gain": "SFMeanEntropyGain",
        "sf_mean_prior_entropy": "SFMeanPriorEntropy",
        "sf_mean_scheme_entropy": "SFMeanSchemeEntropy",
        "sf_prior_entropy": "SFPriorEntropy",
        "size_of_predicted_regions": "sizeOfPredictedRegions",
        "total_cost": "totalCost",
        "weighted_area_under_prc": "weightedAreaUnderPRC",
        "weighted_area_under_roc": "weightedAreaUnderROC",
        "unweighted_area_under_roc": "unweightedAreaUnderRoc",
        "weighted_f_measure": "weightedFMeasure",
        "weighted_false_negative_rate": "weightedFalseNegativeRate",
        "weighted_false_positive_rate": "weightedFalsePositiveRate",
        "weighted_matthews_correlation": "weightedMatthewsCorrelation",
        "weighted_precision": "weightedPrecision",
        "weighted_recall": "weightedRecall",
        "weighted_true_negative_rate": "weightedTrueNegativeRate",
        "weighted_true_positive_rate": "weightedTruePositiveRate",
        "unweighted_macro_f_measure": "unweightedMacroFmeasure",
        "unweighted_micro_f_measure": "unweightedMicroFmeasure"
    }


def __get_relation__(path):
    """
    Get experiment results as a dataframe.

    :param path: Path pointing to folder with experiment results as a series of csv files.
    :return: A dataframe with the relation of experiment results as a dataframe.
    """
    files = os.listdir(path)
    df = pd.DataFrame(list(map(lambda x: x[:-len('.txt')].split('_'), files)),
                      columns=['dataset', 'sample', 'fold']).dropna()
    return df


def check_missing_experiments(path, n_samples, n_folds):
    """
    Checks missing results files in the "overall" folder.

    Please point to a folder that contains the .csv results of the experiments.

    :param path: Path to folder with csv results.
    :param n_samples: Number of trial sampples in the experiment.
    :param n_folds: Number of folds in the cross-validation.
    :rtype: list
    :return: A list of tuples where the first item in each tuple is the trial, and the second the number of the fold,
        that are missing in the results folder pointed by path.
    """

    assert path is not None, ValueError('path must point to the folder with csv results.')

    folds = range(1, n_folds + 1)
    samples = range(1, n_samples + 1)

    files = os.listdir(path)

    missing = []
    combs = it.product(samples, folds)
    for sample, fold in combs:
        if 'test_sample-%02d_fold-%02d.csv' % (sample, fold) not in files:
            missing += [(sample, fold)]

    return missing


def single_experiment_process(this_path, n_samples, n_folds, write=True):
    relation = __get_relation__(this_path)

    missing = check_missing_experiments(
        path=this_path,
        n_samples=n_samples, n_folds=n_folds
    )
    if len(missing) > 0:
        raise ValueError('Could not collapse metrics for dataset. Some runs are missing.')

    samples_dicts = dict()

    samples = relation['sample'].unique()

    ens_names = None
    for sample in samples:
        this_sample_relation = relation.loc[relation['sample'] == sample]

        rels = list(map(
            lambda z: pd.read_csv('%s/%s.csv' % (this_path, '_'.join(z)), index_col=0),
            this_sample_relation.values
        ))
        ens_names = rels[0].index

        condensed = pd.DataFrame(
            index=ens_names,
            columns=pd.MultiIndex.from_product([list(zip(*EDAEvaluation.metrics))[0], ['mean', 'std']],
                                               names=['metric', 'statistics']),
            dtype=np.float64
        )
        for metric_name, metric_operation in EDAEvaluation.metrics:
            if (metric_name == 'confusion_matrix') or (metric_name == 'class_priors'):
                dict_ens = {}
                for ens_name in ens_names:
                    for rel in rels:
                        try:
                            dict_ens[ens_name] += eval(rel.loc[ens_name, EDAEvaluation.metrics_dict[metric_name]])
                        except KeyError:
                            dict_ens[ens_name] = eval(rel.loc[ens_name, EDAEvaluation.metrics_dict[metric_name]])

                condensed[(metric_name, 'mean')] = pd.Series(dict_ens)
                condensed[(metric_name, 'std')] = np.repeat(np.nan, len(dict_ens))

            else:
                for ens_name in ens_names:
                    values = [rel.loc[ens_name, EDAEvaluation.metrics_dict[metric_name]] for rel in rels]
                    condensed.loc[ens_name, (metric_name, 'mean')] = np.mean(values)
                    condensed.loc[ens_name, (metric_name, 'std')] = np.std(values)

        condensed['sample'] = np.repeat(sample, len(condensed))
        samples_dicts[sample] = condensed

    # adds mean of means
    summary = reduce(lambda x, y: x.append(y), samples_dicts.values())

    pre_agg = summary.drop('std', axis=1, level=1)
    pre_agg.columns = pre_agg.columns.droplevel(1)

    agg = pre_agg.groupby(level=0).agg([np.mean, np.std])

    summary.index = ['-'.join([x, y]) for x, y in zip(summary.index, summary['sample'])]
    del summary['sample']
    agg.index = ['-'.join([x, y]) for x, y in zip(agg.index, np.repeat('mean-of-means', len(agg)))]
    summary = summary.append(agg)

    if write:
        summary.to_csv(os.path.join(this_path, 'summary.csv'))

    # tides up for display
    to_return_index = list(map(lambda x: '-'.join(x), it.product(ens_names, ['mean-of-means'])))
    to_return_columns = pd.MultiIndex.from_product([['unweighted_area_under_roc', 'percent_correct'], ['mean', 'std']])
    to_return = summary.loc[to_return_index, to_return_columns]  # type: pd.DataFrame
    to_return.columns = pd.MultiIndex.from_product([['AUC', 'Accuracy'], ['mean', 'std']])
    to_return.index = [x.split('-')[0] for x in to_return.index]
    to_return.loc[:, (slice(None), 'mean')] = to_return.loc[:, (slice(None), 'mean')].applymap('{:,.4f}'.format)
    to_return.loc[:, (slice(None), 'std')] = to_return.loc[:, (slice(None), 'std')].applymap('{:,.2f}'.format)

    return to_return


def get_n_samples_n_folds(path):

    outer_n_samples = -1
    outer_n_folds = -1

    if os.path.isdir(path):
        if 'overall' in path:
            relation = __get_relation__(path)
            outer_n_samples = max(outer_n_samples, max([int(x.split('-')[-1]) for x in relation['sample'].unique()]))
            outer_n_folds = max(outer_n_folds, max([int(x.split('-')[-1]) for x in relation['fold'].unique()]))
        else:
            _files = os.listdir(path)
            if 'summary' in _files:
                _files.remove("summary")

            for _file in _files:
                n_samples, n_folds = get_n_samples_n_folds(os.path.join(path, _file))
                outer_n_samples = max(outer_n_samples, n_samples)
                outer_n_folds = max(outer_n_folds, n_folds)

    return outer_n_samples, outer_n_folds


def recursive_experiment_process(this_path, n_samples, n_folds, write=True):
    # TODO join everything in a single csv!

    if os.path.isdir(this_path):
        if 'overall' in this_path:
            single_experiment_process(this_path, n_samples, n_folds, write=write)
        else:
            _files = os.listdir(this_path)
            for _file in _files:
                recursive_experiment_process(
                    os.path.join(this_path, _file),
                    n_samples=n_samples,
                    n_folds=n_folds,
                    write=write
                )


def main(metadata_path, write=True):
    n_samples, n_folds = get_n_samples_n_folds(metadata_path)
    recursive_experiment_process(metadata_path, n_samples=n_samples, n_folds=n_folds, write=write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for collapsing metrics'
    )

    parser.add_argument(
        '--csv-path', action='store', required=True,
        help='A path where .csv files (one for each run) are stored. Works recursively (i.e. as long as .csv files are '
             'within the specified folder, it will work).'
    )

    args = parser.parse_args()
    main(metadata_path=args.csv_path)  

