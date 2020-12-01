"""
After running an experiment, use this script to generate .csv files (one for each dataset, placed at <metadata_folder>/overall
for that dataset) with results for each of the metrics considered in EDAEvaluation.metrics

export java_home before using it:

export JAVA_HOME="/usr/lib/jvm/jdk1.8.0_221
"""

import argparse
import itertools as it
import json
import os
import sys
from functools import reduce

import numpy as np
import pandas as pd
import operator as op

from pandas.errors import EmptyDataError


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
        file_name = 'test_sample-%02d_fold-%02d.csv' % (sample, fold)

        try:
            if file_name not in files:
                raise FileNotFoundError('not found')
            df = pd.read_csv(os.path.join(path, file_name))
        except (FileNotFoundError, EmptyDataError) as e:
            missing += [(sample, fold)]

    return missing


def single_experiment_process(this_path, n_samples, n_folds, write=True):
    relation = __get_relation__(this_path)

    missing = check_missing_experiments(
        path=this_path,
        n_samples=n_samples, n_folds=n_folds
    )
    if len(missing) > 0:
        dataset_name = this_path.split(os.sep)[-2]
        experiment_name = this_path.split(os.sep)[-3]
        print('Dataset %s for experiment %s is not complete and hence will not be written to files' % (dataset_name, experiment_name), file=sys.stderr)
        return None

    samples_dicts = dict()

    samples = relation['sample'].unique()

    ens_names = None
    for sample in samples:
        this_sample_relation = relation.loc[relation['sample'] == sample]

        rels = list(map(
            lambda z: pd.read_csv('%s%s%s.csv' % (this_path, os.sep, '_'.join(z)), index_col=0),
            this_sample_relation.values
        ))
        ens_names = rels[0].index

        condensed = pd.DataFrame(
            index=ens_names,
            columns=pd.MultiIndex.from_product([list(zip(*EDAEvaluation.metrics))[0], ['mean', 'std']],
                                               names=['metric', 'statistics']),
            dtype=np.float64
        )

        column_labels = np.unique(rels[0].columns.get_level_values(0).tolist())
        n_header = 0
        n_values = 0
        for column_label in column_labels:
            n_header += int(column_label in list(EDAEvaluation.metrics_dict.keys()))
            n_values += int(column_label in list(EDAEvaluation.metrics_dict.values()))

        if n_header > n_values:
            needs_dict = False
        else:
            needs_dict = True

        for metric_name, metric_operation in EDAEvaluation.metrics:
            if (metric_name == 'confusion_matrix') or (metric_name == 'class_priors'):
                dict_ens = {}
                for ens_name in ens_names:
                    for rel in rels:
                        to_process = rel.loc[ens_name, EDAEvaluation.metrics_dict[metric_name] if needs_dict else metric_name]
                        is_nan = False
                        try:
                            is_nan = np.isnan(to_process)
                        except TypeError:  # not a nan value
                            pass
                        finally:
                            if is_nan:
                                dict_ens[ens_name] = np.nan
                            else:
                                try:
                                    dict_ens[ens_name] += eval(to_process)
                                except KeyError:
                                    dict_ens[ens_name] = eval(to_process)

                condensed[(metric_name, 'mean')] = pd.Series(dict_ens)
                condensed[(metric_name, 'std')] = np.repeat(np.nan, len(dict_ens))

            else:
                for ens_name in ens_names:
                    values = [rel.loc[ens_name, EDAEvaluation.metrics_dict[metric_name] if needs_dict else metric_name] for rel in rels]
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
        folders = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

        if 'overall' in folders:  # is within a single experiment folder
            relation = __get_relation__(os.path.join(path, 'overall'))
            if len(relation) > 0:
                outer_n_samples = max(outer_n_samples, max([int(x.split('-')[-1]) for x in relation['sample'].unique()]))
                outer_n_folds = max(outer_n_folds, max([int(x.split('-')[-1]) for x in relation['fold'].unique()]))
        else:  # all experiment folders must have the same number of samples and folds
            for folder in folders:
                n_samples, n_folds = get_n_samples_n_folds(os.path.join(path, folder))
                outer_n_samples = max(outer_n_samples, n_samples)
                outer_n_folds = max(outer_n_folds, n_folds)
                # if (outer_n_samples != -1) and (n_samples != outer_n_samples):
                #     raise Exception("experiments have different number of samples!")
                # else:
                #     outer_n_samples = n_samples
                #
                # if (outer_n_folds != -1) and (n_folds != outer_n_folds):
                #     raise Exception("experiments have different number of folds!")
                # else:
                #     outer_n_folds = n_folds
    else:
        raise Exception("experiment path does not point to a valid experiment directory!")

    return outer_n_samples, outer_n_folds


def recursive_experiment_process(this_path, n_samples, n_folds, dict_means, write=True):
    if os.path.isdir(this_path):
        folders = [x for x in os.listdir(this_path) if os.path.isdir(os.path.join(this_path, x))]

        if 'overall' in folders:
            experiment_name = this_path.split(os.sep)[-2]
            dataset_name = this_path.split(os.sep)[-1]

            single_experiment_process(os.path.join(this_path, 'overall'), n_samples, n_folds, write=write)

            single_experiment_folders = [x for x in folders if x != 'overall']
            for folder in single_experiment_folders:
                df = pd.read_csv(os.path.join(this_path, folder, 'loggerData.csv'))
                numeric_columns = [x for x in df if pd.api.types.is_numeric_dtype(df[x])]
                means = df[numeric_columns].mean(axis=0)
                stds = df[numeric_columns].std(axis=0)
                generations = len(df)

                if experiment_name not in dict_means:
                    dict_means[experiment_name] = dict()
                if dataset_name not in dict_means[experiment_name]:
                    dict_means[experiment_name][dataset_name] = dict()
                if any([x not in dict_means[experiment_name][dataset_name] for x in ['means', 'number_generations']]):
                    dict_means[experiment_name][dataset_name]['means'] = dict()
                    dict_means[experiment_name][dataset_name]['number_generations'] = dict()

                for data in means.index:
                    try:
                        dict_means[experiment_name][dataset_name]['means'][data] += [means[data]]
                        dict_means[experiment_name][dataset_name]['number_generations'] += [generations]
                    except:
                        dict_means[experiment_name][dataset_name]['means'][data] = [means[data]]
                        dict_means[experiment_name][dataset_name]['number_generations'] = [generations]

        else:
            for folder in folders:
                dict_means = recursive_experiment_process(
                    os.path.join(this_path, folder),
                    n_samples=n_samples,
                    n_folds=n_folds,
                    dict_means=dict_means,
                    write=write
                )

    return dict_means


def __find_which_level__(path):
    """
    Determines at which level it is.
    """

    level = 1

    folders = os.listdir(path)
    if 'overall' not in folders:
        level = 1 + max([__find_which_level__(os.path.join(path, x)) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))])

    return level


def generate_raw(path):
    level = __find_which_level__(path)

    lines = []
    header = None

    if level == 2:
        folders = ['.']
    else:
        folders = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

    for folder in folders:
        sub_path = os.path.join(path, folder)

        datasets_names = [x for x in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, x))]
        for dataset_name in datasets_names:
            try:
                df = pd.read_csv(
                    os.path.join(sub_path, dataset_name, 'overall', 'summary.csv'),
                    header=[0, 1], index_col=0
                )
                line_selector = ['-mean-of-means' in x for x in df.index]
                data = df.loc[line_selector, ('unweighted_area_under_roc', 'mean')]

                if header is None:
                    header = ['experiment_name', 'dataset_name'] + data.index.tolist()

                lines += [[folder, dataset_name] + data.values.tolist()]
            except FileNotFoundError:  # this dataset was not completed; just ignore it
                pass

    last = pd.DataFrame(lines, columns=header)
    last.to_csv(os.path.join(path, 'final_summary.csv'), index=False)
    return last


def for_comparison(df, experiment_path):
    # df = pd.read_csv(args.csv_path, index_col=[0, 1])

    df.index = pd.MultiIndex.from_arrays([df['experiment_name'], df['dataset_name']])
    del df['experiment_name']
    del df['dataset_name']

    datasets_names = df.index.get_level_values('dataset_name').unique().sort_values().tolist()
    experiments_names = df.index.get_level_values('experiment_name').unique().sort_values().tolist()

    new_table_columns = ['dataset_name']

    res = []
    for experiment in experiments_names:
        to_process_columns = []
        for column in df.columns:
            splitted = column.split('-')  # type: list
            index_mean = splitted.index('mean')
            if index_mean - 1 == 0:
                to_process_columns += [column]
                new_table_columns += ['_'.join([experiment, splitted[0]])]

        for dataset in datasets_names:
            experiments_res = []
            try:
                experiments_res += df.loc[(experiment, dataset)][to_process_columns].tolist()
            except KeyError:
                experiments_res += [np.repeat(np.NaN, len(to_process_columns)).tolist()]

            res += [[dataset] + experiments_res]

    new_table = pd.DataFrame(res, columns=new_table_columns)

    # new_table = pd.DataFrame(
    #     res, columns=['dataset_name'] + list(reduce(op.add, [[x + '_last', x + '_overall'] for x in experiments_names]))
    # )

    new_table.to_csv(
        os.path.join(args.experiment_path, "for_comparison.csv"), index=False
    )

    outer_hypers = []
    hyper_columns = []
    for experiment in experiments_names:
        j = json.load(open(os.path.join(experiment_path, experiment, 'parameters.json')))
        hypers_names = sorted(j.keys())
        hyper_columns = ['experiment_name'] + hypers_names
        local_hypers = []
        local_hypers += [experiment]
        for hyper_name in hypers_names:
            local_hypers += [j[hyper_name]]
        outer_hypers += [local_hypers]

    hypers_table = pd.DataFrame(outer_hypers, columns=hyper_columns)
    hypers_table.to_csv(
        os.path.join(experiment_path, "hyperparameters.csv")
    )


def generate_mean_json(dict_means, experiment_path):
    column_names = None
    datasets_names = set()
    metadata_generations = dict()
    for experiment in dict_means:
        metadata_generations[experiment] = dict()
        datasets_names = datasets_names.union(set(dict_means[experiment]))
        for dataset in dict_means[experiment]:
            metadata_generations[experiment][dataset] = dict()
            if column_names is None:
                column_names = list(dict_means[experiment][dataset]['means'].keys())
            for name in dict_means[experiment][dataset]['means']:
                metadata_generations[experiment][dataset][name] = {
                    'mean': np.mean(dict_means[experiment][dataset]['means'][name]),
                    'std': np.std(dict_means[experiment][dataset]['means'][name])
                }

            metadata_generations[experiment][dataset]['number_generations'] = {
                'mean': np.mean(dict_means[experiment][dataset]['number_generations']),
                'std': np.std(dict_means[experiment][dataset]['number_generations']),
            }

    # pd.DataFrame.from_records(
    #     metadata_generations,
    #     index=pd.MultiIndex.from_product([list(metadata_generations.keys()), list(datasets_names)]),
    #     # columns=pd.MultiIndex.from_product([column_names, ['mean', 'std']])
    # ).to_csv(os.path.join(experiment_path, 'metadata_generations.csv'))

    json.dump(
        metadata_generations,
        open(
            os.path.join(experiment_path, 'metadata_generations.json'), 'w'
        ),
        indent=2
    )


def main(experiment_path, write=True):
    n_samples, n_folds = get_n_samples_n_folds(experiment_path)
    dict_means = recursive_experiment_process(
        experiment_path, n_samples=n_samples, n_folds=n_folds, dict_means=dict(), write=write
    )
    try:
        generate_mean_json(dict_means=dict_means, experiment_path=experiment_path)
    except:
        pass

    raw = generate_raw(experiment_path)
    for_comparison(df=raw, experiment_path=experiment_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='After running experiments (one or more), use this script to generate a summary of the 10-fold '
                    'cross validation procedures. If more than one sample was taken, it will display the mean metrics '
                    'in the document. It also generates a csv with the comparison of all experiments.'
    )

    parser.add_argument(
        '--experiment-path', action='store', required=True,
        help='Either a path to where several experiments are, or the path of a single experiment. Will act according.'
    )

    args = parser.parse_args()
    main(experiment_path=args.experiment_path)

