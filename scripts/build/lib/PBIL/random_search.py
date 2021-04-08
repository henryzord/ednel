"""
A script for sampling hyper-parameters for EDNEL, for future use in experiments.
"""
import argparse
import sys
from copy import deepcopy
import numpy as np
import os
import pandas as pd
from functools import reduce
import operator as op
import itertools as it


from sklearn.feature_selection import mutual_info_regression
from characteristics_to_pca import to_all_numeric_columns


def generate_ednel_search(output_path):
    template = 'java -Xmx6G -jar ednel.jar --datasets_path keel_datasets_10fcv ' \
               '--metadata_path /A/henry/ednel/metadata/<experiment_set> --n_samples 1 --thinning_factor 0 ' \
               '--timeout 3600 --timeout_individual 60 --log ' \
               '--n_jobs 10 --n_generations 30 --n_individuals 100 --selection_share ' \
               '<selection_share> --burn_in <burn_in> --max_parents <max_parents> --early_stop_generations ' \
               '<early_stop_generations> --delay_structure_learning <delay_structure_learning> --learning_rate ' \
               '<learning_rate> --datasets_names <datasets_names>\n'

    n_samples = 25

    parameters = {
        "selection_share": [0.1, 0.9],
        "learning_rate": [0.1, 1],
        "burn_in": [0, 101],
        "max_parents": [0, 3],
        "delay_structure_learning": [0, 26],
        "early_stop_generations": [5, 26]
    }

    with open(os.path.join(output_path, 'ednel_experiments.sh'), 'w') as write_file:
        write_file.write('#!/bin/bash\n')

        for i in range(n_samples * 2):
            cpy = deepcopy(template)

            this_sample = dict()
            for parameter, range_vals in parameters.items():
                if isinstance(range_vals, list):
                    if isinstance(range_vals[0], float):  # floating point
                        this_sample[parameter] = np.random.choice(np.linspace(range_vals[0], range_vals[1]))
                    else:  # integer
                        this_sample[parameter] = np.random.choice(np.arange(range_vals[0], range_vals[1]))
                else:
                    this_sample[parameter] = range_vals

                cpy = cpy.replace('<' + parameter + '>', str(this_sample[parameter]))

            cpy = cpy.replace('<experiment_set>', 'a_group' if ((i % 2) == 0) else 'b_group')
            cpy = cpy.replace('<datasets_names>', '<a_datasets>' if ((i % 2) == 0) else '<b_datasets>')
            write_file.write(cpy)


def plot_search_space(df, characteristics):
    fitness = df['unweighted_area_under_roc']['mean']
    fitness.index = fitness.index.droplevel(0)

    char_df_columns = reduce(
        op.add,
        map(
            lambda x: [x.split('=')[0]] if len(x) > 0 else [],
            characteristics['characteristics'].iloc[0].split(';')
        )
    )

    dict_cols = {c: list() for c in char_df_columns}
    dict_cols['n_draw'] = list()
    dict_cols['fitness'] = list()

    for i, row in characteristics.iterrows():
        pairs = dict(map(lambda x: x.split('=') if len(x) > 0 else ['n_draw', row['n_draw']], row['characteristics'].split(';')))
        try:
            pairs['fitness'] = fitness.loc[pairs['n_draw']]
            for k, v in pairs.items():
                dict_cols[k] += [v]
        except KeyError:  # happens when trying to query a draw that was removed for not being complete; does nothing
            pass

    char_df = pd.DataFrame(dict_cols)

    for column in char_df.columns:
        if not pd.api.types.is_numeric_dtype(char_df[column]):
            char_df[column] = char_df[column].astype('category')

    char_df.index = char_df['n_draw']
    del char_df['n_draw']

    all_numeric = to_all_numeric_columns(char_df)
    to_mutual_info = all_numeric.copy(deep=True)

    fitness = to_mutual_info['fitness']
    del to_mutual_info['fitness']

    infos = mutual_info_regression(to_mutual_info, fitness)
    print('most relevant hyper-parameters (along with most frequent values):')
    first_five = infos.argsort()[::-1][:5]
    for i in first_five:
        counted = to_mutual_info[to_mutual_info.columns[i]].value_counts(normalize=True, ascending=False)
        print('\t%s: %.2f (%r appeared %.2f%%)' % (
                to_mutual_info.columns[i], infos[i], counted.index[0], counted.iloc[0] * 100
            )
        )

    # from matplotlib import pyplot as plt
    # from matplotlib import cm
    # from sklearn.neighbors import KNeighborsRegressor
    # from characteristics_to_pca import run_pca
    # transformed = run_pca(all_numeric)
    # margin = 0.1
    # mesh_size = 0.01
    #
    # # Load and split data
    # xrange = np.arange(transformed.x.min() - margin, transformed.x.max() + margin, mesh_size)
    # yrange = np.arange(transformed.y.min() - margin, transformed.y.max() + margin, mesh_size)
    # xx, yy = np.meshgrid(xrange, yrange)
    #
    # # Create classifier, run predictions on grid
    # clf = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    # clf.fit(transformed[['x', 'y']], transformed.fitness)
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    #
    # fig, ax = plt.subplots()
    #
    # cs = ax.contourf(xx, yy, Z, cmap=cm.PuBu_r)
    # ax.scatter(transformed['x'], transformed['y'], c=transformed['fitness'], cmap=cm.PuBu_r)
    # ax.axis('off')
    # plt.show()


def interpret_search(results_path: str, max_draws: int):
    files = [x for x in os.listdir(results_path) if (not os.path.isdir(os.path.join(results_path, x)) and x.split('.')[-1] == 'csv')]

    with open(os.path.join(results_path, 'script_experiments.sh'), 'w') as ff:
        template = "java -classpath ednel.jar ednel.RandomSearchApply --datasets_path <datasets_path> " \
                   "--datasets_names <datasets_names> " \
                   "--metadata_path <metadata_path> --string_options \"<string_options>\" " \
                   "--string_characteristics \"<string_characteristics>\" --n_samples <n_samples>"

        for i, some_file in enumerate(files):
            print('file: %s' % os.path.join(results_path, some_file))

            df = pd.read_csv(os.path.join(results_path, some_file))
            if len(df['classifier'].unique()) > 1:
                raise ValueError('%s table must have at most one classifier!' % os.path.join(results_path, some_file))

            clf_name = df['classifier'].unique()[0]

            gbo = df.groupby(by=['classifier', 'n_draw'])
            proper = gbo.agg([np.mean, np.std])

            count = df.groupby(by=['n_draw', 'n_sample']).count()['n_fold']

            # gets name of draws that were complete
            to_drop_draws = count.loc[count.values != count.max()].index.get_level_values(0).values.tolist()
            set_draws = set(df['n_draw'].unique()) - set(to_drop_draws)
            if len(set_draws) > max_draws:
                to_drop_draws.extend(np.random.choice(list(set_draws), replace=False, size=len(set_draws) - max_draws))

            if len(to_drop_draws) > 0:
                zipped = list(it.product([clf_name], to_drop_draws))
                # to_drop_indices = pd.MultiIndex.from_product([clf_name], to_drop_draws)
                to_drop_indices = pd.MultiIndex.from_tuples(zipped)
                proper = proper.drop(to_drop_indices)
                # print('removed %d draws in file %s for being incomplete' % (len(zipped), some_file), file=sys.stderr)
                print('removed %d draws: [%s]' % (len(to_drop_draws), ','.join(map(str, to_drop_draws))), file=sys.stderr)

            best_draw = proper['unweighted_area_under_roc']['mean'].idxmax()[1]
            print('Classifier: %s AUC: %f best draw: %d' % (
                    clf_name,
                    proper['unweighted_area_under_roc']['mean'].max(),
                    best_draw
                )
            )

            # print('string to use:')
            # print(
            #     'java -classpath ednel.jar ednel.RandomSearchApply --datasets_path <datasets_path> '
            #     '--datasets_names NOT_%s --metadata_path <metadata_path> --string_options \"-%s %s\" '
            #     '--string_characteristics \"%s\" --n_samples <n_samples>' % (
            #         clf_name,
            #         df.loc[df['n_draw'] == best_draw]['options'].values[0],
            #         df.loc[df['n_draw'] == best_draw]['characteristics'].values[0],
            #     )
            # )

            tt = deepcopy(template)
            tt = tt.replace('<string_options>', '-%s %s' % (clf_name, df.loc[df['n_draw'] == best_draw].iloc[0]['options']))
            tt = tt.replace('<string_characteristics>', df.loc[df['n_draw'] == best_draw].iloc[0]['characteristics'])
            tt = tt.replace('<datasets_names>', 'NOT_%s' % ','.join(df['dataset_name'].unique().tolist()))
            tt = tt.replace('<datasets_names>', '<datasets_names_%d>' % i)

            ff.write(tt + '\n')

            plot_search_space(proper, characteristics=df[['n_draw', 'characteristics']].drop_duplicates())


def interpret_apply(results_path):
    files = [x for x in os.listdir(results_path) if
             (not os.path.isdir(os.path.join(results_path, x)) and x.split('.')[-1] == 'csv')]

    dict_dataframes = dict()

    for some_file in files:
        df = pd.read_csv(os.path.join(results_path, some_file))

        characteristics = [x for x in df['characteristics'].iloc[0].split(';') if len(x) > 0]
        alg_names = np.unique(list(map(lambda x: x.split('=')[0].split('_')[0], characteristics)))

        grouped = df.groupby(by=['dataset_name']).agg([np.mean, np.std])

        this_hash = hash(','.join(grouped.index))
        active_classifiers = ','.join(alg_names)

        if this_hash not in dict_dataframes:
            dict_dataframes[this_hash] = list()
        dict_dataframes[this_hash] += [(active_classifiers, grouped[['unweighted_area_under_roc']])]

        print('activate classifiers: %s file: %s' % (active_classifiers, some_file))
        print(grouped[['unweighted_area_under_roc']])

    for k in dict_dataframes.keys():
        collapsed = None

        for i, (clf_name, data) in enumerate(dict_dataframes[k]):
            data.columns = [clf_name + '_mean', clf_name + '_std']
            if i == 0:
                collapsed = data
            else:
                collapsed = collapsed.join(data)

        collapsed.to_csv(os.path.join(results_path, str(k) + '.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for either generating or interpreting trials of random search procedure for hyper-parameter'
                    'optimization.'
    )

    parser.add_argument(
        '--search-results-path', action='store', required=False,
        help='A path to where results of random search trials are stored as .csv files.'
    )

    parser.add_argument(
        '--apply-results-path', action='store', required=False,
        help='Path to where results of already-applied best hyper-parameters are stored, as .csv files.'
    )

    parser.add_argument(
        '--ednel-search-output', action='store', required=False,
        help='Path to where hyper-parametrizations for EDNEL will be written.'
    )

    parser.add_argument(
        '--max-draws', action='store', required=False, type=int,
        help='Maximum number of draws to use per file. For example, if 25 draws are present, but one wish to use only '
             '10 (e.g. one of the baseline algorithms did not complete all draws), then 10 draws will be randomly '
             'sampled from the 25 group.'
    )

    args = parser.parse_args()

    if args.search_results_path is not None:
        interpret_search(results_path=args.search_results_path, max_draws=args.max_draws)
    elif args.apply_results_path is not None:
        interpret_apply(results_path=args.apply_results_path)
    elif args.ednel_search_output is not None:
        generate_ednel_search(output_path=args.ednel_search_output)
    else:
        raise Exception('should either generate hyper-parameters or interpret them!')
