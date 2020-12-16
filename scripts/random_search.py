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
    a_sets = ['hayes-roth', 'tae', 'haberman', 'newthyroid', 'bupa', 'wine', 'balancescale', 'bloodtransfusion',
              'heart', 'cleveland', 'mammographic', 'banknotes', 'pima', 'tictactoe', 'australian', 'spectfheart',
              'car', 'vowel', 'contraceptive', 'diabetic', 'soybean', 'syntheticcontrol', 'segment',
              'artificialcharacters', 'chess', 'thyroid', 'turkiye', 'waveform', 'magic']
    b_sets = ['iris', 'breast', 'monk-2', 'led7digit', 'saheart', 'wisconsin', 'titanic', 'crx', 'creditapproval',
              'ionosphere', 'sonar', 'flare', 'dermatology', 'banana', 'vehicle', 'wdbc', 'german', 'phoneme',
              'seismicbumps', 'drugconsumption', 'page-blocks', 'steelfaults', 'krvskp', 'ring', 'twonorm', 'penbased',
              'splice', 'texture', 'spambase']

    template = 'java -Xmx6G -jar ednel.jar --datasets_path keel_datasets_10fcv ' \
               '--metadata_path /A/henry/ednel/metadata/<experiment_set> --n_samples 1 --thinning_factor 0 ' \
               '--timeout 3600 --timeout_individual 60 --log ' \
               '--n_jobs 5 --n_generations <n_generations> --n_individuals <n_individuals> --selection_share ' \
               '<selection_share> --burn_in <burn_in> --max_parents <max_parents> --early_stop_generations ' \
               '<early_stop_generations> --delay_structure_learning <delay_structure_learning> --learning_rate ' \
               '<learning_rate> --datasets_names <datasets_names>\n'

    n_samples = 25

    parameters = {
        "n_individuals": [25, 201],
        "n_generations": [25, 201],
        "selection_share": [0.1, 0.9],
        "learning_rate": [0.1, 1],
        "burn_in": [0, 101],
        "max_parents": [0, 3],
        "delay_structure_learning": [0, 26],
        "early_stop_generations": [5, 26]
    }

    sampled = []

    for i in range(n_samples):
        this_sample = dict()
        for parameter, range_vals in parameters.items():
            if isinstance(range_vals, list):
                if isinstance(range_vals[0], float):  # floating point
                    this_sample[parameter] = np.random.choice(np.linspace(range_vals[0], range_vals[1]))
                else:  # integer
                    this_sample[parameter] = np.random.choice(np.arange(range_vals[0], range_vals[1]))
            else:
                this_sample[parameter] = range_vals

        sampled += [this_sample]

    with open(os.path.join(output_path, 'a_sets_experiments.sh'), 'w') as a_file, \
            open(os.path.join(output_path, 'b_sets_experiments.sh'), 'w') as b_file:

        a_file.write('#!/bin/bash\n')
        b_file.write('#!/bin/bash\n')

        for i in range(len(sampled)):
            a_cpy = deepcopy(template)
            b_cpy = deepcopy(template)
            for param in sampled[i].keys():
                a_cpy = a_cpy.replace('<' + param + '>', str(sampled[i][param]))
                b_cpy = b_cpy.replace('<' + param + '>', str(sampled[i][param]))

            a_cpy = a_cpy.replace('<datasets_names>', ','.join(a_sets))
            b_cpy = b_cpy.replace('<datasets_names>', ','.join(b_sets))

            a_cpy = a_cpy.replace('<experiment_set>', 'a_experiments')
            b_cpy = b_cpy.replace('<experiment_set>', 'b_experiments')

            a_file.write(a_cpy)
            b_file.write(b_cpy)


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


def interpret_search(results_path):
    files = [x for x in os.listdir(results_path) if (not os.path.isdir(os.path.join(results_path, x)) and x.split('.')[-1] == 'csv')]

    with open(os.path.join(results_path, 'script_experiments.sh'), 'w') as ff:
        template = "java -classpath ednel.jar ednel.RandomSearchApply --datasets_path <datasets_path> " \
                   "--datasets_names <datasets_names> " \
                   "--metadata_path <metadata_path> --string_options \"<string_options>\" " \
                   "--string_characteristics \"<string_characteristics>\" --n_samples <n_samples>"

        for i, some_file in enumerate(files):
            df = pd.read_csv(os.path.join(results_path, some_file))
            if len(df['classifier'].unique()) > 1:
                raise ValueError('%s table must have at most one classifier!' % os.path.join(results_path, some_file))

            clf_name = df['classifier'].unique()[0]

            gbo = df.groupby(by=['classifier', 'n_draw'])
            proper = gbo.agg([np.mean, np.std])

            count = df.groupby(by=['n_draw', 'n_sample']).count()['n_fold']

            # gets name of draws that were complete
            to_drop_draws = count.loc[count.values != count.max()].index.get_level_values(0).values.tolist()

            if len(to_drop_draws) > 0:
                zipped = list(it.product([clf_name], to_drop_draws))
                # to_drop_indices = pd.MultiIndex.from_product([clf_name], to_drop_draws)
                to_drop_indices = pd.MultiIndex.from_tuples(zipped)
                proper = proper.drop(to_drop_indices)
                print('removed %d draws in file %s for being incomplete' % (len(zipped), some_file), file=sys.stderr)
                print('removed draws: [%s]' % ','.join(map(str, to_drop_draws)), file=sys.stderr)

            best_draw = proper['unweighted_area_under_roc']['mean'].idxmax()[1]
            print('best draw: %d AUC: %f File: %s' % (
                    best_draw,
                    proper['unweighted_area_under_roc']['mean'].max(),
                    os.path.join(results_path, some_file)
                )
            )

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

    for some_file in files:
        df = pd.read_csv(os.path.join(results_path, some_file))

        characteristics = [x for x in df['characteristics'].iloc[0].split(';') if len(x) > 0]
        alg_names = np.unique(list(map(lambda x: x.split('=')[0].split('_')[0], characteristics)))

        grouped = df.groupby(by=['dataset_name']).agg([np.mean, np.std])
        print('activated classifiers: %s file: %s' % (','.join(alg_names), some_file))
        print(grouped[['unweighted_area_under_roc']])


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
        '--output-path', action='store', required=False,
        help='Path to where hyper-parametrizations for EDNEL will be written.'
    )

    args = parser.parse_args()

    if args.search_results_path is not None:
        interpret_search(results_path=args.search_results_path)
    elif args.apply_results_path is not None:
        interpret_apply(results_path=args.apply_results_path)
    elif args.output_path is not None:
        generate_ednel_search(output_path=args.output_path)
    else:
        raise Exception('should either generate hyper-parameters or interpret them!')