"""
A script for sampling hyper-parameters for EDNEL, for future use in experiments.
"""
import argparse
from copy import deepcopy
import numpy as np
import os
import pandas as pd


def generate_ednel():
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
               '--timeout 10000 --timeout_individual 60 --log ' \
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
        "max_parents": [1, 6],
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

    with open('a_sets_experiments.sh', 'w') as a_file, open('b_sets_experiments.sh', 'w') as b_file:
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


def interpret_singles(results_path):
    files = [x for x in os.listdir(results_path) if not os.path.isdir(os.path.join(results_path, x))]
    for some_file in files:
        df = pd.read_csv(os.path.join(results_path, some_file))
        z = 0

        break  # TODO remove later


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for either generating or interpreting trials of random search procedure for hyper-parameter'
                    'optimization.'
    )

    parser.add_argument(
        '--results-path', action='store', required=True,
        help='A path to where results of random search trials are stored as .csv files.'
    )

    args = parser.parse_args()
    interpret_singles(results_path=args.results_path)
