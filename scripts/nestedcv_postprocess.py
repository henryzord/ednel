import argparse
import os
import subprocess
import pandas as pd
import numpy as np


def main(experiment_path: str):
    experiments = [x for x in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, x))]

    all_df = pd.DataFrame(
        dtype=np.float,
        columns=pd.MultiIndex.from_product([[None], ['mean', 'std']], names=['classifier', 'metric'])
    )

    for experiment in experiments:
        dataset_names = [x for x in os.listdir(os.path.join(experiment_path, experiment)) if os.path.isdir(os.path.join(experiment_path, experiment, x))]
        for dataset in dataset_names:
            overall_folder = os.path.join(experiment_path, experiment, dataset, 'overall')
            file_preds = [x for x in os.listdir(overall_folder) if '.preds' in x]
            if (len(file_preds) % 10) != 0:
                if os.path.exists(os.path.join(overall_folder, 'summary_1.csv')):
                    os.remove(os.path.join(overall_folder, 'summary_1.csv'))
                continue

            returncode = subprocess.call([
                'java',
                '-classpath',
                os.path.join(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-1]), 'ednel.jar'),
                'ednel.utils.analysis.CompilePredictions',
                '--path_predictions',
                overall_folder
            ])

            if returncode == 0:
                try:
                    df = pd.read_csv(os.path.join(overall_folder, 'summary_1.csv'), index_col=0, header=None)
                    new_columns = pd.MultiIndex.from_arrays(df.iloc[:2].values)
                    df = pd.DataFrame(data=df.iloc[2:].values, columns=new_columns, index=df.index[2:])

                    if dataset not in all_df.index:
                        all_df.loc[dataset] = np.ones(len(all_df.columns), dtype=np.float) * -1

                    for clf in df.index:
                        if 'sample' not in clf:
                            value = df.loc[clf, ('unweightedAreaUnderRoc', ('mean', 'std'))]

                            if clf not in all_df.columns.get_level_values(0):
                                all_df[(clf, 'mean')] = pd.Series(index=all_df.index, dtype=np.float, name=(clf, 'mean'))
                                all_df[(clf, 'std')] = pd.Series(index=all_df.index, dtype=np.float, name=(clf, 'std'))

                            all_df.loc[dataset, (clf, 'mean')] = value[0]
                            all_df.loc[dataset, (clf, 'std')] = value[1]

                except Exception as e:
                    z = 0

    del all_df[(None, 'mean')]
    del all_df[(None, 'std')]

    all_df.replace(to_replace=-1, value=np.nan, inplace=True)

    all_df.to_csv(os.path.join(experiment_path, 'nestedcv_summarized.csv'))


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


