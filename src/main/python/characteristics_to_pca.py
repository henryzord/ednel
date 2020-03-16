import argparse
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.offline import plot
from IPython.display import display, HTML
from matplotlib.colors import to_hex
from matplotlib.cm import viridis
from copy import deepcopy

"""
Script for generating a visualization of individuals from a csv table.
"""


def read_and_discretize(csv_path):
    """
    Reads a csv file into a DataFrame.

    This csv file must be the result of a (partial) run of the EDNEL algorithm.
    Will swap NaN values with -1, and convert categorical columns to one-hot.

    :param csv_path: Full path to the csv file.
    :type csv_path: str
    :return: A dataframe with the information.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(csv_path, index_col=0)
    for column in df.columns:
        if df[column].dtype == np.object:
            df.loc[:, column] = df[column].astype('category')
            one_hot = pd.get_dummies(df[column])
            df = df.drop(column, axis=1)
            for subcolumn in one_hot.columns:
                df.loc[:, column + '_' + str(subcolumn)] = one_hot[subcolumn]

    # replaces nan values with -1
    df = df.fillna(value=-1)
    return df


def run_pca(df):
    index = deepcopy(df.apply(lambda x: x.name.split('_')[:2][-1], axis=1))
    fitness = deepcopy(df['fitness'].values)

    pca = PCA(n_components=2, copy=False)
    npa = pca.fit_transform(df)

    new_df = pd.DataFrame(npa, columns=['x', 'y'], index=index)
    new_df.insert(len(new_df.columns), column='fitness', value=fitness)
    return new_df


def local_plot(df, csv_path):
    fig = go.Figure(
        layout=go.Layout(
            title='Individuals throughout generations',
            xaxis=go.layout.XAxis(
                range=(min(df['x']) - 2, max(df['x']) + 2),
                showticklabels=False,
                title='First PCA dimension'
            ),
            yaxis=go.layout.YAxis(
                range=(min(df['y']) - 2, max(df['y']) + 2),
                showticklabels=False,
                title='Second PCA dimension'
            )
        )
    )

    active = 0

    numeric_gens = []
    categoric_gens = []
    for gen in df.index.unique():
        try:
            numeric_gens += [int(gen)]
        except:
            categoric_gens += [gen]

    always_display = np.array(list(map(lambda x: x in categoric_gens, df.index)))

    # add traces, one for each slider step
    for gen in sorted(numeric_gens):
        sub = df.loc[(df.index == '%03d' % gen) | always_display]
        x = sub['x']
        y = sub['y']

        hovertext = []
        for i, row in sub.iterrows():
            if i in categoric_gens:
                hovertext += [i]
            else:
                hovertext += ['Fitness: %.4f' % row.loc['fitness']]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                visible=False,
                marker=dict(
                    color=sub['fitness'],
                    size=12,
                    colorscale='Viridis',
                    colorbar=dict(
                        title='Fitness'
                    )
                ),
                hovertext=hovertext,
            )
        )

    # makes first scatter visible
    fig.data[active].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=active,
        currentvalue={"prefix": "Generation: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    to_write_path = os.sep.join(csv_path.split(os.sep)[:-1])
    plot(fig, filename=os.path.join(to_write_path, 'characteristics.html'))


def main(args):
    df = read_and_discretize(args.csv_path)
    new_df = run_pca(df)
    local_plot(new_df, args.csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for collapsing metrics'
    )

    parser.add_argument(
        '--csv-path', action='store', required=True,
        help='Path to .csv with characteristics of individuals in a run of the algorithm.'
    )

    args = parser.parse_args()

    main(args)
