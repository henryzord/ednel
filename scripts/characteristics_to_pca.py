"""
Script for generating a visualization of individuals from a csv table.
"""
import copy
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


def run_pca(df: pd.DataFrame):
    """
    Runs PCA on dataFrame of individual characteristics.
    DataFrame must be already transformed (e.g. categorical attributes to one-hot encoding).

    :param df: DataFrame with meta-characteristics of individuals in all evolutionary process.
    """
    index = deepcopy(df.index)
    fitness = deepcopy(df.fitness.values)

    del df['fitness']
    if 'test_auc' in df.columns:
        del df['test_auc']

    pca = PCA(n_components=2, copy=False)
    npa = pca.fit_transform(df)

    new_df = pd.DataFrame(npa, columns=['x', 'y'], index=index)
    new_df.insert(len(new_df.columns), column='fitness', value=fitness)
    return new_df


def read_population_dataframe(csv_path: str):
    """
    Reads a csv file into a DataFrame.

    This csv file must be the result of a (partial) run of EDNEL.
    """
    df = pd.read_csv(csv_path, index_col=0)
    for column in df.columns:
        if df[column].dtype == np.object:
            df[column].fillna('null', inplace=True)
            df.loc[:, column] = df[column].astype('category')

    index = df.apply(lambda x: x.name.split('_')[:2][-1], axis=1)
    df.index = index
    if 'test_auc' in df.columns:
        del df['test_auc']
    return df


def to_all_numeric_columns(df: pd.DataFrame):
    """
    Will swap NaN values with -1, and convert categorical columns to one-hot encoding.

    :param df: a DataFrame with categorical columns
    :type df: pandas.DataFrame
    :return: The transformed DataFrame.
    :rtype: pandas.DataFrame
    """

    new_df = copy.deepcopy(df)
    
    for column in new_df.columns:
        if new_df[column].dtype.name == 'category':
            one_hot = pd.get_dummies(new_df[column])
            new_df = new_df.drop(column, axis=1)
            for subcolumn in one_hot.columns:
                new_df.loc[:, column + '_' + str(subcolumn)] = one_hot[subcolumn]

    # replaces nan values with -1
    new_df = new_df.fillna(value=-1)
    return new_df


def update_population_contour(df: pd.DataFrame, gen: str, n_neighbors: int = 1, mesh_size: float = 0.5):
    margin = 0.1

    fig = go.Figure(
        layout=go.Layout(
            title='Individuals throughout generations',
            xaxis=go.layout.XAxis(
                range=(min(df['x']) - margin, max(df['x']) + margin),
                showticklabels=False,
                title='First PCA dimension'
            ),
            yaxis=go.layout.YAxis(
                range=(min(df['y']) - margin, max(df['y']) + margin),
                showticklabels=False,
                title='Second PCA dimension'
            )
        )
    )  # type: go.Figure

    # removes last and overall individuals from DataFrame, to be added later
    # otherwise scale of colors will be compromised
    df = df.drop(['last', 'overall'])

    # Load and split data
    xrange = np.arange(df.x.min() - margin, df.x.max() + margin, mesh_size)
    yrange = np.arange(df.y.min() - margin, df.y.max() + margin, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Create classifier, run predictions on grid
    clf = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
    clf.fit(df[['x', 'y']], df.fitness)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the figure
    fig.add_trace(
        go.Contour(
            x=xrange,
            y=yrange,
            z=Z,
            colorscale='RdBu',
            hovertemplate='Predicted fitness: %{z}',
            name='%03d generation' % int(gen)
        )
    )

    sub = df.loc[(df.index == gen)]

    fig.add_trace(
        go.Scatter(
            x=sub.x,
            y=sub.y,
            mode='markers',
            visible=True,
            marker=dict(
                color='white',
                size=4,
            ),
            text=['Actual fitness: %f' % x for x in df.fitness],
            hovertemplate='%{text}',
            name='%03d generation' % int(gen)
        )
    )

    return fig


def update_population_scatter(df: pd.DataFrame, csv_path: str):
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

    # removes last and overall individuals from DataFrame, to be added later
    # otherwise scale of colors will be compromised
    df = df.drop(['last', 'overall'])

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
                    ),
                    cmin=df.fitness.min(),
                    cmax=df.fitness.max()
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
