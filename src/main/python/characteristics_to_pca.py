import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from ipywidgets import widgets
from plotly import offline as py
from IPython.display import display, HTML

"""
Script for generating a visualization of individuals from a csv table.
"""


def read_and_discretize(csv_path):
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
    pca = PCA(n_components=2, copy=False)
    npa = pca.fit_transform(df)
    index = df.apply(lambda x: x.name.split('_')[:2][-1], axis=1)

    new_df = pd.DataFrame(npa, columns=['x', 'y'], index=index)
    return new_df


def plot(df):
    generations = []
    not_generations = []
    for gen in df.index.unique():
        try:
            generations += [int(gen)]
        except ValueError:
            not_generations += [gen]

    genSlider = widgets.IntSlider(
        value=0,
        min=min(generations),
        max=max(generations),
        step=1,
        description='Generation:',
        continuous_update=False
    )
    show_last = widgets.Checkbox(
        description='Show Last',
        value=True
    )
    show_overall = widgets.Checkbox(
        description='Show Overall',
        value=True
    )

    container = widgets.HBox(children=[genSlider, show_last, show_overall])

    trace1 = go.Figure(
        data=go.Scatter(
            x=df.loc[:, df.columns[0]],
            y=df.loc[:, df.columns[1]],
            mode='markers'
        )
    )

    g = go.FigureWidget(
        data=trace1,
        layout=go.Layout(
            title=dict(
                text='Population throughout generations'
            ),
            barmode='overlay'
        )
    )

    def validate():
        if genSlider.value in generations:
            return True
        return False

    def response(change):
        if validate():
            filter_list = df.index == '%03d' % genSlider.value

            if show_overall:
                filter_list = filter_list | df.index == 'overall'

            if show_last:
                filter_list = filter_list | df.index == 'last'

            local = df.loc[filter_list]

            with g.batch_update():
                g.data[0].x = local[local.columns[0]]
                g.data[1].y = local[local.columns[1]]
                g.layout.barmode = 'overlay'

    genSlider.observe(response, names='value')
    show_last.observe(response, names='value')
    show_overall.observe(response, names='value')

    vbox = widgets.VBox([container, g])
    display(vbox)


def main(args):
    py.init_notebook_mode()

    df = read_and_discretize(args.csv_path)
    new_df = run_pca(df)
    plot(new_df)


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
