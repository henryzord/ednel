
# utilities
import argparse
import copy
import json
import os
import re
from collections import Counter
from zipfile import ZipFile

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from matplotlib import pyplot as plt
from matplotlib.cm import Pastel1, viridis
from matplotlib.colors import to_hex
from networkx.drawing.nx_agraph import graphviz_layout
from pandas.api.types import is_numeric_dtype

from characteristics_to_pca import read_population_dataframe, run_pca, update_population_contour, to_all_numeric_columns


def read_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}


def print_version(G: nx.DiGraph):
    """
    Generates a pdf of deterministic relationship between variables, to be used in a document (e.g. paper).
    :param G: deterministic DiGraph
    """

    def __get_node_label__(_label, _line_limit=11):
        _candidate = _label.split('_')[-1]  # type: str
        if len(_candidate) <= _line_limit:
            return _candidate

        # else
        _words = []
        _last_index = 0
        _before_last_case = 'lower' if _candidate[0].islower() else 'upper'
        _last_case = 'lower' if _candidate[1].islower() else 'upper'
        for i in range(2, len(_candidate)):
            _current_case = 'lower' if _candidate[i].islower() else 'upper'

            if _current_case != _last_case:
                # all uppercase word
                if (_before_last_case == 'upper') and (_last_case == 'upper'):
                    _words += [_candidate[_last_index:i]]
                    _last_index = i
                elif _last_case == 'lower' and _current_case == 'upper':
                    _words += [_candidate[_last_index:i]]
                    _last_index = i

            _before_last_case = _last_case
            _last_case = _current_case

        _words += [_candidate[_last_index:len(_candidate)]]

        _answer = ''
        _line_counter = 0
        for i, _word in enumerate(_words):
            if (_line_counter + len(_word) >= _line_limit) and i > 0:
                _answer += '\n' + _word
                _line_counter = len(_word)
            else:
                _answer += _word
                _line_counter += len(_word)

        return _answer

    out_degrees = dict(G.out_degree)
    min_degree = min(out_degrees.values())
    roots = [k for k, v in out_degrees.items() if v == min_degree]

    degrees = dict()
    max_degree = -np.inf
    for node in G.nodes:
        for root in roots:
            try:
                degree = len(nx.shortest_path(G, source=node, target=root)) - 1
                max_degree = max(degree, max_degree)
                try:
                    degrees[node] = min(degrees[node], degree)
                except KeyError:
                    degrees[node] = degree
            except nx.exception.NetworkXNoPath:
                pass

    all_colors = list(map(to_hex, viridis(np.linspace(0, 1, num=10))))

    colors = all_colors[5::2]

    node_colors = list()
    node_labels = dict()
    for node in G.nodes:
        node_colors += [colors[degrees[node]]]
        node_labels[node] = __get_node_label__(node) # type: str

    fig, ax = plt.subplots(figsize=(16, 10))

    pos = graphviz_layout(G, root='0', prog='neato')

    edge_list = G.edges(data=True)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, node_color=node_colors, edgecolors='black', alpha=1)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list, style='solid', alpha=1)
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=8)

    plt.axis('off')
    plt.show()


def get_plot_colors(df: pd.DataFrame):
    """
    Builds a dictionary for colors for different projections.
    """
    numeric_columns = []
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            numeric_columns += [column]

    plot_colors = dict(zip(
        numeric_columns,
        map(lambda x: to_hex(Pastel1(x)), np.linspace(0, 1, num=len(numeric_columns)))
    ))

    return plot_colors


def get_node_colors(all_variables: list):
    """
    Builds a dictionary assigning a color to each variable.
    """
    families = list(zip(*map(lambda x: x.split('_'), all_variables)))[0]
    families_set = sorted(list(set(families)))
    families_colors = dict(zip(
        families_set,
        map(lambda x: to_hex(Pastel1(x)), np.linspace(0, 1, num=len(families_set)))
    ))

    var_color_dict = dict(zip(all_variables, [families_colors[x] for x in families]))
    return var_color_dict


def read_probabilistic_graphs(probabilistic_path: str):
    """
    Given a path to a json file, reads a JSON that encodes a Dependency Network structure (with probabilities)
    throughout an evolutionary process of EDNEL.
    """

    dict_str = read_zip(probabilistic_path)
    _dict = json.loads(list(dict_str.values())[0])

    structure_dict = dict()
    probabilities_dict = dict()

    eq_splitter = lambda x: re.split('=(?![^(]*\))', x)
    co_splitter = lambda x: re.split(',(?![^(]*\))', x)

    for gen in _dict.keys():
        G = nx.DiGraph()
        for node in _dict[gen].keys():
            G.add_node(node)

        this_gen_probabilities = dict()
        for variable in _dict[gen].keys():
            lines = list(_dict[gen][variable].keys())
            probs = list(_dict[gen][variable].values())
            splitted_lines = list(map(co_splitter, lines))

            table = []
            parentnames = None
            for i, splitted_line in enumerate(splitted_lines):
                _vars, _vals = zip(*(map(eq_splitter, splitted_line)))
                table += [list(_vals) + [probs[i]]]
                parentnames = _vars

            for parent in list(set(parentnames) - {variable}):
                G.add_edge(variable, parent)

            this_gen_probabilities[variable] = pd.DataFrame(table, columns=list(parentnames) + ['probability'])

        structure_dict[gen] = G
        probabilities_dict[gen] = this_gen_probabilities

    return structure_dict, probabilities_dict


def update_probabilities_table(probs, gen, variable):
    df = probs[gen][variable]  # type: pd.DataFrame

    sortable_columns = list(df.columns)
    sortable_columns.remove('probability')

    df.sort_values(by=sortable_columns, inplace=True)

    dt = dash_table.DataTable(
        id='probabilities-table',
        columns=[
            {"name": i, "id": i} for i in list(probs[gen][variable].columns)
        ],
        data=df.to_dict('records'),
        sort_action="native",
        sort_mode="multi",
    )
    return dt


def add_probabilities_table(probs, gen, variable):
    probabilities_table = html.Div(
        [update_probabilities_table(probs, gen, variable)],
        id='probabilities-table-container'
    )
    return probabilities_table


def update_gen_structure_map(prob_structs, gen, var_color_dict):
    fig = go.Figure(
        layout=go.Layout(
            title='Dependency network structure throughout generations',
            xaxis=go.layout.XAxis(
                visible=False
            ),
            yaxis=go.layout.YAxis(
                visible=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    G = prob_structs[gen]
    pos = graphviz_layout(G, prog='neato')  # generates layout for variables

    variable_names = list(pos.keys())
    x, y = zip(*list(pos.values()))

    node_list = G.nodes(data=True)
    edge_list = G.edges(data=True)

    node_colors = [var_color_dict[nd[0]] for nd in node_list]

    # adds probabilistic relationship between variables
    for some_tuple in edge_list:
        edge_properties = some_tuple[2]

        a = some_tuple[0]
        b = some_tuple[1]

        a_coord = pos[a]
        b_coord = pos[b]
        fig.add_annotation(
            x=a_coord[0],  # arrows' head
            y=a_coord[1],  # arrows' head
            ax=b_coord[0],  # arrows' tail
            ay=b_coord[1],  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            standoff=11,  # move away head of arrow n pixels
            startstandoff=9,  # move away base of arrow n pixels
            arrowcolor='black'
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=node_colors,
                size=20,
            ),
            text=variable_names,
            hovertemplate='%{text}',
            name='%03d generation' % int(gen)
        )
    )

    return fig


def add_gen_structure_map(prob_structs, gen, var_color_dict):
    fig = update_gen_structure_map(prob_structs=prob_structs, gen=gen, var_color_dict=var_color_dict)
    structure_map = dcc.Graph(id='structure-map', figure=fig)
    return structure_map


def add_variable_dropdown(variables: list):
    variable_dropdown = dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': s, 'value': s} for s in variables],
        value=variables[0]
    )
    return variable_dropdown


def add_generation_slider(gens: list):
    int_gens = list(map(int, gens))

    generation_slider = dcc.Slider(
                id='generation-slider',
                min=min(int_gens),  # min value
                max=max(int_gens),  # max value
                value=min(int_gens),  # current value
                marks={int_gens[0]: gens[0], int_gens[-1]: gens[-1], int_gens[int(len(int_gens)/2)] : gens[int(len(gens)/2)]},
                step=1
            )
    return generation_slider


def add_population_contour(population_df: pd.DataFrame, gen):
    fig = update_population_contour(population_df, gen=gen)
    graph = dcc.Graph(id='population-map', figure=fig)
    return graph


def add_neighbors_dropdown():
    values = [1, 3, 5]

    neighbors_dropdown = dcc.Dropdown(
        id='neighbors-dropdown',
        options=[{'label': str(s), 'value': s} for s in values],
        value=values[-1]
    )
    return neighbors_dropdown


def add_mesh_dropdown():
    values = [0.5, 0.75, 1]

    mesh_dropdown = dcc.Dropdown(
        id='mesh-dropdown',
        options=[{'label': str(s), 'value': s} for s in values],
        value=values[0]
    )
    return mesh_dropdown


def update_fitness_plot(logger_data, plots, options):
    fig = go.Figure(
        layout=go.Layout(
            # title='Population metrics throughout generations',
            xaxis=go.layout.XAxis(
                visible=True
            ),
            yaxis=go.layout.YAxis(
                visible=True
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    plot_colors = get_plot_colors(logger_data)

    for plot in plots:
        if plot in logger_data.columns and is_numeric_dtype(logger_data[plot]):
            if 'smooth' in options:
                x = np.arange(len(logger_data[plot]) - 1)
                y = [(logger_data[plot].iloc[xi] + logger_data[plot].iloc[xi + 1])/2. for xi in x]

            else:
                x = np.arange(len(logger_data[plot]))
                y = logger_data[plot]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    marker=dict(
                        color=plot_colors[plot],
                        size=20,
                    ),
                    line=dict(
                        smoothing=1.3 if 'smooth' in options else 0.,
                        shape='spline' if 'smooth' in options else 'linear'
                    ),
                    text=[plot] * len(logger_data[plot]),
                    name=plot,
                    hovertemplate='Generation: %{x}<br>%{text}: %{y}',
                )
            )

    if 'logscale' in options:
        fig.update_yaxes(type="log")

    return fig


def add_fitness_plot(logger_data):
    fig = update_fitness_plot(logger_data, plots=['median'], options=[])
    plot = dcc.Graph(id='fitness-plot', figure=fig)
    return plot


def add_plot_dropdown(logger_data):
    plots = [plot for plot in logger_data.columns if is_numeric_dtype(logger_data[plot])]

    dropdown = dcc.Dropdown(
        options=[{'label': plot, 'value': plot} for plot in plots],
        value=['median'],
        multi=True,
        id='plot-dropdown',
    )
    return dropdown


def add_fitness_plot_options():
    options = dcc.Checklist(
        options=[
            {'label': 'smooth', 'value': 'smooth'},
            {'label': 'logscale', 'value': 'logscale'},
        ],
        value=[],
        labelStyle={'display': 'inline-block'},
        id='fitness-plot-checklist'
    )
    return options


def add_chromosome_heatmap(characteristics_df, gen):
    fig = update_chromossome_heatmap(characteristics_df=characteristics_df, gen=gen)
    graph = dcc.Graph(id='chromossome-heatmap', figure=fig)
    return graph


def update_chromossome_heatmap(characteristics_df, gen):
    fig = go.Figure(
        layout=go.Layout(
            title='Most frequent value frequency in each allele',
            xaxis=go.layout.XAxis(
                visible=False
            ),
            yaxis=go.layout.YAxis(
                visible=False
            ),
            height=210,
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    df = characteristics_df.loc[characteristics_df.index == gen]

    columns = sorted(set(df.columns) - {'fitness'})

    common_values_fracts = np.empty((1, len(columns)), dtype=np.float)

    texts = np.empty((1, len(columns)), dtype=np.object)

    for i, column in enumerate(columns):
        counter = Counter(df[column])

        if df[column].dtype.name != 'category':

            keys = copy.deepcopy(list(counter.keys()))

            corrected_counter = dict(null=0)
            for k in keys:
                if np.isnan(k):
                    corrected_counter['null'] += counter[k]
                    del counter[k]

            counter['null'] = corrected_counter['null']

        common_values_fracts[0][i] = counter.most_common(1)[0][1] / len(df)
        texts[0][i] = '%s: %s' % (column, str(counter.most_common(1)[0][0]))

    fig.add_trace(go.Heatmap(
        z=common_values_fracts,
        zmin=0,
        zmax=1,
        colorscale='BuPu',
        colorbar=None,
        text=texts,
        hovertemplate='%{text}<br>Frequency: %{z}',
        name='%03d generation' % int(gen),
        showscale=False
    ))
    return fig


def init_app(structure_map, population_map, chromossome_heatmap, probabilities_table, fitness_plot, plot_dropdown,
             fitness_plot_options, generation_slider, variable_dropdown, neighbors_dropdown, mesh_dropdown):
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    app.layout = html.Div([
        html.Div([
            structure_map,
            html.Div([html.P("Generation"), generation_slider],
                     style={'width': '97%', 'margin-left': '10px', 'margin-bottom': '25px'}),
            chromossome_heatmap,
            html.Div([
                html.Div([html.P("Number of Neighbors"), neighbors_dropdown],
                         style={'width': '49%', 'display': 'inline-block'}),
                html.Div([html.P("Mesh Size"), mesh_dropdown], style={'width': '49%', 'display': 'inline-block'}),
            ]),
            population_map,
        ], className="six columns"),
        html.Div([
            variable_dropdown,
            html.Div([probabilities_table], style={"maxHeight": "415px", 'height': '415px', "overflow": "scroll"}),
            html.P("Population Metrics throughout Generations"),
            html.P("Metrics"),
            plot_dropdown,
            fitness_plot_options,
            fitness_plot
        ], className="six columns"),
    ], id='dash-container')

    return app


def main(args):
    prob_structs, probs = read_probabilistic_graphs(
        os.path.join(args.experiment_path, 'dependency_network_structure.zip')
    )

    if args.print is True:
        print_version(prob_structs)
    elif args.print is False:
        characteristics_df = read_population_dataframe(os.path.join(args.experiment_path, 'characteristics.csv'))
        population_df = run_pca(to_all_numeric_columns(characteristics_df))
        logger_data = pd.read_csv(open(os.path.join(args.experiment_path, 'loggerData.csv'), 'r'), sep=',', quotechar='\"')

        gens = sorted(list(probs.keys()))
        first_gen = gens[0]
        variables = sorted(list(probs[first_gen].keys()))
        any_var = variables[0]
        var_color_dict = get_node_colors(all_variables=variables)

        chromossome_heatmap = add_chromosome_heatmap(characteristics_df=characteristics_df, gen=first_gen)
        structure_map = add_gen_structure_map(prob_structs=prob_structs, gen=first_gen, var_color_dict=var_color_dict)
        population_map = add_population_contour(population_df=population_df, gen=first_gen)
        fitness_plot = add_fitness_plot(logger_data=logger_data)
        fitness_plot_options = add_fitness_plot_options()
        plot_dropdown = add_plot_dropdown(logger_data=logger_data)
        probabilities_table = add_probabilities_table(probs=probs, gen=first_gen, variable=any_var)
        generation_slider = add_generation_slider(gens=gens)
        variable_dropdown = add_variable_dropdown(variables=variables)
        neighbors_dropdown = add_neighbors_dropdown()
        mesh_dropdown = add_mesh_dropdown()

        app = init_app(
            structure_map=structure_map,
            population_map=population_map,
            chromossome_heatmap=chromossome_heatmap,
            probabilities_table=probabilities_table,
            fitness_plot=fitness_plot,
            plot_dropdown=plot_dropdown,
            fitness_plot_options=fitness_plot_options,
            generation_slider=generation_slider,
            variable_dropdown=variable_dropdown,
            neighbors_dropdown=neighbors_dropdown,
            mesh_dropdown=mesh_dropdown
        )

        @app.callback(
            Output('population-map', 'figure'),
            [Input('generation-slider', 'value'),
             Input('neighbors-dropdown', 'value'),
             Input('mesh-dropdown', 'value')]
        )
        def population_map_callback(gen, n_neighbors, mesh_size):
            _str_gen = '%03d' % gen

            struct_map_fig = update_population_contour(
                df=population_df, gen=_str_gen, n_neighbors=n_neighbors, mesh_size=mesh_size
            )
            return struct_map_fig

        @app.callback(
            Output('structure-map', 'figure'),
            [Input('generation-slider', 'value')]
        )
        def structure_map_callback(gen):
            _str_gen = '%03d' % gen

            struct_map_fig = update_gen_structure_map(prob_structs=prob_structs, gen=_str_gen, var_color_dict=var_color_dict)
            return struct_map_fig

        @app.callback(
            Output('probabilities-table-container', 'children'),
            [Input('generation-slider', 'value'),
             Input('variable-dropdown', 'value')]
        )
        def probabilities_table_callback(gen, variable):
            _str_gen = '%03d' % gen

            probabilities_table_fig = update_probabilities_table(probs=probs, gen=_str_gen, variable=variable)
            return probabilities_table_fig

        @app.callback(
            Output('fitness-plot', 'figure'),
            [Input('plot-dropdown', 'value'),
             Input('fitness-plot-checklist', 'value')]
        )
        def population_fitness_callback(plots, options):
            fitness_plot_fig = update_fitness_plot(logger_data=logger_data, plots=plots, options=options)
            return fitness_plot_fig

        @app.callback(
            Output('chromossome-heatmap', 'figure'),
            [Input('generation-slider', 'value')]
        )
        def population_heatmap_callback(gen):
            _str_gen = '%03d' % gen
            chromossome_heatmap_fig = update_chromossome_heatmap(characteristics_df=characteristics_df, gen=_str_gen)
            return chromossome_heatmap_fig

        app.run_server(debug=False)
    else:
        raise ValueError('either --experiment-path or --print must be set')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for ploting a plotly graph of the graphical model'
    )

    parser.add_argument(
        '--experiment-path', action='store', required=False, default=None,
        help='Path to folder with experiment metadata.'
    )

    parser.add_argument(
        '--print', action='store_true', required=False, default=False,
        help='If provided, will generate a .pdf file with the structure of first generation graphical model.'
    )

    main(parser.parse_args())
