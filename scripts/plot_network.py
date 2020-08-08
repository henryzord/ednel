
# utilities
import argparse
import copy
import json
import re

# graphic libraries
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
# graph libraries
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# dash callback
from dash.dependencies import Input, Output
from matplotlib.cm import Pastel1
from matplotlib.colors import to_hex
from networkx.drawing.nx_agraph import graphviz_layout


def read_probabilistic_graphs(json_path: str, det_G):
    """
    Given a path to a json file, reads a JSON that encodes a Dependency Network structure (with probabilities)
    throughout an evolutionary process of EDNEL.
    """

    _dict = json.load(open(json_path))

    structure_dict = dict()
    probabilities_dict = dict()

    eq_splitter = lambda x: re.split('=(?![^(]*\))', x)
    co_splitter = lambda x: re.split(',(?![^(]*\))', x)

    for gen in _dict.keys():
        G = copy.deepcopy(det_G)
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
                G.add_edge(variable, parent, type='probabilistic')

            this_gen_probabilities[variable] = pd.DataFrame(table, columns=list(parentnames) + ['probability'])

        structure_dict[gen] = G
        probabilities_dict[gen] = this_gen_probabilities

    return structure_dict, probabilities_dict


def get_colors(all_variables: list):
    """
    Builds a dictionary assigning a color to each variable.
    """
    families = list(zip(*map(lambda x: x.split('_'), all_variables)))[0]
    families_set = set(families)
    families_colors = dict(zip(
        families_set,
        map(lambda x: to_hex(Pastel1(x)), np.linspace(0, 1, num=len(families_set)))
    ))

    var_color_dict = dict(zip(all_variables, [families_colors[x] for x in families]))
    return var_color_dict


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

        is_probabilistic = edge_properties['type'] == 'probabilistic'

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
            arrowcolor='grey' if is_probabilistic else 'black'
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
            name='%03d Variables' % int(gen)
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


def init_app(structure_map, probabilities_table, generation_slider, variable_dropdown):
    app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

    app.layout = html.Div([
        html.Div([
            structure_map,
            generation_slider
        ], className="six columns"),
        html.Div([
            variable_dropdown,
            probabilities_table
        ], className="six columns"),
    ], id='dash-container')

    return app


def read_deterministic_graph(deterministic_path=None):
    if deterministic_path is None:
        return nx.DiGraph()

    raw = json.load(open(deterministic_path, 'r'))
    G = nx.DiGraph()
    for variable, parents in raw.items():
        G.add_node(variable)
        for parent in list(parents.keys()):
            G.add_edge(variable, parent, type='deterministic')

    return G


def main(args):
    det_G = read_deterministic_graph(args.deterministic_path)
    prob_structs, probs = read_probabilistic_graphs(args.json_path, det_G=det_G)

    gens = sorted(list(probs.keys()))
    first_gen = gens[0]
    variables = sorted(list(probs[first_gen].keys()))
    any_var = variables[0]
    var_color_dict = get_colors(all_variables=variables)

    structure_map = add_gen_structure_map(prob_structs=prob_structs, gen=first_gen, var_color_dict=var_color_dict)
    probabilities_table = add_probabilities_table(probs=probs, gen=first_gen, variable=any_var)
    generation_slider = add_generation_slider(gens=gens)
    variable_dropdown = add_variable_dropdown(variables=variables)
    app = init_app(structure_map=structure_map, probabilities_table=probabilities_table, generation_slider=generation_slider, variable_dropdown=variable_dropdown)

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

    app.run_server(debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for ploting a plotly graph of the graphical model'
    )

    parser.add_argument(
        '--json-path', action='store', required=True,
        help='Path to .json file with all (probabilistic) dependencies between variables, one for each generation.'
    )

    parser.add_argument(
        '--deterministic-path', action='store', required=False, default=None,
        help='Path to .json file with all (deterministic) dependencies between variables.'
    )

    args = parser.parse_args()

    main(args)
