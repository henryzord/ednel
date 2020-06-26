import re

from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.cm import Pastel1
from matplotlib.colors import to_hex
import networkx as nx
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot
import pandas as pd
import argparse
import json
import os


def slider_callback(gen: int, var: int, n_variables: int, n_gens: int):
    pass


def dropdown_callback(gen: int, var: int, n_variables: int, n_gens: int):
    pass


def read_graph(json_path: str):
    """
    Given a path to a json file, reads a JSON that encodes a Dependency Network structure (with probabilities)
    throughout an evolutionary process of EDNEL.
    """

    _dict = json.load(open(json_path))

    structure_dict = dict()
    probabilities_dict = dict()

    eq_splitter = lambda x: re.split('=(?![^(]*\))', x)
    co_splitter = lambda x: re.split(',(?![^(]*\))', lines[0])

    for gen in _dict.keys():
        this_gen_structrues = dict()
        this_gen_probabilities = dict()
        for variable in _dict[gen].keys():
            lines = list(_dict[gen][variable].keys())
            probs = list(_dict[gen][variable].values())

            table = []
            parentnames = None
            for i, line in enumerate(lines):
                splitted_lines = map(co_splitter, lines)
                for splitted_line in splitted_lines:
                    _vars, _vals = zip(*(map(eq_splitter, splitted_line)))
                    table += [list(_vals) + [probs[i]]]
                    parentnames = _vars

            this_gen_structrues[variable] = list(set(parentnames) - {variable})
            this_gen_probabilities[variable] = pd.DataFrame(table, columns=list(parentnames) + ['probability'])

        structure_dict[gen] = this_gen_structrues
        probabilities_dict[gen] = this_gen_probabilities

    return structure_dict, probabilities_dict


def get_colors(structs: dict):
    """
    Builds a dictionary assigning a color to each variable.
    """

    all_variables = list(structs[list(structs.keys())[0]].keys())
    families = list(zip(*map(lambda x: x.split('_'), all_variables)))[0]
    families_set = set(families)
    families_colors = dict(zip(
        families_set,
        map(lambda x: to_hex(Pastel1(x)), np.linspace(0, 1, num=len(families_set)))
    ))

    var_color_dict = dict(zip(all_variables, [families_colors[x] for x in families]))
    return var_color_dict


def add_gen_structure_map(structs, probs, fig):
    var_color_dict = get_colors(structs)

    # adds projections
    for gen in sorted(structs.keys()):
        G = nx.from_dict_of_lists(structs[gen])
        pos = graphviz_layout(G, prog='neato')

        variable_names = list(pos.keys())
        x, y = zip(*list(pos.values()))

        node_list = G.nodes(data=True)
        edge_list = G.edges(data=False)

        node_colors = [var_color_dict[nd[0]] for nd in node_list]

        x_edges = []
        y_edges = []
        for a, b in edge_list:
            a_coord = pos[a]
            b_coord = pos[b]

            x_edges += [a_coord[0], b_coord[0], None]
            y_edges += [a_coord[1], b_coord[1], None]

        fig.add_trace(
            go.Scatter(
                x=x_edges,
                y=y_edges,
                visible=False,
                marker=dict(
                    color='black',
                ),
                name='%03d edges' % int(gen)
            ),
            row=1, col=1
        )
        fig.data[-1].visible = False

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='markers',
                visible=False,
                marker=dict(
                    color=node_colors,
                    size=20,
                ),
                text=variable_names,
                hovertemplate='%{text}',
                name='%03d Variables' % int(gen)
            ),
            row=1, col=1
        )
        fig.data[-1].visible = False

        for var in probs[gen].keys():
            fig.add_trace(
                go.Table(
                    header=dict(values=list(probs[gen][var].columns)),
                    cells=dict(values=probs[gen][var].T)
                ),
                row=1, col=2
            )
            fig.data[-1].visible = False  # TODO testing

    return fig


def init_fig():
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'table'}]])

    fig.layout.title = 'Dependency network structure throughout generations'
    fig.layout.xaxis.visible = False
    fig.layout.yaxis.visible = False
    fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
    fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'

    return fig


def local_plot(json_path):
    # structs contain dictionary of past structures
    # probs contain dictionary of past probabilities
    structs, probs = read_graph(args.json_path)

    fig = init_fig()
    fig = add_gen_structure_map(structs=structs, probs=probs, fig=fig)

    active = 0
    n_variables = len(structs[list(structs.keys())[0]])

    steps = []
    for i in range(len(structs.keys())):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][(i * (2 + n_variables)) + 0] = True  # Toggle i'th trace to "visible"
        step["args"][1][(i * (2 + n_variables)) + 1] = True  # Toggle i'th trace to "visible"
        step["args"][1][(i * (2 + n_variables)) + 2] = True  # Toggle i'th trace to "visible"  # TODO will break once dropdown changes
        steps.append(step)

    # makes first scatter visible
    # TODo will break because of probabilities table; fix!
    fig.data[active + 0].visible = True
    fig.data[active + 1].visible = True
    fig.data[active + 2].visible = True

    sliders = [dict(
        active=active,
        currentvalue={"prefix": "Generation: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.update_layout(updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="None",
                     method="update",
                     args=[{"visible": [True, False, True, False]},
                           {"title": "Yahoo",
                            "annotations": []}]),
                dict(label="High",
                     method="update",
                     args=[{"visible": [True, True, False, False]},
                           {"title": "Yahoo High",
                            "annotations": high_annotations}]),
            ]),
        )
    ])

    to_write_path = os.sep.join(json_path.split(os.sep)[:-1])
    plot(fig, filename=os.path.join(to_write_path, 'structures.html'))


def main(args):
    local_plot(args.json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for ploting a plotly graph of the graphical model'
    )

    parser.add_argument(
        '--json-path', action='store', required=True,
        help='Path to .json that stores a dictionary, one entry for each variable in each generation.'
    )

    args = parser.parse_args()

    main(args)
