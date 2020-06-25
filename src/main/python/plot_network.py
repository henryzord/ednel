# import matplotlib.pyplot as plt
import re

from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.cm import Pastel1
from matplotlib.colors import to_hex
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import argparse
import json
import os


def read_graph(json_path: str):
    _dict = json.load(open(json_path))

    structure_dict = dict()

    eq_splitter = lambda x: re.split('=(?![^(]*\))', x)

    for gen in _dict.keys():
        this_gen = dict()
        for variable in _dict[gen].keys():
            lines = list(_dict[gen][variable])

            parentnames, parentvals = zip(*list(map(eq_splitter, re.split(',(?![^(]*\))', lines[0]))))
            this_gen[variable] = list(set(parentnames) - {variable})
        structure_dict[gen] = this_gen

    return structure_dict


def local_plot(graphs, json_path):
    fig = go.Figure(
        layout=go.Layout(
            title='Dependency network structure throughout generations',
            xaxis=go.layout.XAxis(
                # range=(min(df['x']) - 2, max(df['x']) + 2),
                visible=False
            ),
            yaxis=go.layout.YAxis(
                # range=(min(df['y']) - 2, max(df['y']) + 2),
                visible=False
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )

    all_variables = list(graphs[list(graphs.keys())[0]].keys())
    families = list(zip(*map(lambda x: x.split('_'), all_variables)))[0]
    families_set = set(families)
    families_colors = dict(zip(
        families_set,
        map(lambda x: to_hex(Pastel1(x)), np.linspace(0, 1, num=len(families_set)))
    ))

    family_dict = dict(zip(all_variables, [families_colors[x] for x in families]))

    active = 0
    # adds projections
    for gen in sorted(graphs.keys()):
        G = nx.from_dict_of_lists(graphs[gen])
        pos = graphviz_layout(G, prog='neato')

        variable_names = list(pos.keys())
        x, y = zip(*list(pos.values()))

        node_list = G.nodes(data=True)
        edge_list = G.edges(data=False)

        node_colors = [family_dict[nd[0]] for nd in node_list]

        x_edges = []
        y_edges = []
        for a, b in edge_list:
            a_coord = pos[a]
            b_coord = pos[b]

            x_edges += [a_coord[0], b_coord[0], None]
            y_edges += [a_coord[1], b_coord[1], None]

        fig.add_trace(go.Scatter(
            x=x_edges,
            y=y_edges,
            visible=False,
            marker=dict(
                color='black',
            ),
            name='%03d edges' % int(gen)
        ))

        # node_labels = {node_name: node_attr['label'] for (node_name, node_attr) in node_list}
        # node_colors = [node_attr['color'] for (node_name, node_attr) in node_list]
        # node_edgecolors = [node_attr['edgecolor'] for (node_name, node_attr) in node_list]

        # TODO retrieve edges!

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
            )
        )

    # makes first scatter visible
    fig.data[active].visible = True
    fig.data[active + 1].visible = True

    steps = []
    for i in range(len(graphs.keys())):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][i * 2] = True  # Toggle i'th trace to "visible"
        step["args"][1][(i * 2) + 1] = True  # Toggle i'th trace to "visible"
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

    to_write_path = os.sep.join(json_path.split(os.sep)[:-1])
    plot(fig, filename=os.path.join(to_write_path, 'structures.html'))


def main(args):
    graphs = read_graph(args.json_path)
    local_plot(graphs, args.json_path)


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
