# import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
# from matplotlib.cm import viridis
# from matplotlib.colors import to_hex
# from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
import argparse
import json
import os
import itertools as it


def read_graph(json_path):
    _dict = json.load(open(json_path))

    keys = list(_dict.keys())
    variables = np.unique([x.split('var_')[-1] for x in keys])
    generations = map(int, np.unique([x.split('_var_')[0].split('gen_')[-1] for x in keys]))

    variables = np.unique(variables)

    local_networks = dict()
    for gen in generations:
        local_networks[gen] = dict()
        for variable in variables:
            local_networks[gen][variable] = _dict['gen_%03d_var_%s' % (gen, variable)]

    return local_networks


def get_colors(G):
    degrees = {}
    shortest_paths = nx.shortest_path(G, '0')
    for node in G.nodes:
        s_path = shortest_paths[node]
        d = 0
        for n in s_path:
            if 'color' not in G.nodes[n]:
                d += 1
        degrees[node] = d

    max_degree = max(degrees.values())

    colors = list(map(to_hex, viridis(np.linspace(0, 1, num=max_degree * 2))))[(max_degree - 1):]

    for node in G.nodes:
        if 'color' not in G.nodes[node]:
            G.nodes[node]['color'] = colors[degrees[node]]

    return G


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
            )
        )
    )

    # defines positions before plotting subplots
    G = nx.from_dict_of_lists(graphs[0])
    # pos = nx.circular_layout(G)
    pos = graphviz_layout(G, prog='dot')

    variable_names = list(pos.keys())
    x, y = zip(*list(pos.values()))

    active = 0
    # adds projections
    for gen in sorted(graphs.keys()):
        G = nx.from_dict_of_lists(graphs[gen])

        node_list = G.nodes(data=True)
        edge_list = G.edges(data=False)

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
            name='%03d edges' % gen
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
                    color='blue',
                    size=12,
                    # colorscale='Viridis',
                    # colorbar=dict(
                    #     title='Fitness'
                    # )
                ),
                hovertext=variable_names,
                name='%03d Variables' % gen
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

#
#     fig, ax = plt.subplots(figsize=(16, 10))
#
#     pos = graphviz_layout(graph, root='0', prog='sfdp')
#
#     nx.draw_networkx_nodes(
#         graph, pos, ax=ax, node_size=2200, node_color=node_colors, edgecolors=node_edgecolors, alpha=1
#     )  # nodes
#     nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edge_list, style='solid', alpha=1)  # edges
#     nx.draw_networkx_labels(graph, pos, node_labels, ax=ax, font_size=8)  # node labels
#     # nx.draw_networkx_edge_labels(digraph, pos, edge_labels=edge_labels, font_size=16)
#
#     box = ax.get_position()
#
#     legend_elements = [Line2D([0], [0], marker='o', color='white', label='Value', markerfacecolor='#AAAAAA', markersize=15)] + \
#                       [Line2D([0], [0], marker='o', color='black', label='EDNEL', markerfacecolor=colors[0], markersize=15)] + \
#                       [Line2D([0], [0], marker='o', color='black', label='Variable (level %#2.d)' % (i + 1), markerfacecolor=color, markersize=15) for i, color in enumerate(colors[1:])]
#
#     ax.legend(handles=legend_elements, loc='lower right', fancybox=True, shadow=True, ncol=1)
#
#     plt.axis('off')
#
#     if savepath is not None:
#         plt.savefig(savepath, format='pdf')
#         plt.close()
#
#     # plt.show()


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
