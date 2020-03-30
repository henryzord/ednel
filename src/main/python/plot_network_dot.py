import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.cm import viridis
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import pandas as pd


def rec_read_path(path, contents):
    files = os.listdir(path)
    for file in files:
        my_name = file.split('.')[0]

        if file.split('.')[-1] == 'csv':
            df = pd.read_csv(os.path.join(path, file))
            parents = list(set(df.columns) - {'probability', my_name})
            try:
                contents[my_name] += parents
            except KeyError:
                contents[my_name] = parents
        elif os.path.isdir(os.path.join(path, file)):
            contents = rec_read_path(os.path.join(path, file), contents)
    return contents


def read_graph(distributions_path):
    contents = rec_read_path(distributions_path, contents={})
    G = nx.from_dict_of_lists(contents)  # type: nx.Graph
    return G


def local_plot(G, write_path=None):
    """
    Draw this individual.
    """
    # degrees = {}
    # shortest_paths = nx.shortest_path(graph, '0')
    # for node in graph.nodes:
    #     s_path = shortest_paths[node]
    #     d = 0
    #     for n in s_path:
    #         if 'color' not in graph.nodes[n]:
    #             d += 1
    #     degrees[node] = d

    # max_degree = max(degrees.values())

    # colors = list(map(to_hex, viridis(np.linspace(0, 1, num=max_degree * 2))))[(max_degree - 1):]

    # for node in graph.nodes:
    #     if 'color' not in graph.nodes[node]:
    #         graph.nodes[node]['color'] = colors[degrees[node]]

    fig, ax = plt.subplots(figsize=(16, 10))

    prog = 'dot' if os.name == 'nt' else 'sfdp'

    pos = graphviz_layout(G, prog=prog)
    nx.draw_networkx(G=G, pos=pos, with_labels=True)

    # node_list = G.nodes(data=True)
    # edge_list = graph.edges(data=True)
    #
    # node_labels = {node_name: node_attr['label'] for (node_name, node_attr) in node_list}
    # node_colors = [node_attr['color'] for (node_name, node_attr) in node_list]
    # node_edgecolors = [node_attr['edgecolor'] for (node_name, node_attr) in node_list]
    # edge_labels = {(x1, x2): d['threshold'] for x1, x2, d in edge_list}
    #
    # nx.draw_networkx_nodes(
    #     graph, pos, ax=ax, node_size=2200, node_color=node_colors, edgecolors=node_edgecolors, alpha=1
    # )  # nodes
    # nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=edge_list, style='solid', alpha=1)  # edges
    # nx.draw_networkx_labels(graph, pos, node_labels, ax=ax, font_size=8)  # node labels
    # nx.draw_networkx_edge_labels(digraph, pos, edge_labels=edge_labels, font_size=16)

    # box = ax.get_position()
    #
    # legend_elements = [Line2D([0], [0], marker='o', color='white', label='Value', markerfacecolor='#AAAAAA', markersize=15)] + \
    #                   [Line2D([0], [0], marker='o', color='black', label='PBIL', markerfacecolor=colors[0], markersize=15)] + \
    #                   [Line2D([0], [0], marker='o', color='black', label='Variable (level %#2.d)' % (i + 1), markerfacecolor=color, markersize=15) for i, color in enumerate(colors[1:])]
    #
    # ax.legend(handles=legend_elements, loc='lower right', fancybox=True, shadow=True, ncol=1)

    plt.axis('off')

    if write_path is not None:
        plt.savefig(write_path, format='pdf')
        plt.close()

    # plt.show()


def main(args):
    G = read_graph(args.distributions_path)
    local_plot(G, write_path=os.path.join(args.write_path, 'graphical_model.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='script for ploting a plotly graph of the graphical model'
    )

    parser.add_argument(
        '--distributions-path', action='store', required=True,
        help='Path to folder that contains joint distribution files. May have subdirectories (script will take care of it).'
    )

    parser.add_argument(
        '--write-path', action='store', required=True,
        help='Path write a .pdf file with the graphical model.'
    )

    args = parser.parse_args()

    main(args)
