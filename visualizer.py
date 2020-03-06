# %%
import joblib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

training_data = joblib.load('training_data.joblib')

weight_history = training_data['weight_history']
layer_lengths = training_data['layer_lengths']
num_layers = len(layer_lengths)
config = training_data['config']


G = nx.Graph()

# Init visualization nodes
vis_mappings = []

# Bias node
G.add_node(0, pos=(4, 0))
node_num = 1

for i in range(num_layers):
    new_nodes = range(node_num, node_num + layer_lengths[i])
    vis_mappings.append(np.array(new_nodes))
    for index, node in enumerate(new_nodes):
        G.add_node(node, pos=(4 * (i + 1), 2 * index + 2))
    node_num += len(new_nodes)


def draw_graph(G, weight_matrices, with_colorbar=False):
    # Init visualization edges
    for i in range(num_layers - 1):
        # Bias edges
        for in_node in range(layer_lengths[i + 1]):
            G.add_edge(0, vis_mappings[i + 1][in_node],
                       weight=round(weight_matrices[i][0, in_node], 2))

        # Next layer edges
        for out_node in range(1, layer_lengths[i] + 1):
            for in_node in range(layer_lengths[i + 1]):
                G.add_edge(vis_mappings[i][out_node - 1], vis_mappings[i + 1][in_node],
                           weight=round(weight_matrices[i][out_node, in_node], 2))

    # Draw graph
    pos = nx.get_node_attributes(G, 'pos')
    weights = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx(G, pos)
    # nx.draw(G, pos)

    nodes = G.nodes()
    colors = []
    weights = nx.get_edge_attributes(G, 'weight')
    for node in G.nodes():
        if node > layer_lengths[0]:
            colors.append(weights[(0, node)])
        else:
            colors.append(0)

    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
                                with_labels=True, node_size=100, cmap=plt.cm.viridis)

    if with_colorbar:
        plt.colorbar(nc)

    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=abs(
            edge[2]), edge_color='r' if edge[2] > 0 else 'b')

    # Uncomment to display edge labels
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels,
            font_size = 8)

    plt.pause(0.01)

plot_iteration_interval = 100

i = 0
draw_graph(G, weight_history[0], with_colorbar=True)
for weight_matrices in weight_history[1:]:
    i += 1
    if i % plot_iteration_interval == 0:
        plt.clf()
        draw_graph(G, weight_matrices, with_colorbar=True)

plt.show()


# %%
