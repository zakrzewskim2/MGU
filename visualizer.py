# %%
import joblib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Visualizer():
    def plot_training_history(self, filename = 'training_data.joblib'):
        training_info = joblib.load(filename)

        weight_history = training_info['weight_history']
        layer_lengths = training_info['layer_lengths']
        num_layers = len(layer_lengths)
        config = training_info['config']

        G = nx.Graph()

        # Init visualization nodes
        vis_mappings = []

        # # # Bias node
        # G.add_node(0, pos=(4, 0))
        node_num = 1

        for i in range(num_layers):
            new_nodes = range(node_num, node_num + layer_lengths[i])
            vis_mappings.append(np.array(new_nodes))
            for index, node in enumerate(new_nodes):
                G.add_node(node, pos=(4 * (i + 1), 2 * index + 2))
            node_num += len(new_nodes)


        def draw_graph(G, weight_matrices, with_colorbar=False):
            # Add temporary bias node
            G.add_node(0, pos=(4, 0))

            # Init visualization edges
            for i in range(num_layers - 1):
                if config.bias:
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
                    colors.append(weights[(node, 0)])
                elif node == 0:
                    # Skip temporary bias
                    continue
                else:
                    colors.append(0)

            # Remove temporary bias node
            G.remove_node(0)

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

    def plot_classification_dataset(self, x, y, classes, \
            show = False, save_path = None):
        plt.scatter(x, y, c = classes)
        if type(save_path) is str:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()

    def plot_regression_dataset(self, x, y, \
            show = False, save_path = None):
        plt.scatter(x, y)
        if type(save_path) is str:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()

    def plot_classification_result(self, clf, x, y, real_class, \
            margin = 1, grid_point_distance = 0.1, \
            show = False, save_path = None):
        x_min, x_max = np.min(x) - margin, np.max(x) + margin
        y_min, y_max = np.min(y) - margin, np.max(y) + margin
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_point_distance),
                            np.arange(y_min, y_max, grid_point_distance))

        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        plt.contourf(xx, yy, z, alpha=0.4)
        plt.scatter(x, y, c=real_class, s=20, edgecolor='k')
        if type(save_path) is str:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()

    def plot_regression_result(self, estimator, x, y, \
            show = False, save_path = None):
        x_min, x_max = np.min(x) - 1, np.max(x) + 1
        xx = np.arange(x_min, x_max, 0.1)

        yy = estimator.predict(xx.reshape(-1, 1))
        yy = yy.reshape(xx.shape)

        plt.scatter(x, y, s = 5, edgecolor='k')
        plt.plot(xx, yy, 'r--')
        if type(save_path) is str:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()

    def plot_train_test_error(self, estimator, \
            X_train, y_train, X_test, y_test, log_scale = True, \
            show = False, save_path = None):
        train_errors = estimator.error_by_iteration(X_train, y_train)
        test_errors = estimator.error_by_iteration(X_test, y_test)

        if log_scale:
            plt.yscale('log')
        plt.plot(train_errors)
        plt.plot(test_errors)
        if type(save_path) is str:
            plt.savefig(save_path, dpi = 300)
        if show:
            plt.show()
