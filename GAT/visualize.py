import networkx as nx
import matplotlib.pyplot as plt
import biographs as bg
import numpy as np
import torch
from itertools import count

class visualize():
    def __init__(self, path, prot_seq, top5index, filename, weights):
        self.weights = weights
        self.file = path
        self.prot_seq = prot_seq
        self.top5 = top5index
        self.save_file_name = filename

    def _get_adjacency(self):
        edge_ind =[]
        molecule = bg.Pmolecule(self.file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        return m

    def _get_edgeindex(self):
        """
        It takes the adjacency matrix and returns the edge indices
        
        :param file: the path to the file containing the adjacency matrix
        :param adjacency_mat: the adjacency matrix of the graph
        :return: The edge_ind is a list of two tensors. The first tensor contains the row indices of the
        nonzero elements in the adjacency matrix. The second tensor contains the column indices of the
        nonzero elements in the adjacency matrix.
        """
        edge_ind = []
        m = self._get_adjacency()
        
        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        edge_ind.append(a)
        edge_ind.append(b)

        #normalized edge distance
        norm_m = (m - np.min(m)) / (np.max(m) - np.min(m))
        norm_m = np.squeeze(np.asarray(norm_m))

        # add comma to the list
        dist = list(norm_m[np.nonzero(norm_m)])
        dist = np.asarray(dist)
        dist = dist.transpose()

        print(torch.tensor(np.array(edge_ind), dtype= torch.long))
        # print(torch.tensor(np.array(dist)))
        return torch.tensor(np.array(edge_ind), dtype= torch.long)

    def _edge_index_to_tuples(self, edge_index):
        print(edge_index.shape[1])
        counter = edge_index.shape[1]
        list_edge_idx_tuple = []

        for i in range(counter):
            list_edge_idx_tuple.append((edge_index[0][i].item(), edge_index[1][i].item()))

        return list_edge_idx_tuple

    def _create_node_attr_dict(self, nodes):
        node_attr = {}
        print(nodes)
        print(self.weights)
        # exit()
        for count, i in enumerate(nodes):
            print(i)
            node_attr[i] = {"weights" : self.weights[i]}
            # print(count)
            # print(i)
            # exit()
        return node_attr

    def _color_node_base_on_weight(self, gr):
        # get unique groups
        groups = set(nx.get_node_attributes(gr,'weights').values())
        mapping = dict(zip(sorted(groups),count()))
        print(len(mapping))
        nodes = gr.nodes()
        colors = [mapping[gr.nodes[n]['weights']] for n in nodes]
        
        pos = nx.spring_layout(gr,seed = 100)
        ec = nx.draw_networkx_edges(gr, pos, alpha=0.2)
        nc = nx.draw_networkx_nodes(gr, pos, nodelist=nodes, node_color=colors, node_size=100, cmap=plt.cm.jet)
        plt.colorbar(nc)
        plt.axis('off')
        plt.savefig(self.save_file_name[:-4]+".png")
        plt.show()

    def _color_top_n_node(self, gr):
        color_map = []
        labels = {}
        for node in gr:
            if node in self.top5:
                print(node)
                aa = self.prot_seq[node]
                print(aa)
                color_map.append('red')
                labels[node] = aa
            else:
                color_map.append('gray')
        pos = nx.spring_layout(gr,seed = 100)
        nx.draw_networkx(gr, pos, node_color = color_map, labels=labels, node_size=50)

        if len(labels) != 10:
            print("Dict is empty!!!!")
            exit()
        plt.savefig(self.save_file_name[:-4]+".png")
        plt.show()

    def _show_graph_with_labels(self):
        # print(adjacency_matrix)
        # rows, cols = np.where(adjacency_matrix > 0)
        # edges = zip(rows.tolist(), cols.tolist())
        # gr = nx.Graph()
        # print(rows)
        # print(cols)
        # gr.add_edges_from(edges)
        print("seq_len")
        print(self.prot_seq)
        print(len(self.prot_seq))
        # keys = list(range(len(self.prot_seq)))
        # print(keys)
        # values = [*self.prot_seq]
        # print(values)
        # mapping = dict(zip(keys,values))
        
        # self.A = self._get_adjacency()
        # print(self.A)
        # exit()
        edge_index = self._get_edgeindex()
        list_edge_idx_tuple = self._edge_index_to_tuples(edge_index)

        gr = nx.from_edgelist(list_edge_idx_tuple)
        
        
        node_attr_dict = {}
        print("node_len")
        print(gr.nodes)
        print(len(gr.nodes))
        # node_attr = self._create_node_attr_dict(gr.nodes)

        # print(node_attr)
        # nx.set_node_attributes(gr, node_attr)

        # self._color_node_base_on_weight(gr)
        # self._color_top_n_node(gr)


        node_num = len(gr.nodes())
        edge_num = len(gr.edges())

        return node_num, edge_num
