import numpy as np


class LouvainCommunityDetection(object):
    def __init__(self, adj, y=None, node_list=None):
        self.adj = adj
        self.num_nodes, self.num_edges, self.degree = self.graph_attribute()

        self.y = y if y else [*range(self.num_nodes)]
        self.node_list = node_list if node_list else [[i] for i in range(self.num_nodes)]
        # list of original node ind after clustering

    def graph_attribute(self):
        adj = self.adj
        num_edges = np.sum(adj)
        num_nodes = adj.shape[0]
        degree = np.sum(adj, axis=0)
        return num_nodes, num_edges, degree

    def modularity_graph(self, y):
        num_nodes, num_edges, degree, adj = self.num_nodes, self.num_edges, self.degree, self.adj
        Q = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                Q += adj[i, j] - degree[i] * degree[j] / num_edges if y[i] == y[j] else 0
        return Q / num_edges

    def modularity_optimization(self, iter_times=3):
        num_nodes, y, adj = self.num_nodes, self.y, self.adj
        # phase one
        cur_y = y.copy()
        for _ in range(iter_times):
            for i in range(num_nodes):
                tem_y = cur_y.copy()
                for j in range(num_nodes):
                    if i != j and adj[i, j] != 0 and cur_y[i] != cur_y[j]:
                        tem_y[i] = cur_y[j]
                        dq = self.modularity_graph(tem_y) - self.modularity_graph(cur_y)
                        if dq > 0:
                            cur_y = tem_y.copy()
                        else:
                            tem_y = cur_y.copy()
        return cur_y

    def community_aggregation(self):
        # phase two
        num_nodes, y, adj = self.num_nodes, self.y, self.adj
        n_community = np.unique(np.sort(y))
        new_num_nodes = len(n_community)
        new_adj = np.zeros((new_num_nodes, new_num_nodes))
        communities = {j: [] for j in n_community}
        communities_order = {j: i for i, j in enumerate(n_community)}
        new_node_list = [[] for _ in range(new_num_nodes)]
        node_stack = []
        for i in range(num_nodes):
            communities[y[i]].append(i)
            for j in range(num_nodes):
                if y[i] == y[j] and adj[i, j] != 0:
                    new_adj[communities_order[y[i]], communities_order[y[i]]] += adj[i, j]
                    if j not in node_stack:
                        new_node_list[communities_order[y[i]]] += self.node_list[j]
                        node_stack.append(j)
                elif adj[i, j] != 0:
                    new_adj[communities_order[y[i]], communities_order[y[j]]] += adj[i, j]

        return new_adj, communities, communities_order, new_node_list

    def recursive_pass(self, ntimes=1):
        result = []
        for i in range(ntimes):
            self.y = self.modularity_optimization(iter_times=10)
            M = self.modularity_graph(self.y)
            new_adj, communities, communities_order, new_node_list = self.community_aggregation()
            result.append({'A': new_adj, 'modularity': M, 'Community': communities, 'nodegroup': new_node_list})
            self.adj = new_adj
            self.y = np.array(list(communities_order.values()))
            self.node_list = new_node_list.copy()
            self.num_nodes, self.num_edges, self.degree = self.graph_attribute()
        return result


def clique_graph(num_cliques=4):
    clique = np.ones((4, 4)) - np.eye(4)
    num_nodes = num_cliques*4
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_cliques):
        adj[4 * i:4 * i + 4, 4 * i:4 * i + 4] = clique
        s = (4 * i + 1) % (4 * num_cliques)
        t = (4 * i + 5) % (4 * num_cliques)
        adj[s, t] = 1
        adj[t, s] = 1
    return adj


# execute q3_2
num_cliques = 4
G = clique_graph(num_cliques=num_cliques)
q32a = LouvainCommunityDetection(G)
result32a = q32a.recursive_pass(ntimes=5)
print('adjacency matrix of G is:\n{}\n the modularity is: {}\n'.format(result32a[0]['A'], result32a[0]['modularity']))
J = result32a[0]['A']
yj = [i for i in range(2) for _ in range(2)]
node_list = result32a[0]['nodegroup']
q32b = LouvainCommunityDetection(J,  yj, node_list)
result32b = q32b.recursive_pass(ntimes=5)
print('adjacency matrix of J is:\n{}\n the modularity is: {}\n'.format(result32b[0]['A'], result32b[0]['modularity']))


if __name__ == '__main__':
    # execute q3_3
    num_cliques = 32
    G_big = clique_graph(num_cliques=num_cliques)
    q33a = LouvainCommunityDetection(G_big)
    result33a = q33a.recursive_pass(ntimes=1)
    print('adjacency matrix of G_big is:\n{}\n the modularity is: {}\n'.format(result33a[0]['A'], result33a[0]['modularity']))
    J_big = result33a[0]['A']
    yjbig = [i for i in range(64) for _ in range(2)]
    node_list = result33a[0]['nodegroup']
    q33b = LouvainCommunityDetection(J_big,  yjbig, node_list)
    result33b = q33b.recursive_pass(ntimes=5)
    print('adjacency matrix of J_big is:\n{}\n the modularity is: {}\n'.format(result33b[0]['A'], result33b[0]['modularity']))
