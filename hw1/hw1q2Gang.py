import numpy as np
import snap
import matplotlib.pyplot as plt


def LoadGraph(pth='hw1-q2.graph'):
    """
    :param pth: data path
    :return: loaded graph g
    """
    g = snap.TUNGraph.Load(snap.TFIn(pth))
    return g

def ExtractBasicFeatures(node, graph):
    """
    :param node: node from snap.PUNGraph object.
    :param graph: snap.PUNGraph object representing an undirected graph
    :return: features: a list
    """
    d = node.GetDeg()
    features = list()
    features.append(d)
    num_edges = 0
    nbrs = []
    for i in range(d):
        nbrs.append(graph.GetNI(node.GetNbrNId(i)))
        num_edges += nbrs[-1].GetDeg()
    inner_edges = 0
    for i in range(d):
        for j in range(i):
            inner_edges += nbrs[i].IsInNId(nbrs[j].GetId())
    features.append(inner_edges)
    features.append(num_edges - 2 * inner_edges)
    return features

def CosSim(x, y):
    """
    :param x: vector 1 np.array
    :param y: vector 2 np.array
    :return: cosine similarity
    """
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 0.0
    else:
        return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

def AllFeatures(g):
    """
    :param g: snap.PUNGraph object representing an undirected graph
    :return: allfeature: feature list for all nodes
    """
    allfeatures = []
    for node in g.Nodes():
        allfeatures.append(ExtractBasicFeatures(node, g))
    return allfeatures

def q2_1():
    g = LoadGraph()
    features = ExtractBasicFeatures(g.GetNI(9), g)
    allfeatures = AllFeatures(g)

    features = np.array(features)
    allfeatures = np.array(allfeatures)
    cos_sim = [(i, CosSim(features, j)) for i, j in enumerate(allfeatures)]
    cos_sim.sort(key=lambda x: x[1], reverse=True)
    print("Feature vector of node 9 is:{}".format(features))
    print("Top 5 nodes are:{}".format(cos_sim[1:6]))

#execute q2_1
q2_1()
print('--------------------------------------------------------------------------')

def Aggregate(features, graph):
    """
    :param features: np.array(feature list)
    :param graph: snap.PUNGraph object representing an undirected graph
    :return: aggregated features
    """
    features_ = features.copy()
    n, f = features.shape
    features_mean = np.zeros((n, f))
    features_sum = np.zeros((n, f))
    for node in graph.Nodes():
        deg = node.GetDeg()
        for i in range(deg):
            features_sum[node.GetId(), :] += features_[node.GetNbrNId(i), :]
        features_mean[node.GetId(), :] = features_sum[node.GetId(), :] / deg if deg != 0 else 0
    return np.concatenate((features_, features_mean, features_sum), axis=1)

def q2_2():
    g = LoadGraph()
    allfeatures = np.array(AllFeatures(g))
    for k in range(2):
        allfeatures = Aggregate(allfeatures, g)
    features = allfeatures[9]

    cos_sim = [(i, CosSim(features, j)) for i, j in enumerate(allfeatures)]
    cos_sim.sort(key=lambda y: y[1], reverse=True)
    print("Feature vector of node 9 is: {}".format(features))
    print("Top 5 nodes are: {}".format(cos_sim[1:6]))

#execute q2_2
q2_2()
print('--------------------------------------------------------------------------')

def FindNode(lower, upper, data):
    """
    :param lower: lower bound of cosine similarity
    :param upper: upper bound of cosine similarity
    :param data: cosinse similarity between selected node and other nodes
    :return: the similarity random selected node
    """
    subset = [data[i] for i in range(len(data)) if upper >= data[i][1] >= lower]
    return np.random.permutation(len(subset))[0]

def GetSubgraph(node, graph, K=2):
    """
    :param node: node from snap.PUNGraph object.
    :param graph: snap.PUNGraph object
    :param K: repeat times
    :return: node set
    """
    node_set = set()
    node_set.add(int(node))
    cur_node = set()
    cur_node.add(int(node))

    for k in range(K):
        next_nbrs = set()
        for u in cur_node:
            u = graph.GetNI(u)
            for i in range(u.GetDeg()):
                node_set.add(u.GetNbrNId(i))
                next_nbrs.add(u.GetNbrNId(i))
        cur_node = next_nbrs.copy()
    V = snap.TIntV()
    for node in node_set:
        V.Add(node)
    return V

def DrawGraph(lower, upper, cos_sim, g, filename, title='', color='blue'):
    """
    output the graph
    """
    sub = GetSubgraph(FindNode(lower, upper, cos_sim), g)
    whole_graph = snap.TIntStrH()
    whole_graph[sub[0]] = color
    sub = snap.ConvertSubGraph(snap.PUNGraph, g, sub)
    snap.DrawGViz(sub, snap.gvlNeato, filename, title, True, whole_graph)

def q2_3():
    g = LoadGraph()
    allfeatures = np.array(AllFeatures(g))
    for k in range(2):
        allfeatures = Aggregate(allfeatures, g)
    features = allfeatures[9]

    cos_sim = [(i, CosSim(features, j)) for i, j in enumerate(allfeatures)]
    cos_sim.sort(key=lambda y: y[1], reverse=True)
    sim_nodes = [i[1] for i in cos_sim]
    plt.hist(sim_nodes, bins=20)
    plt.title("distribution of cosine similarity")
    plt.show()

    # groups = [(0, 0.05), (0.4, 0.45), (0.6, 0.65), (0.9, 0.95)]
    # for i, group in enumerate(groups):
    #     lower, upper = group
    #     DrawGraph(lower, upper, cos_sim, g, filename='graph'+str(lower)+str(upper)+'.png', title='', color='blue')

#execute q2_3
q2_3()