################################################################################
# CS 224W (Fall 2019) - HW1
# Starter code for Question 1
# Last Updated: Sep 25, 2019
################################################################################

import snap
import numpy as np
import matplotlib.pyplot as plt

# Setup
erdosRenyi = None
smallWorld = None
collabNet = None


# Problem 1.1
def genErdosRenyi(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Erdos-Renyi graph with N nodes and E edges
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.PUNGraph.New()
    for i in range(N):
        Graph.AddNode(i)
    nums = np.arange(N * (N - 1) / 2)
    nums = np.random.permutation(nums)
    edge_set = set()
    for idx in nums:
        if len(edge_set) < E:
            u = int(idx // N)
            v = int(idx % N)
            if (u, v) not in edge_set and (v, u) not in edge_set:
                edge_set.add((u, v))
                Graph.AddEdge(u, v)
        else:
            break
    ############################################################################
    return Graph


def genCircle(N=5242):
    """
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Circle graph with N nodes and N edges. Imagine the nodes form a
        circle and each node is connected to its two direct neighbors.
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.PUNGraph.New()
    for i in range(N):
        Graph.AddNode(i)
    for i in range(N):
        Graph.AddEdge(i, (i + 1) % N)
    ############################################################################
    return Graph


def connectNbrOfNbr(Graph, N=5242):
    """
    :param - Graph: snap.PUNGraph object representing a circle graph on N nodes
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Graph object with additional N edges added by connecting each node
        to the neighbors of its neighbors
    """
    ############################################################################
    # TODO: Your code here!
    for i in range(N):
        Graph.AddEdge(i, (i + 2) % N)
    ############################################################################
    return Graph


def connectRandomNodes(Graph, M=4000):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - M: number of edges to be added

    return type: snap.PUNGraph
    return: G.raph object with additional M edges added by connecting M randomly
        selected pairs of nodes not already connected
    """
    ############################################################################
    # TODO: Your code here!
    N = Graph.GetNodes()
    nums = np.arange(N * (N - 1) / 2)
    nums = np.random.permutation(nums)
    edge_set = set()
    for edge in Graph.Edges():
        edge_set.add((edge.GetSrcNId(), edge.GetDstNId()))
    E = M + len(edge_set)
    for idx in nums:
        if len(edge_set) < E:
            u = int(idx // N)
            v = int(idx % N)
            if (u, v) not in edge_set and (v, u) not in edge_set:
                edge_set.add((u, v))
                Graph.AddEdge(u, v)
    ############################################################################
    return Graph


def genSmallWorld(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Small-World graph with N nodes and E edges
    """
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 4000)
    return Graph


def loadCollabNet(path):
    """
    :param - path: path to edge list file

    return type: snap.PUNGraph
    return: Graph loaded from edge list at `path and self edges removed

    Do not forget to remove the self edges!
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.LoadEdgeList(snap.PUNGraph, path, 0, 1)
    for E in Graph.Edges():
        if E.GetSrcNId() == E.GetDstNId():
            Graph.DelEdge(E.GetSrcNId(), E.GetDstNId())
    ############################################################################
    return Graph


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    X, Y = [], []
    max_deg = 0
    for node in Graph.Nodes():
        max_deg = max(max_deg, node.GetDeg())
    X = list(range(max_deg + 1))
    Y = [0] * len(X)
    for node in Graph.Nodes():
        Y[node.GetDeg()] += 1
    ############################################################################
    return X, Y


def Q1_1():
    """
    Code for HW1 Q1.1
    """
    global erdosRenyi, smallWorld, collabNet
    erdosRenyi = genErdosRenyi(5242, 14484)
    smallWorld = genSmallWorld(5242, 14484)
    collabNet = loadCollabNet("ca-GrQc.txt")

    x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdosRenyi, y_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')

    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallWorld, y_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()


# Execute code for Q1.1
Q1_1()


# Problem 1.2 - Clustering Coefficient

def calcClusteringCoefficientSingleNode(Node, Graph):
    """
    :param - Node: node from snap.PUNGraph object. Graph.Nodes() will give an
                   iterable of nodes in a graph
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: local clustering coeffient of Node
    """
    ############################################################################
    # TODO: Your code here!
    C = 0.0
    deg = Node.GetDeg()
    if deg < 2:
        return C
    else:
        nbrs = []
        for i in range(deg):
            nbrs.append(Graph.GetNI(Node.GetNbrNId(i)))
        count_tnbr = 0
        for i in range(deg):
            for j in range(i):
                count_tnbr += nbrs[i].IsInNId(nbrs[j].GetId())
        C = 2.0 * count_tnbr/(deg * (deg - 1.0))
    ############################################################################
    return C

def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of Graph
    """
    ############################################################################
    # TODO: Your code here! If you filled out calcClusteringCoefficientSingleNode,
    #       you'll probably want to call it in a loop here
    C = 0.0
    for node in Graph.Nodes():
        C += calcClusteringCoefficientSingleNode(node, Graph)
    C /= Graph.GetNodes()

    ############################################################################
    return C

def Q1_2():
    """
    Code for Q1.2
    """
    C_erdosRenyi = calcClusteringCoefficient(erdosRenyi)
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)

    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)


# Execute code for Q1.2
Q1_2()
