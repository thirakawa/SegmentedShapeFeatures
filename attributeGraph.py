# -*- coding:utf-8 -*-

import numpy as np
import networkx as nx
from TreeOfShapes import *

# Graph ================================================
# constructing tree structure using NetworkX
def createAttributeGraph(tree1, im):
    num_x = im.shape[1]
    num_y = im.shape[0]
    g = nx.Graph()
    nodePixMap = np.zeros((im.shape[0]-2, im.shape[1]-2), dtype=int)
    g.graph['leaves'] = []

    # initialize tree
    for i in tree1.iterateFromRootToLeaves(False):
        # not root node
        if tree1.data[i] != -1:
            g.add_edge(i, tree1.data[i])
            g.node[i]['grayLevel'] = tree1.levels[i]
            g.node[i]['tag'] = False
            if 'children' in g.node[tree1.data[i]]:    # exist
                g.node[tree1.data[i]]['children'].append(i)
            else:                                 # not exist
                g.node[tree1.data[i]]['children'] = [i]

            if 'parent' in g.node[i]:
                g.node[i]['parent'].append(tree1.data[i])
            else:
                g.node[i]['parent'] = [tree1.data[i]]
        # root node
        else:
            g.add_node(i)
            g.node[i]['parent'] = None
            g.node[i]['grayLevel'] = tree1.levels[i]
            g.node[i]['tag'] = False

    # adding node attribution
    for i in tree1.iterateFromRootToLeaves(False):
        # in case of the node has children attribute (i.e. not leaf node)
        if 'children' in g.node[i]:
            g.node[i]['leaf'] = False
            g.node[i]['pixels'] = []

            # root
            if tree1.data[i] == -1:
                if len(g.node[i]['children']) > 1:
                    g.node[i]['isBranch'] = True
                else:
                    g.node[i]['isBranch'] = False
                g.node[i]['startPoint'] = True
                g.node[i]['unionIndex'] = i
                g.node[i]['connectIndex'] = i
                continue

            # case 1, 2, 5, 6
            if len(g.node[i]['children']) > 1:
                g.node[i]['isBranch'] = True
            else:
                g.node[i]['isBranch'] = False

            if g.node[tree1.data[i]]['isBranch']:
                g.node[i]['startPoint'] = True
            else:
                g.node[i]['startPoint'] = False

            if g.node[i]['isBranch'] and g.node[i]['startPoint']:
                g.node[i]['connectIndex'] = tree1.data[i]
            elif g.node[i]['isBranch'] and not g.node[i]['startPoint']:
                g.node[i]['connectIndex'] = g.node[tree1.data[i]]['unionIndex']
            elif not g.node[i]['isBranch'] and g.node[i]['startPoint']:
                g.node[i]['connectIndex'] = tree1.data[i]
                g.node[i]['unionIndex'] = i
            else:
                g.node[i]['unionIndex'] = g.node[tree1.data[i]]['unionIndex']

        # no children attribute (i.e. leaf node)
        else:
            g.node[i]['leaf'] = True
            g.graph['leaves'].append(i)
            g.node[i]['pixels'] = []

            g.node[i]['isBranch'] = False

            if g.node[tree1.data[i]]['isBranch']:
                g.node[i]['startPoint'] = True
                g.node[i]['connectIndex'] = tree1.data[i]
            else:
                g.node[i]['startPoint'] = False
                g.node[i]['unionIndex'] = g.node[tree1.data[i]]['unionIndex']

    # adding leaf
    for y in range(3,num_y*2-2, 2):
        for x in range(3, num_x*2-2, 2):
            g.node[ tree1.data[(2*num_x+1)*y + x] ]['pixels'].append([(x-1)/2-1, (y-1)/2-1])
            nodePixMap[(y-1)/2-1, (x-1)/2-1] = int(tree1.data[(2*num_x+1)*y + x])

    # adding number of nodes
    g.graph['n_nodes'] = ( max(g.nodes()) - min(g.nodes()) + 1 )


    return g, nodePixMap
# ======================================================


# Creating simplified Graph ============================
# reducing the number of nodes
def createSimplyfiedGraph(g, tree, nodePixMap):
    # create new graph
    sGraph = nx.Graph()
    newNPMap = nodePixMap.copy()

    for i in tree.iterateFromRootToLeaves(False):
        # NOT root node
        if tree[i] != -1:

            # leaf node
            if g.node[i]['leaf']:
                if g.node[i]['startPoint']:
                    sGraph.add_edge(i, g.node[i]['connectIndex'])
                    sGraph.node[i]['pixels'] = g.node[i]['pixels']
                    sGraph.node[g.node[i]['connectIndex']]['children'].append(i)
                    sGraph.node[i]['parent'] = g.node[i]['connectIndex']
                    sGraph.node[i]['leaf'] = True
                    sGraph.node[i]['subTree'] = [i]
                else:
                    sGraph.node[ g.node[i]['unionIndex'] ]['pixels'].extend(g.node[i]['pixels'])
                    newNPMap[newNPMap == i] = g.node[i]['unionIndex']
                    sGraph.node[ g.node[i]['unionIndex'] ]['leaf'] = True
                    sGraph.node[ g.node[i]['unionIndex'] ]['subTree'].append(i)

            # not leaf node
            else:
                if not g.node[i]['isBranch'] and not g.node[i]['startPoint']:
                    sGraph.node[ g.node[i]['unionIndex'] ]['pixels'].extend(g.node[i]['pixels'])
                    newNPMap[newNPMap == i] = g.node[i]['unionIndex']
                    sGraph.node[ g.node[i]['unionIndex'] ]['subTree'].append(i)

                elif g.node[i]['isBranch'] and not g.node[i]['startPoint']:
                    sGraph.add_edge(i, g.node[i]['connectIndex'])
                    sGraph.node[i]['pixels'] = g.node[i]['pixels']
                    sGraph.node[i]['children'] = []
                    sGraph.node[g.node[i]['connectIndex']]['children'].append(i)
                    sGraph.node[i]['parent'] = g.node[i]['connectIndex']
                    sGraph.node[i]['leaf'] = False
                    sGraph.node[i]['subTree'] = [i]

                elif not g.node[i]['isBranch'] and g.node[i]['startPoint']:
                    sGraph.add_edge(i, g.node[i]['connectIndex'])
                    sGraph.node[i]['pixels'] = g.node[i]['pixels']
                    sGraph.node[i]['children'] = []
                    sGraph.node[g.node[i]['connectIndex']]['children'].append(i)
                    sGraph.node[i]['parent'] = g.node[i]['connectIndex']
                    sGraph.node[i]['leaf'] = False
                    sGraph.node[i]['subTree'] = [i]

                else:
                    sGraph.add_edge(i, g.node[i]['connectIndex'])
                    sGraph.node[i]['pixels'] = g.node[i]['pixels']
                    sGraph.node[i]['children'] = []
                    sGraph.node[g.node[i]['connectIndex']]['children'].append(i)
                    sGraph.node[i]['parent'] = g.node[i]['connectIndex']
                    sGraph.node[i]['leaf'] = False
                    sGraph.node[i]['subTree'] = [i]

        # root node
        else:
            sGraph.add_node(i)
            sGraph.node[i]['parent'] = None
            sGraph.node[i]['children'] = []
            sGraph.node[i]['pixels'] = g.node[i]['pixels']
            sGraph.node[i]['leaf'] = False
            sGraph.node[i]['subTree'] = [i]

    return sGraph, newNPMap
