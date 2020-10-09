import random
from collections import Collection, Iterable
from random import randrange

import numpy as np
from matplotlib import pyplot as plt

class node:
    def __init__(self, index: int, inputs = [None, None], weights = [None, None],
                 input_nodes=[None, None], output_nodes=[None, None]):
        self.weights = weights  # list of weights
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.inputs = inputs
        self.index = index

    def set_input(self, input, index):
        self.inputs.__setitem__(index, input)

    def change_weight(self, weight, index):
        self.weights[index] = weight

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_output(self, indices=None):
        wx = [n1*n2 for n1, n2 in zip(self.inputs, self.weights)]
        return self.sigmoid(sum(wx)) > 0

    def hasNext(self):
        return self.output_nodes is not None

    def feedForward(self, inputs):
        if self.hasNext():
            output = self.get_output(inputs) > 0
            for i in self.output_nodes:
                i.set_input(output, self.index)
        else:
            return self.get_output(inputs)

def listRand(lbound, ubound, count):
    rlist = []
    for i in range(count):
        rlist.append(random.random() *(ubound - lbound) + lbound)
    return rlist

def main():
    sensory_neurons = 2
    interneurons = 2
    outputs = 1
    f1P =[]
    f2P = []
    f1N = []
    f2N = []
    wrange = 0.1
    interNweights = [listRand(-wrange, wrange, 2) for i in range(interneurons)]
    outputWeights = [listRand(-wrange, wrange, 2) for i in range(outputs)]
    inputs = [listRand(-5, 5, 2) for i in range(1000)]
    for i,inp in enumerate(inputs):
        #random.seed(1)
        sensoryNs = [node(index=0, weights=[1], inputs=[inp[0]]), node(index=1, weights=[1], inputs=[inp[1]])]
        #interNs = [node(index=0, input_nodes=sensoryNs, weights=listRand(-wrange, wrange,2)),
         #      node(index=1, input_nodes=sensoryNs, weights=listRand(-wrange, wrange,2))]
        interNs = [node(index=ind, input_nodes=sensoryNs, weights=interNweights[ind]) for ind in range(interneurons)]
        outputNs = node(index=0, weights=outputWeights[0], input_nodes=interNs)
        for i, e in enumerate(interNs):
            for ii, ee in enumerate(sensoryNs):
                e.set_input(ee.get_output(), ee.index)
        for i, e in enumerate(interNs):
            outputNs.set_input(e.get_output(), e.index)
        if outputNs.get_output():
            f1P.append(inp[0])
            f2P.append(inp[1])
        else:
            f1N.append(inp[0])
            f2N.append(inp[1])
    plt.scatter(f1P, f2P, color = 'red')
    plt.scatter(f1N, f2N, color = 'blue')
    plt.show()
    #show the outputs
if __name__ == "__main__":
    main()
