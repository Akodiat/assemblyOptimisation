
import math
from multiprocessing.dummy import Array
from optparse import Values
import random
import networkx as nx
import numpy as np

class Genome:
    def __init__(self):
        self.age = 0

    def clone():
        raise NotImplementedError("Clone not implemented in baseclass")

    def mutate():
        raise NotImplementedError("Mutate not implemented in baseclass")

class MultiGenome(Genome):
    def __init__(self, genomes):
        super().__init__()
        self.genomes = genomes

    def clone(self):
        other = MultiGenome([g.clone() for g in self.genomes])
        other.age = self.age + 1
        return other

    def mutate(self):
        for g in self.genomes:
            g.mutate()

class ArrayGenome(Genome):
    def __init__(self, values, mutationRate = 1, minVal=None, maxVal=None, mutationDeviation = 0.5):
        super().__init__()
        self.values = values
        self.mutationRate = mutationRate
        self.minVal = minVal
        self.maxVal = maxVal
        self.mutationDeviation = mutationDeviation

    def clone(self):
        other = ArrayGenome(
            self.values[:],
            self.mutationRate,
            self.minVal,
            self.maxVal,
            self.mutationDeviation
        )
        other.age = self.age + 1
        return other

    def mutate(self):
        if random.random() < self.mutationRate:
            for _ in range(max(self.mutationRate, 1)):
                i = random.randrange(len(self.values))
                self.values[i] += random.gauss(0, self.mutationDeviation)
                if self.minVal is not None:
                    self.values[i] = max(self.minVal, self.values[i])
                if self.maxVal is not None:
                    self.values[i] = min(self.maxVal, self.values[i])

    def __str__(self) -> str:
        return str(self.values)


class PolynomialGenome(ArrayGenome):
    # a(x - e)^3 + b(x - e)^2 + c(x - e) + d
    def __init__(self, mutationRate=1, mutationDeviation = 0.01):
        values = [random.gauss(0, 1) for _ in range(5)]
        super().__init__(values, mutationRate, minVal=None, maxVal=None, mutationDeviation=mutationDeviation)

    def evaluate(self, x):
        [a,b,c,d,e] = self.values
        return a*(x - e)**3 + b*(x - e)**2 + c*(x - e) + d

    def clone(self):
        other = PolynomialGenome(self.mutationRate, self.mutationDeviation)
        other.values = self.values[:]
        other.age = self.age + 1
        return other

    def __str__(self) -> str:
        [a,b,c,d,e] = self.values
        return f'{a}(x - {e})^3 + {b}(x - {e})^2 + {c}(x - {e}) + {d}'


# Activation functions:

fActs = ['lin', 'sigm', 'log', 'tanh', 'relu', 'sq']
def activate(x, fAct='lin'):
    if fAct == 'lin': # Linear
        return x
    elif fAct == 'sigm': # Sigmoid
        return 1/(1 + math.exp(-x))
    elif fAct == 'log':
        return math.log(abs(x)+1)
    elif fAct == 'sin':
        return math.sin(x)
    elif fAct == 'tanh':
        return math.tanh(x)
    elif fAct == 'sq':
        return x**2
    elif fAct == 'relu':
        return x if x>0 else 0

    raise ValueError("Unknown activation function: '{}'".format(fAct))

class NeatNode:
    def __init__(self, inputs=[], fAct='sigm'):
        self.inputs = inputs
        self.fAct = fAct
        self.val = 0

    def getValue(self, passedNodes=[]):
        # Avoid recurrance by keeping track of passed nodes
        return activate(
            sum(input['weight'] * input['node'].getValue(passedNodes=[self, *passedNodes]) for input in self.inputs if not input['node'] in passedNodes),
            self.fAct
        )

    def updateValFromInput(self):
        self.val = activate(
            sum(input['weight'] * input['node'].val for input in self.inputs),
            self.fAct
        )

class NeatOutputNode(NeatNode):
    def __init__(self, inputs=[], fAct='lin'):
        super().__init__(inputs, fAct)
        self.prefix = 'o'


class NeatHiddenNode(NeatNode):
    def __init__(self, inputs=[], fAct=None):
        if fAct == None:
            fAct = random.choice(fActs)
        super().__init__(inputs, fAct)
        self.prefix = 'h'

class NeatInputNode(NeatNode):
    def __init__(self, fAct='lin'):
        self.fAct = fAct
        self.val = 0
        self.prefix = 'i'

    def setValue(self, val):
        self.val = val

    def getValue(self, passedNodes=[]):
        return activate(self.val, self.fAct)

    def updateValFromInput(self):
        pass

class NeatBiasNode(NeatInputNode):
    def __init__(self):
        self.prefix = 'b'
        self.val = 1
        self.fAct = 'lin'
        return
    def getValue(self, passedNodes=[]):
        return 1

class NeatGenome(Genome):
    def __init__(self, nInputs, nOutputs, mutationWeights=[
        10, # Mutate connection weight,
        4,  # Remove connection,
        4,  # Add connection,
        1   # Add node
    ], mutationRate = 0.75):
        super().__init__()
        self.inputs = [NeatInputNode() for _ in range(nInputs)]
        self.bias = NeatBiasNode()
        self.hidden = []
        self.outputs = [NeatOutputNode(
            # Make fully connected
            inputs=[{'weight': random.uniform(-1,1), 'node': node} for node in [*self.inputs, self.bias]]
        ) for i in range(nOutputs)]
        self.mutationWeights = mutationWeights
        self.mutationRate = mutationRate

    def clone(self):

        # Stringify connections
        s = str(self)

        # Create new genome with correct number of nodes
        other = NeatGenome(
            len(self.inputs), len(self.outputs),
            self.mutationWeights, self.mutationRate
        )

        other.hidden = [NeatHiddenNode() for n in self.hidden]

        # Clear connectivity
        for n in [*other.outputs, *other.hidden]:
            n.inputs = []

        nodeLabelMap = {other.labelNode(n): n  for n in [
            *other.inputs, *other.outputs, other.bias, *other.hidden
        ]}

        if s != "":
            for connection in s.split(', '):
                iNodeStr, weightStr, oNodeStr = connection.split('~')
                oNodeStr, fAct = oNodeStr.split('_')
                for nStr in [iNodeStr, oNodeStr]:
                    if not nStr in nodeLabelMap:
                        raise ValueError("Mystery node "+nStr)
                iNode = nodeLabelMap.get(iNodeStr)
                oNode = nodeLabelMap.get(oNodeStr)
                oNode.fAct = fAct
                weight = float(weightStr)
                oNode.inputs.append({
                    'weight': weight,
                    'node': iNode
                })

        # assert str(other) == s, "Incorrect copy \n\t{}\n\t{}".format(s, str(other))
        assert len(other.hidden) <= len(self.hidden), "{} != {}".format(len(other.hidden), len(self.hidden))

        return other

    def getGraph(self):
        G = nx.Graph()
        for n in [*self.inputs, self.bias]:
            G.add_node(self.labelNode(n), layer=0, fAct=n.fAct)
        hiddenDivs = int(np.sqrt(len(self.hidden)))
        for n in self.hidden:
            G.add_node(self.labelNode(n), layer=1+random.choice(
                # Distribute nodes a bit
                range(hiddenDivs)
            ), fAct=n.fAct)
        for n in self.outputs:
            G.add_node(self.labelNode(n), layer=hiddenDivs+1, fAct=n.fAct)
        for c in self.allConnections():
            G.add_edge(
                self.labelNode(c['input']['node']), 
                self.labelNode(c['output']),
                weight=c['input']['weight']
            )
        return G

    def draw(self):
        G = self.getGraph()
        pos = nx.multipartite_layout(G, subset_key="layer",)

        nx.draw_networkx(G, pos, arrows=True)

    def allConnections(self):
        return ({
            'input': i,
            'output': o
        } for o in [
            *self.hidden,
            *self.outputs
        ] for i in o.inputs)

    def getRandomConnection(self):
        connections = list(self.allConnections())
        if connections:
            return random.choice(connections)
        else:
            return None

    def mutateAddConnection(self):
        #print("Adding connection")
        allInput = [*self.inputs, *self.hidden, self.bias]
        allOutput = [*self.hidden, *self.outputs]

        input = random.choice(allInput)
        output = random.choice(allOutput)

        output.inputs.append({'weight': 0.0, 'node': input})

    def mutateRemoveConnection(self):
        c = self.getRandomConnection()
        if not c:
            return

        #print("Removing connection {}-{}-{}".format(
        #    self.labelNode(c['input']['node']),
        #    c['input']['weight'],
        #    self.labelNode(c['output']))
        #)

        # Remove connection
        c['output'].inputs.remove(c['input'])

        """
        # Remove hidden nodes they are no longer connected
        if not c['output'].inputs and c['output'] in self.hidden:
            print("Also removing node {}".format(
                self.labelNode(c['output']))
            )
            self.hidden.remove(c['output'])

            # Remove any downstream connections
            for d in self.allConnections():
                if d['input']['node'] is c['output']:
                    d['output'].inputs.remove(d['input'])
        """

    def mutateConnection(self):
        #print("Mutating connection strength")
        c = self.getRandomConnection()
        if not c:
            return
        c['input']['weight'] += random.gauss(0, 0.1)

    def mutateAddNode(self):
        #print("Adding node")
        # Add a node in place of an existing connection
        c = self.getRandomConnection()
        if not c:
            return
        newNode = NeatHiddenNode(
            inputs = [c['input']], # Keep same weight
            fAct = random.choice(fActs) # Random activation function
        )
        # Replace connection
        i = c['output'].inputs.index(c['input'])
        c['output'].inputs[i] = {
            'weight': 1.0, # Use unit weight on second connection
            'node': newNode
        }
        self.hidden.append(newNode)

    def mutate(self):
        if random.random() < self.mutationRate:
            mutations = random.choices([
                self.mutateConnection,
                self.mutateRemoveConnection,
                self.mutateAddConnection,
                self.mutateAddNode,
            ], weights=self.mutationWeights, k=max(self.mutationRate, 1))
            for m in mutations:
                m()

    def evaluate(self, inputVals):
        for i, v in enumerate(inputVals):
            self.inputs[i].setValue(v)

        return [output.getValue() for output in self.outputs]

    def evaluate2(self, inputVals, nSteps=100):
        for i, v in enumerate(inputVals):
            self.inputs[i].setValue(v)
        for node in [*self.hidden, *self.outputs]:
            node.val = 0
        for i in range(nSteps):
            for node in [*self.hidden, *self.outputs]:
                node.updateValFromInput()

        return [output.val for output in self.outputs]

    def labelNode(self, n):
        nodes = [*self.inputs, self.bias, *self.outputs, *self.hidden]
        return (
            n.prefix +
            # chr(ord('@') + nodes.index(n)+1)
            str(nodes.index(n))
        )

    def __str__(self) -> str:
        return ', '.join("{}~{}~{}_{}".format(
            self.labelNode(c['input']['node']),
            c['input']['weight'],
            self.labelNode(c['output']),
            c['output'].fAct
        ) for c in self.allConnections())



def int2char(n):
    return chr(ord('@')+n)

if __name__ == "__main__":
    gs = [NeatGenome(1, 1)]
    print("{} connections, {} hidden nodes. {}".format(len(list(gs[0].allConnections())), len(gs[0].hidden), gs[0]))
    for i in range(50):
        g = gs[-1].clone()
        g.mutate()
        print("{} connections, {} hidden nodes. {} (age {})".format(len(list(g.allConnections())), len(g.hidden), g, g.age))
        gs.append(g)
    nVals = 10
    print([g.evaluate([i/nVals]) for i in range(nVals)])

    g2 = ArrayGenome([1,1,1,1,1], minVal=0)
    for i in range(50):
        g2.mutate()
        print(g2.values, g2.age)