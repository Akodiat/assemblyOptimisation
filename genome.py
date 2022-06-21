
import numpy as np
import random

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
    def __init__(self, values, mutationRate = 1):
        super().__init__()
        self.values = values
        self.mutationRate = mutationRate

    def clone(self):
        other = ArrayGenome(self.values[:], self.mutationRate)
        other.age = self.age + 1
        return other

    def mutate(self):
        if random.random() < self.mutationRate:
            for _ in range(max(self.mutationRate, 1)):
                i = random.randrange(len(self.values))
                self.values[i] += random.gauss(0, 0.5)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class NeatNode:
    def __init__(self, inputs=[], fAct=sigmoid):
        self.inputs = inputs
        self.fAct = fAct

    def getValue(self, passedNodes=[]):
        # Avoid recurrance by keeping track of passed nodes
        return self.fAct(sum(input['weight'] * input['node'].getValue(passedNodes=[self, *passedNodes]) for input in self.inputs if not input['node'] in passedNodes))

class NeatOutputNode(NeatNode):
    def __init__(self, inputs=[], fAct=sigmoid):
        super().__init__(inputs, fAct)
        self.prefix = 'o'


class NeatHiddenNode(NeatNode):
    def __init__(self, inputs=[], fAct=sigmoid):
        super().__init__(inputs, fAct)
        self.prefix = 'h'

class NeatInputNode(NeatNode):
    def __init__(self, fAct=sigmoid):
        self.fAct = fAct
        self.val = 0
        self.prefix = 'i'

    def setValue(self, val):
        self.val = val

    def getValue(self, passedNodes=[]):
        return self.fAct(self.val)

class BiasNode(NeatInputNode):
    def __init__(self):
        self.prefix = 'b'
        return
    def getValue(self, passedNodes=[]):
        return 1

class NeatGenome(Genome):
    def __init__(self, nInputs, nOutputs, mutationWeights=[
        10, # Mutate connection weight,
        3,  # Remove connection,
        2,  # Add connection,
        3   # Add node
    ], mutationRate = 1):
        super().__init__()
        self.inputs = [NeatInputNode() for _ in range(nInputs)]
        self.bias = BiasNode()
        self.hidden = []
        self.outputs = [NeatOutputNode(
            # Make fully connected
            inputs=[{'weight': 1, 'node': node} for node in [*self.inputs, self.bias]]
        ) for i in range(nOutputs)]
        self.mutationWeights = mutationWeights
        self.mutationRate = mutationRate

    def clone(self):
        # Create new genome with correct number of nodes
        other = NeatGenome(
            len(self.inputs), len(self.outputs),
            self.mutationWeights, self.mutationRate
        )
        other.hidden = [NeatHiddenNode(fAct=n.fAct) for n in self.hidden]

        # Map old nodes to new ones
        oldInNodes = [*self.hidden, *self.inputs, self.bias]
        newInNodes = [*other.hidden, *other.inputs, other.bias]
        assert len(oldInNodes) == len(newInNodes)
        oldOutNodes = [*self.outputs, *self.hidden]
        newOutNodes = [*other.outputs, *other.hidden]
        assert len(oldOutNodes) == len(newOutNodes)

        # Clear new connectivity
        for n in newOutNodes:
            n.inputs = []

        # Clone connectivity from old genome
        for i, n in enumerate(oldOutNodes):
            # Copy activation function
            newOutNodes[i].fAct = n.fAct
            # Create input connections with old
            # connectivity and weights
            newOutNodes[i].inputs = [{
                'weight': oldInput['weight'],
                'node': newInNodes[oldInNodes.index(oldInput['node'])]
            } for oldInput in n.inputs]
        for i, n in enumerate(self.inputs):
            other.inputs[i].fAct = n.fAct
            other.inputs[i].val  = n.val

        other.age = self.age + 1

        return other

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
        print("Adding connection")
        allInput = [*self.inputs, *self.hidden, self.bias]
        allOutput = [*self.hidden, *self.outputs]

        input = random.choice(allInput)
        output = random.choice(allOutput)

        output.inputs.append({'weight': 0, 'node': input})

    def mutateRemoveConnection(self):
        c = self.getRandomConnection()
        if not c:
            return

        print("Removing connection {}-{}-{}".format(
            self.labelNode(c['input']['node']),
            c['input']['weight'],
            self.labelNode(c['output']))
        )

        # Remove connection
        c['output'].inputs.remove(c['input'])

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

    def mutateConnection(self):
        print("Mutating connection strength")
        c = self.getRandomConnection()
        if not c:
            return
        c['input']['weight'] += random.gauss(0, 0.1)

    def mutateAddNode(self):
        print("Adding node")
        # Add a node in place of an existing connection
        c = self.getRandomConnection()
        if not c:
            return
        newNode = NeatHiddenNode(
            inputs = [c['input']] # Keep same weight
            # Add random activation function here?
        )
        i = c['output'].inputs.index(c['input'])
        c['output'].inputs[i] = {
            'weight': 1, # Use unit weight on second connection
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

    def labelNode(self, n):
        nodes = [*self.inputs, self.bias, *self.outputs, *self.hidden]
        return n.prefix+chr(ord('@')+nodes.index(n)+1)

    def __str__(self) -> str:
        return ', '.join("{}~{}~{}".format(
            self.labelNode(c['input']['node']),
            c['input']['weight'],
            self.labelNode(c['output'])
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

    g2 = ArrayGenome([1,1,1,1,1])
    for i in range(50):
        g2.mutate()
        print(g2.values, g2.age)