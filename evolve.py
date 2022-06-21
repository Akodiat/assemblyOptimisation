import oxpy
import networkx as nx
from genome import MultiGenome, ArrayGenome, NeatGenome
from geneticAlgoritm import GeneticAlgorithm
import random

class PatchySimGenome(MultiGenome):
    def __init__(self, nInteractions, maxTemp, minTemp):
        self.strengthsGenome = ArrayGenome([random.random() for _ in range(nInteractions)])
        self.tempGenome = NeatGenome(1,1, mutationWeights=[
            10, # Mutate connection weight,
            3,  # Remove connection,
            2,  # Add connection,
            3   # Add node
        ])
        self.maxTemp = maxTemp
        self.minTemp = minTemp
        super().__init__([
            self.strengthsGenome,
            self.tempGenome
        ])

    def clone(self):
        other = PatchySimGenome(0, self.maxTemp, self.minTemp)
        other.strengthsGenome = self.strengthsGenome.clone()
        other.tempGenome = self.tempGenome.clone()
        other.genomes = [other.strengthsGenome, other.tempGenome]
        other.age = self.age + 1
        return other

    def getTemperatures(self, nTempDivisions=10):
        outToTemp = lambda val: (self.maxTemp-self.minTemp)*val + self.minTemp
        return [outToTemp(self.tempGenome.evaluate([n/nTempDivisions])[0]) for n in range(nTempDivisions)]

    def getStrengths(self):
        return self.strengthsGenome.values

maxStep = 1e6
tempDivisions = 10
maxTemp = 0.1
minTemp = 0.005
nInteractions = 10
populationSize = 10

def getConnectionGraph(bonds):
    lines = bonds.splitlines()
    [step, nParticles] = [int(v) for v in lines[0].split()]
    print("Step {}, using {} particles".format(step, nParticles), flush=True)
    particleLines = lines[1::]
    G = nx.Graph()
    for i in range(nParticles):
        bondBools = particleLines[i*2].strip().split(' ')
        unbound = 0
        for j, patch in enumerate(bondBools):
            if int(patch) == 1:
                bondParticles = particleLines[i*2 + 1].strip().split(' ')
                G.add_edge(
                    i+1,
                    int(bondParticles[j-unbound]),
                    patch=j
                )
            else:
                unbound += 1
    return G

def annealingOxDNA(inputPath, stepsPerTemp, temps):
    with oxpy.Context():
        # init the manager with the given input file
        manager = oxpy.OxpyManager(inputPath)
        manager.load_options()
        manager.init()

        for temp in temps:
            #print("Setting temperature to {}".format(temp), flush=True)
            manager.update_temperature(temp)
            manager.run(stepsPerTemp, False)

        bondsObs = manager.config_info().get_observable_by_id("my_patchy_bonds")
        bonds = bondsObs.get_output_string(manager.current_step)

        G = getConnectionGraph(bonds)
        try:
            connectedParticles = len(max(nx.connected_components(G), key=len))
        except:
            connectedParticles = 0

        fitness = connectedParticles

        print("Fitness: {}".format(fitness), flush=True)

        return fitness

def fitnessFunc(genome):
    temps = genome.getTemperatures(tempDivisions)
    print("Network {}\n\ttemps = {}\n\tstrengths = {}".format(
        genome.tempGenome,
        ', '.join('{:.3f}'.format(v) for v in temps),
        ', '.join('{:.3f}'.format(v) for v in genome.getStrengths())
    ), flush=True)

    setInteractionStrengths(genome, 'LORO.interaction_matrix.template.txt')
    return annealingOxDNA(
        "input",
        int(maxStep/tempDivisions),
        temps
    )

def run():
    nInteractions = countInteractionStrengths('LORO.interaction_matrix.template.txt')

    # Initialize population
    population = [PatchySimGenome(nInteractions, maxTemp, minTemp) for _ in range(populationSize)]

    # Introduce some genetic diversity
    for i in population:
        i.mutate()

    # Initialise genetic algorithm
    evolver = GeneticAlgorithm(population, fitnessFunc)

    # Evolve for a given number of generations
    evolver.run(10)

def countInteractionStrengths(templatePath):
    with open(templatePath) as f:
        ls = f.readlines()
    return len(ls)

def setInteractionStrengths(genome, templatePath, outPath='LORO.interaction_matrix.txt'):
    with open(templatePath) as f:
        s = f.read()
    with open(outPath, 'w') as f:
        f.write(s.format(*genome.getStrengths()))

if __name__ == "__main__":
    run()
