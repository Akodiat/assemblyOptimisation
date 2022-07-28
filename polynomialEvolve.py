import oxpy
import networkx as nx
from genome import MultiGenome, ArrayGenome, PolynomialGenome
from geneticAlgoritm import GeneticAlgorithm
import random
import pickle

class PatchySimGenome(MultiGenome):
    def __init__(self, nInteractions, initialTempGuess=None):
        self.strengthsGenome = ArrayGenome(
            [1 for _ in range(nInteractions)],
            minVal=0,
        )
        self.tempGenome = PolynomialGenome(mutationDeviation=0.01)
        if initialTempGuess:
            self.tempGenome.values = [0,0,0, initialTempGuess, 0]
        super().__init__([
            self.strengthsGenome,
            self.tempGenome
        ])

    def clone(self):
        other = PatchySimGenome(0)
        other.strengthsGenome = self.strengthsGenome.clone()
        other.tempGenome = self.tempGenome.clone()
        other.genomes = [other.strengthsGenome, other.tempGenome]
        other.age = self.age + 1
        return other

    def getTemperatures(self, nTempDivisions=10):
        return [max(
            # Avoid negative temperatures
            0, self.tempGenome.evaluate(n/nTempDivisions)
        ) for n in range(nTempDivisions)]

    def getStrengths(self):
        return self.strengthsGenome.values

    def __str__(self) -> str:
        return f"Temperature function: {self.tempGenome}\nStrengths: {self.strengthsGenome}"

maxStep = 1e6
tempDivisions = 10
maxTemp = 0.1
minTemp = 0.005
nInteractions = 10
populationSize = 5
nGenerations = 100

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
    print("\nNetwork {}\n\ttemps = {}\n\tstrengths = {}".format(
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

genomeLog = []
def onGenerationStep(generation, maxFitness, bestGenome):
    genomeLog.append({
        'generation': generation,
        'fitness': maxFitness,
        'genome': bestGenome
    })
    print("\nGeneration {}\n  Max fitness: {}\n  Best genome: {}\n".format(
        generation, maxFitness, bestGenome))

def run():
    nInteractions = countInteractionStrengths('LORO.interaction_matrix.template.txt')

    # Initialize population with random constant temperature protocols
    population = [PatchySimGenome(nInteractions, random.uniform(0.01,0.1)) for _ in range(populationSize)]

    # Introduce some genetic diversity
    #for i in population:
    #    i.mutate()

    # Initialise genetic algorithm
    evolver = GeneticAlgorithm(population, fitnessFunc)

    # Evolve for a given number of generations
    evolver.run(nGenerations, onGenerationStep)

    # Save data
    with open('genomeLog.pickle', 'wb') as f:
        pickle.dump(genomeLog, f)


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
