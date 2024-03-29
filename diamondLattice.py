import sys
import oxpy
import networkx as nx
from gapy.genome import MultiGenome, ArrayGenome, PolynomialGenome
from gapy.geneticAlgoritm import GeneticAlgorithm
import random
import pickle
import os
from distutils.dir_util import copy_tree, remove_tree
import subprocess

"""
study_boo is the program that outputs what type of crystals you have 
The second column is the id of the cluster (0 is the largest, 1 is the second largest and so on)
The first column is the type of crystal lattice surrounding the particle
1 is diamond
2 is hexagonal (we want to avoid)
4 or 6 are generally amorhpous/undefined neigborhoods (that usually happens to particles on the surface)
We want to have as many 1s as possible

"""
def studyBoo(simDir, op_range=2.5, maxNeighbours=16, threshold=0.75, connections=12):
    executablePath = simDir+'/study_boo'
    result = subprocess.run([
        executablePath, str(op_range), str(maxNeighbours),
        str(threshold), str(connections), simDir+"/last_conf.dat"
    ], capture_output=True)
    nOnes = 0
    for line in result.stdout.decode().split('\n'):
        vs = line.split()
        if len(vs)>0 and int(vs[0]) == 1:
            nOnes += 1
    return nOnes


class PatchySimGenome(MultiGenome):
    def __init__(self, nInteractions, initialTempGuess=None):
        self.strengthsGenome = ArrayGenome(
            [1 for _ in range(nInteractions)],
            minVal=0, maxVal=1
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

def replaceInFile(path, search, replace):
    with open(path, 'r+') as f:
        content = f.read()
        f.seek(0)
        f.truncate()
        f.write(content.replace(search, replace))


def annealingOxDNA(stepsPerTemp, temps, simDir):
    with oxpy.Context():
        # init the manager with the given input file
        my_input = oxpy.InputFile()
        my_input.init_from_filename(simDir+"/input")

        # Make sure we have correct paths
        for p in ["topology", "conf_file", "trajectory_file", "log_file", "lastconf_file", "DPS_interaction_matrix_file"]:
            my_input[p] = simDir+'/'+my_input[p]
        replaceInFile(my_input["topology"],
            "p_mytetra.dat", simDir+"/p_mytetra.dat"
        )

        manager = oxpy.OxpyManager(my_input)
        manager.load_options()
        manager.init()

        for temp in temps:
            #print("Setting temperature to {}".format(temp), flush=True)
            manager.update_temperature(temp)
            manager.run(stepsPerTemp, False)

        fitness = studyBoo(simDir)

        return fitness

genomeLog = []
def onGenerationStep(generation, maxFitness, bestGenome):
    genomeLog.append({
        'generation': generation,
        'fitness': maxFitness,
        'genome': bestGenome
    })
    print("\nGeneration {}\n  Max fitness: {}\n  Best genome: {}\n".format(
        generation, maxFitness, bestGenome))

def run(
    systemName,
    maxStep = 1e7,
    tempDivisions = 10,
    nInteractions = 4,
    populationSize = 8,
    nGenerations = 1000,
):
    nInteractions = countInteractionStrengths(os.path.join(
        'templates',
        systemName,
        'LORO.interaction_matrix.template.txt'
    ))

    # Define fitness function
    def fitnessFunc(genome):
        temps = genome.getTemperatures(tempDivisions)
        print("\nNetwork {}\n\ttemps = {}\n\tstrengths = {}".format(
            genome.tempGenome,
            ', '.join('{:.3f}'.format(v) for v in temps),
            ', '.join('{:.3f}'.format(v) for v in genome.getStrengths())
        ), flush=True)

        templateDir = os.path.join('templates', systemName)
        simDir = os.path.join(
            'simulations',
            f"{systemName}_{genome.__hash__()}")
        copy_tree(templateDir, simDir)

        setInteractionStrengths(genome,
            os.path.join(templateDir, 'LORO.interaction_matrix.template.txt'),
            os.path.join(simDir,'LORO.interaction_matrix.txt')
        )

        currentDir = os.getcwd()
        #os.chdir(simDir)
        fitness = annealingOxDNA(
            int(maxStep/tempDivisions),
            temps,
            simDir
        )
        #os.chdir(currentDir)
        remove_tree(simDir)
        return fitness

    # Initialize population with random constant temperature protocols
    population = [PatchySimGenome(nInteractions, random.uniform(0.01,0.1)) for _ in range(populationSize)]

    # Introduce some genetic diversity
    #for i in population:
    #    i.mutate()

    # Initialise genetic algorithm
    evolver = GeneticAlgorithm(population, fitnessFunc)

    # Evolve for a given number of generations
    evolver.run(nGenerations, onGenerationStep, nProcesses=8)

    # Save data
    with open('genomeLog.pickle', 'wb') as f:
        pickle.dump(genomeLog, f)


def countInteractionStrengths(templatePath):
    with open(templatePath) as f:
        ls = f.readlines()
    return len(ls)

def setInteractionStrengths(genome, templatePath, outPath):
    with open(templatePath) as f:
        s = f.read()
    with open(outPath, 'w') as f:
        f.write(s.format(*genome.getStrengths()))

if __name__ == "__main__":
    if len(sys.argv) < 0:
        systemName = sys.argv[0]
    else:
        systemName = 'diamond'
    run(systemName)
