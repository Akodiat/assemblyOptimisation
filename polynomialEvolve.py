import sys
import oxpy
import networkx as nx
from gapy.genome import MultiGenome, ArrayGenome, PolynomialGenome
from gapy.geneticAlgoritm import GeneticAlgorithm
import random
import pickle
import os
from distutils.dir_util import copy_tree, remove_tree
import time

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

def annealingOxDNA(stepsPerTemp, temps, targetClusterSize):
    with oxpy.Context():
        # init the manager with the given input file
        manager = oxpy.OxpyManager('input')
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

        if connectedParticles == targetClusterSize:
            fitness = float('inf')
        else:
            fitness = 1/abs(connectedParticles-targetClusterSize)

        print("Fitness: {} (largest cluster is {})".format(fitness, connectedParticles), flush=True)

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
    nInteractions = 10,
    populationSize = 10,
    nGenerations = 100,
    targetClusterSize = 60,
    nProcesses = 8
):
    nInteractions = countInteractionStrengths(os.path.join(
        'templates',
        systemName,
        'LORO.interaction_matrix.template.txt'
    ))

    # Define fitness function
    def fitnessFunc(genome):
        if nProcesses > 1:
            time.sleep(random.random()*nProcesses)
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
        os.chdir(simDir)
        fitness = annealingOxDNA(
            int(maxStep/tempDivisions),
            temps,
            targetClusterSize
        )
        os.chdir(currentDir)
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
    evolver.run(nGenerations, onGenerationStep, nProcesses=nProcesses)

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

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("systemName")
    parser.add_argument("targetClusterSize")
    parser.add_argument("nInteractions")
    parser.add_argument("--maxStep", default=1e7)
    parser.add_argument("--tempDivisions", default=10)
    parser.add_argument("--populationSize", default=100)
    parser.add_argument("--nGenerations", default=100)
    args = parser.parse_args()
    run(
        args.systemName,
        maxStep = int(args.maxStep),
        tempDivisions = int(args.tempDivisions),
        nInteractions = int(args.nInteractions),
        populationSize = int(args.populationSize),
        nGenerations = int(args.nGenerations),
        targetClusterSize = int(args.targetClusterSize)
    )
