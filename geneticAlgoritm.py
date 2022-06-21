import random

class GeneticAlgorithm:
    def __init__(self, population, fitnessFunc, fitnessTarget=None):
        self.population = population
        self.fitnessFunc = fitnessFunc
        self.fitnessTarget = fitnessTarget

    def stepGeneration(self):
        fitnesses = [self.fitnessFunc(i) for i in self.population]

        print("Fitnesses: {}".format(fitnesses))

        maxFitness = max(fitnesses)
        if self.fitnessTarget and maxFitness >= self.fitnessTarget:
            print("Reached target fitness: {}".format(maxFitness))
            return

        # Repopulate with roulette wheel selection
        self.population = [g.clone() for g in random.choices(
            self.population,
            weights=fitnesses,
            k=len(self.population)
        )]
        for g in self.population:
            g.mutate()

        return maxFitness

    def run(self, nGenerations):
        for gen in range(nGenerations):
            maxFitness = self.stepGeneration()
            print("Generation {}, max fitness {}".format(gen, maxFitness))