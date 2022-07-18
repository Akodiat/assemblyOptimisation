import random

class GeneticAlgorithm:
    def __init__(self, population, fitnessFunc, fitnessTarget=None):
        self.population = population
        self.fitnessFunc = fitnessFunc
        self.fitnessTarget = fitnessTarget

    def stepGeneration(self):
        fitnesses = [self.fitnessFunc(i) for i in self.population]

        #print("Fitnesses: {}".format(fitnesses))

        maxFitness = max(fitnesses)

        best = self.population[fitnesses.index(maxFitness)]

        if self.fitnessTarget and maxFitness >= self.fitnessTarget:
            print("Reached target fitness: {}".format(maxFitness))
            return

        # Repopulate with roulette wheel selection
        if maxFitness != 0: # Cannot sample with invalid fitness
            self.population = [g.clone() for g in random.choices(
                self.population,
                weights=fitnesses,
                k=len(self.population)
            )]
        for g in self.population:
            g.mutate()

        return maxFitness, best

    def run(self, nGenerations):
        for gen in range(nGenerations):
            maxFitness, best = self.stepGeneration()
            print("Generation {}\n  Max fitness: {}\n  Best genome: {}".format(gen, maxFitness, best))
        return maxFitness, best