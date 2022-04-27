import neat
import oxpy

maxStep = 1e6
tempDivisions = 10

maxTemp = 1
minTemp = 0.005

def annealingOxDNA(inputPath, stepsPerTemp, temps):
    with oxpy.Context():
        # init the manager with the given input file
        manager = oxpy.OxpyManager(inputPath)
        manager.load_options()
        manager.init()

        for temp in temps:
            print("Setting temperature to {}".format(temp))
            manager.update_temperature(temp)
            manager.run(int(stepsPerTemp))

        # Need something clever here...
        particles = manager.config_info().particles()
        fitness = manager.config_info().interaction.pair_interaction(particles[0], particles[1])
        print(fitness)

        return fitness

def outToTemp(val):
    return (maxTemp-minTemp)*val + minTemp

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        temps = [outToTemp(net.activate([n/tempDivisions])[0]) for n in range(tempDivisions)]
        print(temps)
        genome.fitness = annealingOxDNA("input", maxStep/tempDivisions, temps)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    run('config.txt')