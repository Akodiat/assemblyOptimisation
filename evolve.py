import neat
import oxpy

maxStep = 1e6
tempDivisions = 10

maxTemp = 0.1
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
            manager.run(stepsPerTemp)

        # Need something clever here...
        bondsObs = manager.config_info().get_observable_by_id("my_patchy_bonds")
        bonds = bondsObs.get_output_string(manager.current_step)

        nBonds = sum(int(i) for v in bonds.splitlines()[1::2] for i in v.split())
        fitness = nBonds
        print("Fitness: {}".format(fitness))

        return fitness

def outToTemp(val):
    return (maxTemp-minTemp)*val + minTemp

def tempsFromGenome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return [outToTemp(net.activate([n/tempDivisions])[0]) for n in range(tempDivisions)]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        temps = tempsFromGenome(genome, config)
        print(temps)
        genome.fitness = annealingOxDNA("input", int(maxStep/tempDivisions), temps)

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
    print(tempsFromGenome(winner, config))

if __name__ == "__main__":
    run('config.txt')