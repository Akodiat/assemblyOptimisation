import neat
import oxpy
import networkx as nx

maxStep = 1e7
tempDivisions = 10

maxTemp = 0.1
minTemp = 0.005

def getConnectionGraph(bonds):
    lines = bonds.splitlines()
    [step, nParticles] = [int(v) for v in lines[0].split()]
    print("Step {}, using {} particles".format(step, nParticles))
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
            print("Setting temperature to {}".format(temp))
            manager.update_temperature(temp)
            manager.run(stepsPerTemp)

        bondsObs = manager.config_info().get_observable_by_id("my_patchy_bonds")
        bonds = bondsObs.get_output_string(manager.current_step)

        G = getConnectionGraph(bonds)
        connectedParticles = len(max(nx.connected_components(G), key=len))

        fitness = connectedParticles

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
