import os
import neat.population
from neat.statistics import StatisticsReporter
from cooperative_scenario import CooperativeScenario
from competitive_scenario import CompetitiveScenario
from multiagent.environment import MultiAgentEnv


class Ecosystem():
    def __init__(self, ecosystem_type):
        self.ecosystem_type = ecosystem_type
        self.envs = []
        if ecosystem_type == '1_agent':
            scenarios = [CooperativeScenario(num_agents=1)]
            self.pops = create_populations(1)
            self.assigned_pops = [self.pops]
        elif ecosystem_type == '2_competitive':
            scenarios = [CompetitiveScenario()]
            self.pops = create_populations(2)
            self.assigned_pops = [self.pops]
        elif ecosystem_type == '2_cooperative':
            scenarios = [CooperativeScenario()]
            self.pops = create_populations(2)
            self.assigned_pops = [self.pops]
        elif ecosystem_type == '3_mixed':
            scenarios = [CompetitiveScenario(), CooperativeScenario()]
            self.pops = create_populations(3)
            self.assigned_pops = [self.pops[0:2], self.pops[1:3]]
        else:
            raise RuntimeError("Invalid ecosystem type {}".format(ecosystem_type))
        for scenario in scenarios:
            world = scenario.make_world()
            env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                                done_callback=scenario.done)
            self.envs.append(env)

    def run(self, fitness_function, n=None, save_function=None, save_freq=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """

        # if self.config_1_agents.no_fitness_termination and (n is None):
        #     raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            for i, pop in enumerate(self.pops):
                print('Population %d:' % (i + 1))
                pop.reporters.start_generation(pop.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(self.envs, self.assigned_pops)

            # Gather and report statistics.
            best = [None] * len(self.pops)
            for i, pop in enumerate(self.pops):

                for g in pop.population.values():
                    if g.fitness is None:
                        raise RuntimeError("Fitness not assigned to genome {}".format(g.key))
                    if best[i] is None or g.fitness > best[i].fitness:
                        best[i] = g
                pop.reporters.post_evaluate(pop.config, pop.population, pop.species, best[i])

                # Track the best genome ever seen.
                if pop.best_genome is None or best[i].fitness > pop.best_genome.fitness:
                    pop.best_genome = best[i]

                if not pop.config.no_fitness_termination:
                    # End if the fitness threshold is reached.
                    fv = pop.fitness_criterion(g.fitness for g in pop.population.values())
                    if fv >= pop.config.fitness_threshold:
                        pop.reporters.found_solution(pop.config, pop.generation, best[i])
                        break

            # Create the next generation from the current generation.
            for pop in self.pops:
                pop.population = pop.reproduction.reproduce(pop.config, pop.species,
                                                            pop.config.pop_size, pop.generation)

                # Check for complete extinction.
                if not pop.species.species:
                    pop.reporters.complete_extinction()

                    # If requested by the user, create a completely new population,
                    # otherwise raise an exception.
                    if pop.config.reset_on_extinction:
                        pop.population = pop.reproduction.create_new(pop.config.genome_type,
                                                                     pop.config.genome_config,
                                                                     pop.config.pop_size)
                    else:
                        raise neat.population.CompleteExtinctionException()

                # Divide the new population into species.
                pop.species.speciate(pop.config, pop.population, pop.generation)

                pop.reporters.end_generation(pop.config, pop.population, pop.species)

                pop.generation += 1

            if save_freq is not None and k % save_freq == 0 and k != n:
                best_genomes = []
                configs = []
                stats = []
                for pop in self.pops:
                    best_genomes.append(pop.best_genome)
                    configs.append(pop.config)
                    for reporter in pop.reporters.reporters:
                        if isinstance(reporter, StatisticsReporter):
                            stats.append(reporter)
                save_function(self.ecosystem_type, best_genomes, k, configs, stats)

        best_genomes = []
        configs = []
        stats = []
        for pop in self.pops:
            if pop.config.no_fitness_termination:
                pop.reporters.found_solution(pop.config, pop.generation, pop.best_genome)
            best_genomes.append(pop.best_genome)
            configs.append(pop.config)
            for reporter in pop.reporters.reporters:
                if isinstance(reporter, StatisticsReporter):
                    stats.append(reporter)
        save_function(self.ecosystem_type, best_genomes, k, configs, stats)


def create_populations(num_pops):
    populations = []
    for _ in range(num_pops):
        # Load the config file, which is assumed to live in
        # the same directory as this script.
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        pop = neat.Population(config)
        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(neat.StdOutReporter(True))
        populations.append(pop)
    return populations
