import neat.population
from neat.statistics import StatisticsReporter


class Ecosystem():
    def __init__(self, environments, populations, assigned_populations):
        self.envs = environments
        self.pops = populations
        self.assigned_pops = assigned_populations

    def run(self, fitness_function, n=None, save_function=None, save_freq=None, save_path=None):
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

            # save best genomes
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
                save_function(save_path, best_genomes, k, configs, stats)

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
        save_function(save_path, best_genomes, k, configs, stats)

