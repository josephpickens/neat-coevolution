import os

import neat
from multiagent.environment import MultiAgentEnv
from competitive_scenario import CompetitiveScenario
from cooperative_scenario import CooperativeScenario
from ecosystem import Ecosystem
from simple_spread import SimpleSpreadScenario


def build_ecosystem(ecosystem_type):
    if ecosystem_type == '2_competitive':
        scenarios = [CompetitiveScenario()]
        populations = create_populations(2)
        assigned_pops = [populations]
    elif ecosystem_type == '2_cooperative':
        scenarios = [CooperativeScenario()]
        populations = create_populations(2)
        assigned_pops = [populations]
    elif ecosystem_type == '3_mixed':
        scenarios = [CompetitiveScenario(), CooperativeScenario()]
        populations = create_populations(3)
        assigned_pops = [populations[0:2], populations[1:3]]
    elif ecosystem_type == '2_spread':
        scenarios = [SimpleSpreadScenario(num_agents=2)]
        populations = create_populations(2)
        assigned_pops = [populations]

    else:
        raise RuntimeError("Invalid ecosystem type {}".format(ecosystem_type))
    environments = []
    for scenario in scenarios:
        world = scenario.make_world()
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
        environments.append(env)
    ecosystem = Ecosystem(environments, populations, assigned_pops)
    return ecosystem


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
