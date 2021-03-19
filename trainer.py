#!/usr/bin/python
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'multiagent-particle-envs'))
import multiprocessing
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from itertools import zip_longest, product
from competitive_scenario import CompetitiveScenario
from cooperative_scenario import CooperativeScenario
from ecosystem import Ecosystem
from parallel import ParallelEvaluator
import neat
import visualize
from multiagent.environment import MultiAgentEnv
from pareto import ParetoRanker


def pair_all_vs_all(pops):
    genomes = []
    for p in pops:
        genomes.append(list(p.population.values()))
    return list(product(*genomes))


def pair_random_one_vs_one(pops):
    genomes = []
    for p in pops:
        temp = list(p.population.values())
        random.shuffle(temp)
        genomes.append(temp)
    return zip_longest(*genomes, fillvalue=genomes[0])  # needs better solution than fillvalue


# def pair_all_vs_some(pops, n):
#     random.shuffle(pops)
#     pairs = []
#     i = 0
#     j = n
#     while j <= len(pops[0]) and len(pops[0]) - j != 1:
#         pairs.extend(pair_all_vs_all(pops[:, i:j]))
#         i += n
#         j += n
#     pairs.extend(pair_all_vs_all(pops[i:]))
#     genomes = []
#     for p in pops:
#         temp = list(p.population.values())
#         random.shuffle(temp)
#         genomes.append(temp)
#     return genome_pairs


def fitness_function(envs, pops):
    for i in range(len(envs)):
        raw_score_dict = [[{} for _ in pops[i][0].population.values()] for _ in pops[i]]
        configs = [p.config for p in pops[i]]
        genome_pairs = pair_all_vs_all(pops[i])
        indices = {}
        last_indices = [0, 0]
        for genome_pair in genome_pairs:
            raw_scores = eval_genome_pair(envs[i], genome_pair, configs)
            for j, score in enumerate(raw_scores):
                other = (j + 1) % len(genome_pair)
                if genome_pair[other] not in indices.keys():
                    indices[genome_pair[other]] = last_indices[other]
                    last_indices[other] += 1
                index = indices[genome_pair[other]]
                raw_score_dict[j][index][genome_pair[j]] = score
        for p in pops[i]:
            genomes = p.population.values()
            rank, intralayer_rank, layers = pareto_ranking(genomes, raw_score_dict)
            counts = [len(l) for l in layers]
            for g in genomes:
                g.fitness = intralayer_rank[g] / (counts[rank[g]] + 1) - rank[g]


def eval_genome_pair(env, genome_pair, configs):
    nets = []
    for i, genome in enumerate(genome_pair):
        nets.append(neat.nn.FeedForwardNetwork.create(genome, configs[i]))
    # execution loop
    total_reward_n = [0] * len(env.world.agents)
    if len(env.world.agents) == 1:
        num_runs = 5
    else:
        num_runs = 1
    steps_per_run = 300
    for _ in range(num_runs):
        obs_n = env.reset()
        for _ in range(steps_per_run):
            # query for action from each agent's policy
            act_n = []
            for i, net in enumerate(nets):
                actions = net.activate(obs_n[i])
                actions = [0] + [1.0 if a >= 0.5 else 0 for a in actions]
                act_n.append(np.concatenate([actions, np.zeros(env.world.dim_c)]))
            # step environment
            obs_n, reward_n, _, _ = env.step(act_n)
            total_reward_n = [sum(x) for x in zip(total_reward_n, reward_n)]
    return total_reward_n


def pareto_ranking(genomes, raw_score_dict):
    ranker = ParetoRanker(genomes, raw_score_dict)
    return ranker.get_rank()


def run(ecosystem_type, num_gen, parallel=True, save_freq=None):
    ecosystem = build_ecosystem(ecosystem_type)
    if parallel:
        pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome_pair, pair_all_vs_all, pareto_ranking)
        ecosystem.run(ecosystem_type, pe.evaluate, n=num_gen, save_function=save_results, save_freq=save_freq)
    else:
        ecosystem.run(ecosystem_type, fitness_function, n=num_gen, save_function=save_results, save_freq=save_freq)


def build_ecosystem(ecosystem_type):
    if ecosystem_type == '1_agent':
        scenarios = [CooperativeScenario(num_agents=1)]
        populations = create_populations(1)
        assigned_pops = [populations]
    elif ecosystem_type == '2_competitive':
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


def save_results(ecosystem_type, best_genomes, generation, configs, stats):
    # current date and time in string format
    time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create new directory in which to save results
    path = 'results/%s/%s_%d' % (ecosystem_type, time, generation)
    os.mkdir(path)

    # Save results
    node_names = {0: 'right',
                  1: 'left',
                  2: 'forward',
                  3: 'backward',
                  -1: 'agent_E',
                  -2: 'landmark_E',
                  -3: 'agent_NE',
                  -4: 'landmark_NE',
                  -5: 'agent_N',
                  -6: 'landmark_N',
                  -7: 'agent_NW',
                  -8: 'landmark_NW',
                  -9: 'agent_W',
                  -10: 'landmark_W',
                  -11: 'agent_SW',
                  -12: 'landmark_SW',
                  -13: 'agent_S',
                  -14: 'landmark_S',
                  -15: 'agent_SE',
                  -16: 'landmark_SE'
                  }
    for i, genome in enumerate(best_genomes):
        with open(path + '/genome_%d' % (i+1), 'wb') as f:
            pickle.dump(genome, f)
        visualize.draw_net(configs[i], genome, filename=path + "/nn_%d.svg" % (i+1), node_names=node_names)
        configs[i].save(path + '/config')
        visualize.plot_stats(stats[i], ylog=True, filename=path + "/fitness%d.svg" % (i+1))
        visualize.plot_species(stats[i], filename=path + "/speciation%d.svg" % (i+1))


def play_winners(path):
    ecosystem_type = path.split('/')[1].strip()
    # Watch the winners play
    ecosystem = build_ecosystem(ecosystem_type)
    num_genomes = len(ecosystem.pops)
    winners = []
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, path + '/config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    for i in range(num_genomes):
        winner = pickle.load(open(path + '/genome_%d' % (i + 1), 'rb'))
        winners.append(winner)
    env_assign_map = {}
    for i, p in enumerate(ecosystem.pops):
        env_assign_map[p] = winners[i]
    nets = [[] for _ in ecosystem.envs]
    for i, env_pops in enumerate(ecosystem.assigned_pops):
        for p in env_pops:
            genome = env_assign_map[p]
            nets[i].append(neat.nn.FeedForwardNetwork.create(genome, config))
    # execution loop
    steps_per_run = 300
    while True:
        for i, env in enumerate(ecosystem.envs):
            env.render()
            obs_n = env.reset()
            for _ in range(steps_per_run):
                # query for action from each agent's policy
                act_n = []
                for j, net in enumerate(nets[i]):
                    actions = net.activate(obs_n[j])
                    actions = [0] + [1.0 if a >= 0.5 else 0 for a in actions]
                    act_n.append(np.concatenate([actions, np.zeros(env.world.dim_c)]))
                # step environment
                obs_n, _, _, _ = env.step(act_n)
                # render all agent views
                env.render()


def plot_num_net_connections(paths, num_genomes, gen_step, title):
    num_connections = [[] for _ in range(num_genomes)]
    for path in paths:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, path + '/config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        for i in range(num_genomes):
            genome = pickle.load(open(path + '/genome_%d' % (i + 1), 'rb'))
            num_connections[i].append(len([c for c in genome.connections.values() if c.enabled]))
    gen_nums = range(gen_step, gen_step * (len(num_connections[0]) + 1), gen_step)
    for i in range(num_genomes):
        plt.plot(gen_nums, num_connections[i], label=('Best genome from population %d' % (i + 1)))
    plt.xlabel('Generation')
    plt.ylabel('Number of Neural Net Connections')
    plt.title(title)
    plt.legend()
    plt.savefig(paths[-1] + '.svg')
    plt.show()


def test_action():
    scenario = CompetitiveScenario(num_pursuers=0, num_evaders=1)
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        done_callback=scenario.done)
    act_n = [np.concatenate([[1.0, 0, 0, 0, 0], np.zeros(world.dim_c)])]
    k = 0
    while k < 50:
        env.render()
        obs_n, reward_n, done_n, _ = env.step(act_n)
        k += 1


def test_pareto():
    population = ['a', 'b', 'c', 'd', 'e']
    objectives = [{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'b': 1, 'a': 2, 'e': 3, 'c': 4, 'd': 5}]
    pareto = ParetoRanker(population, objectives)
    rank = pareto.get_rank()
    print(pareto.ranked_solutions)


if __name__ == '__main__':
    # run('2_competitive', 1000, save_freq=50)
    play_winners('results/2_competitive/20210318_213301_50')
    # plot_num_net_connections(['results/2_cooperative/%s' % p for p in paths], 2, 50, 'Cooperative Game\n'
    #                                                                                  'Path: 20210307_103127_1000')
