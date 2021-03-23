#!/usr/bin/python
import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import zip_longest, product
from competitive_scenario import CompetitiveScenario
from cooperative_scenario import CooperativeScenario
from ecosystem import Ecosystem
import neat
import visualize
from multiagent.environment import MultiAgentEnv
from pareto import ParetoRanker
import ray
ray.init()


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


def fitness_function(envs, pops):
    for i in range(len(envs)):
        genome_to_ref = []
        ref_to_genome = []
        eval_score_dict = []
        configs = []
        for j, p in enumerate(pops[i]):
            genomes = p.population.values()
            eval_score_dict.insert(0, [{} for _ in genomes])
            configs.append(p.config)
            gtr, rtg = genome_ref_map(genomes)
            genome_to_ref.append(gtr)
            ref_to_genome.append(rtg)
        genome_pairs = pair_all_vs_all(pops[i])
        # evaluate all genomes in parallel
        eval_jobs = [eval_genome_pair.remote(envs[i], gp, configs) for gp in genome_pairs]
        eval_scores = ray.get(eval_jobs)
        genome_index = {}
        last_index = [0, 0]
        for scores, gp in zip(eval_scores, genome_pairs):
            for j, score in enumerate(scores):
                k = (j + 1) % len(gp)
                if gp[k] not in genome_index.keys():
                    genome_index[gp[k]] = last_index[k]
                    last_index[k] += 1
                index = genome_index[gp[k]]
                eval_score_dict[j][index][genome_to_ref[j][gp[j]]] = score
        # pareto rank each genome
        rank_jobs = [pareto_ranking.remote(rtg.keys(), esd) for (rtg, esd) in zip(ref_to_genome, eval_score_dict)]
        all_ranks = ray.get(rank_jobs)
        for j, pop_ranks in enumerate(all_ranks):
            for ref in pop_ranks.keys():
                ref_to_genome[j][ref].fitness = pop_ranks[ref]


@ray.remote
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
    steps_per_run = 100
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


def genome_ref_map(genomes):
    genome_to_ref = {}
    ref_to_genome = {}
    for ref, g in enumerate(genomes):
        genome_to_ref[g] = str(ref)
        ref_to_genome[str(ref)] = g
    return genome_to_ref, ref_to_genome


@ray.remote
def pareto_ranking(genomes, eval_score_dict):
    ranker = ParetoRanker(genomes, eval_score_dict)
    interlayer_ranks = ranker.rank_population()
    intralayer_ranks = ranker.rank_population_intralayer()
    overall_ranks = {}
    for g in genomes:
        overall_ranks[g] = -interlayer_ranks[g] + intralayer_ranks[g]
    return overall_ranks


def run(ecosystem_type, num_gen, save_freq=None):
    ecosystem = Ecosystem(ecosystem_type)
    ecosystem.run(fitness_function, n=num_gen, save_function=save_results, save_freq=save_freq)


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
    ecosystem = Ecosystem(ecosystem_type)
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
    steps_per_run = 100
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


def plot_nn_connections(paths, num_genomes, gen_step, title):
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
    objectives = [{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'b': 1, 'a': 2, 'd': 3, 'c': 4, 'e': 5}]
    pareto = ParetoRanker(population, objectives)
    rank = pareto.rank_population()
    intralayer_rank = pareto.rank_population_intralayer()
    print(intralayer_rank)


if __name__ == '__main__':
    run('2_competitive', 1000, save_freq=50)
    # play_winners('results/3_mixed/20210313_072514_150')
    # plot_nn_connections(['results/2_cooperative/%s' % p for p in paths], 2, 50, 'Cooperative Game\n'
    #                                                                                  'Path: 20210307_103127_1000')
    # test_pareto()
