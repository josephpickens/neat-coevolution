import numpy as np
from itertools import zip_longest, product
import random
import operator

import neat
from pareto import ParetoRanker


def pareto_ranking(genomes, eval_score_dict):
    ranker = ParetoRanker(genomes, eval_score_dict)
    interlayer_ranks = ranker.rank_population()
    intralayer_ranks = ranker.rank_population_intralayer()
    overall_ranks = {}
    for g in genomes:
        overall_ranks[g] = -interlayer_ranks[g] + intralayer_ranks[g]
    best_genome = max(overall_ranks.items(), key=operator.itemgetter(1))[0]
    best_genome_raw_scores = [e[best_genome] for e in eval_score_dict]
    f = sum(best_genome_raw_scores) / len(best_genome_raw_scores)
    for g in genomes:
        overall_ranks[g] += f
    return overall_ranks


def avg_ranking(genomes, eval_score_dict):
    ranks = {}
    for g in genomes:
        ranks[g] = sum([e[g] for e in eval_score_dict]) / len(eval_score_dict)
    return ranks


def pair_all_vs_all(pops):
    genomes = []
    for p in pops:
        genomes.append(list(p.population.values()))
    return list(product(*genomes))


def pair_all_vs_some(pops):
    genomes = []
    for p in pops:
        genomes.append(list(p.population.values()))
    result = []
    for pop_genome in genomes:
        other_genomes = list(filter(lambda pg: pg != pop_genome, genomes))
        other_genomes = [item for sublist in other_genomes for item in sublist]
        for genome in pop_genome:
            samples = random.sample(other_genomes, 50)
            result += product(*[[genome], samples])

    return result


def pair_random_one_vs_one(pops):
    genomes = []
    for p in pops:
        temp = list(p.population.values())
        random.shuffle(temp)
        genomes.append(temp)
    return zip_longest(*genomes, fillvalue=genomes[0])  # needs better solution than fillvalue


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
        for j in range(steps_per_run):
            # query for action from each agent's policy
            act_n = []
            for i, net in enumerate(nets):
                activation = net.activate(obs_n[i])
                action_idx = np.argmax(activation)
                u = np.zeros(5)
                u[action_idx] = 1.0
                act_n.append(np.concatenate([u, np.zeros(env.world.dim_c)]))
            # step environment
            obs_n, reward_n, _, _ = env.step(act_n)
            total_reward_n = [sum(x) for x in zip(total_reward_n, reward_n)]
    return total_reward_n

'''
# parallel fitness function using Ray
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
        # evaluate genomes in parallel
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
                ref_to_genome[j][ref].fitness = pop_ranks[ref]'''


# sequential fitness function
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

        # evaluate genomes
        eval_scores = []
        for gp in genome_pairs:
            eval_scores.append(eval_genome_pair(envs[i], gp, configs))
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
        for j, pop in enumerate(zip(ref_to_genome, eval_score_dict)):
            rtg, esd = pop
            ranks = pareto_ranking(rtg.keys(), esd)
            for ref in ranks.keys():
                ref_to_genome[j][ref].fitness = ranks[ref]


def fitness_function_simple(envs, pops):
    for i in range(len(envs)):
        configs = []
        for j, p in enumerate(pops[i]):
            configs.append(p.config)
        genome_pairs = pair_all_vs_all(pops[i])
        for genome_pair in genome_pairs:
            scores = eval_genome_pair(envs[i], genome_pair, configs)
            if not genome_pair[0].fitness:
                genome_pair[0].fitness = 0
            if not genome_pair[1].fitness:
                genome_pair[1].fitness = 0
            genome_pair[0].fitness += scores[0]
            genome_pair[1].fitness += scores[1]
        for j, p in enumerate(pops[i]):
            genomes = p.population.values()
            for g in genomes:
                g.fitness /= float(len(genomes))


def genome_ref_map(genomes):
    genome_to_ref = {}
    ref_to_genome = {}
    for ref, g in enumerate(genomes):
        genome_to_ref[g] = str(ref)
        ref_to_genome[str(ref)] = g
    return genome_to_ref, ref_to_genome
