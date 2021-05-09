from collections import defaultdict
from multiprocessing import Pool

"""
Runs evaluation functions in parallel subprocesses
to evaluate multiple genomes at once.
"""


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, pairing_function, ranking_function, timeout=None):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.pairing_function = pairing_function
        self.ranking_function = ranking_function
        self.timeout = timeout
        self.pool = Pool(num_workers)

    def __del__(self):
        self.pool.close() # should this be terminate?
        self.pool.join()

    def evaluate(self, envs, pops, assigned_pops):
        num_envs = defaultdict(int)
        for i in range(len(envs)):
            for p in assigned_pops[i]:
                num_envs[p] += 1
        for i in range(len(envs)):
            eval_jobs = []
            configs = []
            for j, p in enumerate(assigned_pops[i]):
                configs.append(p.config)
            genome_pairs = self.pairing_function(assigned_pops[i])
            for genome_pair in genome_pairs:
                eval_jobs.append(self.pool.apply_async(self.eval_function,
                                                       (envs[i], genome_pair, configs)))

            for job, genome_pair in zip(eval_jobs, genome_pairs):
                scores = job.get(timeout=self.timeout)
                if not genome_pair[0].fitness:
                    genome_pair[0].fitness = 0
                if not genome_pair[1].fitness:
                    genome_pair[1].fitness = 0
                genome_pair[0].fitness += scores[0]
                genome_pair[1].fitness += scores[1]
        for i, p in enumerate(pops):
            genomes = p.population.values()
            for g in genomes:
                g.fitness /= float(num_envs[p] * len(genomes))

    def pareto_evaluate(self, envs, pops):
        eval_jobs = []
        objectives = {}

        unique_pops = list(set([i for j in pops for i in j]))
        genome_ref = {}
        ref_genome = {}
        for p in unique_pops:
            genome_ref[p] = {}
            ref_genome[p] = {}
            genomes = p.population.values()
            for ref, g in enumerate(genomes):
                genome_ref[p][g] = str(ref)
                ref_genome[p][str(ref)] = g
            objectives[p] = [defaultdict(int) for _ in envs]

        for i in range(len(envs)):
            configs = []
            for j, p in enumerate(pops[i]):
                configs.append(p.config)
            genome_pairs = self.pairing_function(pops[i])
            for genome_pair in genome_pairs:
                eval_jobs.append(self.pool.apply_async(self.eval_function,
                                                       (envs[i], genome_pair, configs)))
            for job, gp in zip(eval_jobs, genome_pairs):
                scores = job.get(timeout=self.timeout)
                p0 = pops[i][0]
                p1 = pops[i][1]
                objectives[p0][i][genome_ref[p0][gp[0]]] += scores[0]
                objectives[p1][i][genome_ref[p1][gp[1]]] += scores[1]

        for i, p in enumerate(unique_pops):
            if len(objectives[p][0]) == 0:
                del objectives[p][0]
            elif len(objectives[p][1]) == 0:
                del objectives[p][1]
            if len(objectives[p]) < 2:
                genomes = p.population.values()
                for g in genomes:
                    g.fitness = objectives[p][0][genome_ref[p][g]] / float(len(genomes))
                    if i == 2:
                        print(g.fitness)
            else:
                ranks = self.ranking_function(ref_genome[p].keys(), objectives[p])
                for ref in ranks.keys():
                    ref_genome[p][ref].fitness = ranks[ref]
