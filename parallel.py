"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


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

    def evaluate(self, envs, pops):
        for i in range(len(envs)):
            genome_to_ref = []
            ref_to_genome = []
            eval_score_dict = []
            configs = []
            for j, p in enumerate(pops[i]):
                genomes = p.population.values()
                eval_score_dict.insert(0, [{} for _ in genomes])
                configs.append(p.config)
                gtr = {}
                rtg = {}
                for ref, g in enumerate(genomes):
                    gtr[g] = str(ref)
                    rtg[str(ref)] = g
                genome_to_ref.append(gtr)
                ref_to_genome.append(rtg)
            genome_pairs = self.pairing_function(pops[i])

            # evaluate genomes in parallel
            eval_jobs = [self.pool.apply_async(self.eval_function, (envs[i], gp, configs)) for gp in genome_pairs]
            genome_index = {}
            last_index = [0, 0]
            for job, gp in zip(eval_jobs, genome_pairs):
                eval_scores = job.get(timeout=self.timeout)
                for j, score in enumerate(eval_scores):
                    k = (j + 1) % len(gp)
                    if gp[k] not in genome_index.keys():
                        genome_index[gp[k]] = last_index[k]
                        last_index[k] += 1
                    index = genome_index[gp[k]]
                    eval_score_dict[j][index][genome_to_ref[j][gp[j]]] = score

            # pareto rank each genome
            rank_jobs = [self.pool.apply_async(self.ranking_function, (rtg.keys(), esd)) for (rtg, esd) in
                         zip(ref_to_genome, eval_score_dict)]
            for j, job in enumerate(rank_jobs):
                ranks = job.get(timeout=self.timeout)
                for ref in ranks.keys():
                    ref_to_genome[j][ref].fitness = ranks[ref]

    def evaluate_simple(self, envs, pops):
        eval_jobs = []
        for i in range(len(envs)):
            configs = []
            for j, p in enumerate(pops[i]):
                configs.append(p.config)
            genome_pairs = self.pairing_function(pops[i])
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
            for j, p in enumerate(pops[i]):
                genomes = p.population.values()
                for g in genomes:
                    g.fitness /= float(len(genomes))