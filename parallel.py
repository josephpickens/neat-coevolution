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
        eval_jobs = []
        for i in range(len(envs)):
            raw_score_dict = [[{} for _ in pops[i][0].population.values()] for _ in pops[i]]
            configs = [p.config for p in pops[i]]
            genome_pairs = self.pairing_function(pops[i])
            for genome_pair in genome_pairs:
                eval_jobs.append(self.pool.apply_async(self.eval_function, (envs[i], genome_pair, configs)))
            indices = {}
            last_indices = [0, 0]
            for job, genome_pair in zip(eval_jobs, genome_pairs):
                raw_scores = job.get(timeout=self.timeout)
                for j, score in enumerate(raw_scores):
                    other = (j + 1) % len(genome_pair)
                    if genome_pair[other] not in indices.keys():
                        indices[genome_pair[other]] = last_indices[other]
                        last_indices[other] += 1
                    index = indices[genome_pair[other]]
                    raw_score_dict[j][index][genome_pair[j]] = score
            for k, p in enumerate(pops[i]):
                genomes = p.population.values()
                rank, intralayer_rank, layers = self.ranking_function(genomes, raw_score_dict[k])
                counts = [len(l) for l in layers]
                for g in genomes:
                    g.fitness = intralayer_rank[g] / (counts[rank[g]] + 1) - rank[g]
