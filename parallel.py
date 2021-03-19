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
            genome_to_ref = [{} for _ in pops[i]]
            ref_to_genome = [{} for _ in pops[i]]
            eval_score_dict = []
            configs = []
            for j, p in enumerate(pops[i]):
                genomes = p.population.values()
                eval_score_dict.append([{} for _ in genomes])
                configs.append(p.config)
                for ref, g in enumerate(genomes):
                    genome_to_ref[j][g] = str(ref)
                    ref_to_genome[j][str(ref)] = g
            genome_pairs = self.pairing_function(pops[i])
            for genome_pair in genome_pairs:
                eval_jobs.append(self.pool.apply_async(self.eval_function,
                                                       (envs[i], genome_pair, configs)))
            indices = {}
            last_index = [0, 0]
            for job, genome_pair in zip(eval_jobs, genome_pairs):
                eval_scores = job.get(timeout=self.timeout)
                for j, score in enumerate(eval_scores):
                    other_index = (j + 1) % len(genome_pair)
                    if genome_pair[other_index] not in indices.keys():
                        indices[genome_pair[other_index]] = last_index[other_index]
                        last_index[other_index] += 1
                    index = indices[genome_pair[other_index]]
                    eval_score_dict[j][index][genome_to_ref[j][genome_pair[j]]] = score
            rank_jobs = []
            for j in range(len(pops[i])):
                rank_jobs.append(self.pool.apply_async(self.ranking_function,
                                                       (ref_to_genome[j].keys(),
                                                        eval_score_dict[j])))
            for j, job in enumerate(rank_jobs):
                ranks = job.get(timeout=self.timeout)
                for ref in ranks.keys():
                    ref_to_genome[j][ref].fitness = ranks[ref]
