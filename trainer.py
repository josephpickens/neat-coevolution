#!/usr/bin/python
import os
import multiprocessing
from datetime import datetime

from build import build_ecosystem
from fitness_function import fitness_function_simple, eval_genome_pair, pair_all_vs_all, pareto_ranking
from save_results import save_results_spread
from parallel import ParallelEvaluator


def run(ecosystem_type, num_gen, parallel=True, save_freq=None, test=False):
    # Create new directory in which to save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = 'results/%s/%s' % (ecosystem_type, timestamp)
    if not test:
        os.mkdir(path)

    ecosystem = build_ecosystem(ecosystem_type)
    if parallel:
        pe = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome_pair, pair_all_vs_all, pareto_ranking)
        ecosystem.run(pe.evaluate_simple,
                      n=num_gen,
                      save_function=save_results_spread,
                      save_freq=save_freq,
                      save_path=path)
    else:
        ecosystem.run(fitness_function_simple,
                      n=num_gen,
                      save_function=save_results_spread,
                      save_freq=save_freq,
                      save_path=path)


if __name__ == '__main__':
    run('2_spread', 200, save_freq=5, parallel=True)

