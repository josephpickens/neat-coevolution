import datetime
import os
import pickle
import string

import visualize


def save_results_spread(path, best_genomes, generation, configs, stats):
    path += '/' + str(generation)
    os.mkdir(path)

    # Save results
    node_names = {0: 'none',
                  1: 'right',
                  2: 'left',
                  3: 'forward',
                  4: 'backward',
                  -1: 'x-vel',
                  -2:  'y-vel',
                  -3:  'x-pos',
                  -4:  'y-pos',
                  -5:  'x-lm1-vec',
                  -6:  'y-lm1-vec',
                  -7:  'x-lm2-vec',
                  -8:  'y-lm2-vec',
                  -9:  'x-other-vec',
                  -10: 'y-other-vec',
                  -11: 'x-other-vel',
                  -12: 'y-other-vel'
                  }
    for i, genome in enumerate(best_genomes):
        with open(path + '/genome_%d' % (i+1), 'wb') as f:
            pickle.dump(genome, f)
        visualize.draw_net(configs[i], genome, filename=path + "/nn_%d.svg" % (i+1), node_names=node_names)
        configs[i].save(path + '/config')
        visualize.plot_stats(stats[i], ylog=True, filename=path + "/fitness%d.svg" % (i+1))
        visualize.plot_species(stats[i], filename=path + "/speciation%d.svg" % (i+1))


def save_results(path, best_genomes, generation, configs, stats):
    path += '/' + str(generation)
    os.mkdir(path)

    # node name dictionary
    num_obs_windows = 8
    obs_window_ids = list(string.ascii_uppercase)[0:num_obs_windows]
    entity_ids = ['other', 'lm']
    entity_state = [['x-pos', 'y-pos', 'x-vel', 'y-vel'], ['x-pos', 'y_pos']]
    node_names = {0: 'none',
                  1: 'right',
                  2: 'left',
                  3: 'forward',
                  4: 'backward'
                  }
    num_obs_per_window = sum([len(e) for e in entity_state])
    input_node_keys = range(-1, -len(obs_window_ids) * num_obs_per_window - 1, -1)
    for i, j in enumerate(input_node_keys):
        if 0 <= i % num_obs_per_window <= 3:
            entity_index = 0
        else:
            entity_index = 1
        state_index = i % num_obs_per_window % len(entity_state[entity_index])
        entity = entity_ids[entity_index]
        state = entity_state[entity_index][state_index]
        node_names[j] = obs_window_ids[i % num_obs_windows] + '_' + entity + '_' + state

    # save results
    for i, genome in enumerate(best_genomes):
        with open(path + '/genome_%d' % (i+1), 'wb') as f:
            pickle.dump(genome, f)
        visualize.draw_net(configs[i], genome, filename=path + "/nn_%d.svg" % (i+1), node_names=node_names)
        configs[i].save(path + '/config')
        visualize.plot_stats(stats[i], ylog=True, filename=path + "/fitness%d.svg" % (i+1))
        visualize.plot_species(stats[i], filename=path + "/speciation%d.svg" % (i+1))
