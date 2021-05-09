import os
import pickle
import string

import visualize


def save_results(path, best_genomes, generation, configs, stats, directional):
    path += '/' + str(generation)
    os.mkdir(path)

    stack_obs = False
    stack_actions = False

    if directional:
        # node name dictionary
        num_wedges = 8
        wedge_ids = list(string.ascii_uppercase)[0:num_wedges]
        entities = ['agent', 'lm', 'wall']
        node_names = {0: 'no_move',
                      1: 'right',
                      2: 'left',
                      3: 'forward',
                      4: 'backward',
                      5: 'no_turn',
                      6: 'ccw',
                      7: 'cw'}
        num_keys = len(entities) * num_wedges
        input_node_keys = range(-1, -num_keys - 1, -1)
        for i, j in enumerate(input_node_keys):
            node_names[j] = wedge_ids[i // len(entities)] + '_' + entities[i % len(entities)]
        if stack_obs:
            obs_memory = 4
            input_node_keys = list(input_node_keys) + list(range(-num_keys, -num_keys * obs_memory - 1, -1))
            for i, j in enumerate(input_node_keys[num_keys + 1:]):
                node_names[j] = (wedge_ids[(i // len(entities)) % num_wedges] + '_' + entities[i % len(entities)]
                                 + '_t-%d' % (i // num_keys + 1))
        # node_names[-num_keys - 1] = 'hunger_change'
        if stack_actions:
            action_memory = 2
            actions = ['right', 'forward', 'turn']
            input_node_keys = list(input_node_keys) + list(range(-num_keys, -num_keys - 3 * action_memory - 1, -1))
            for i, j in enumerate(input_node_keys[num_keys + 1:]):
                node_names[j] = actions[(i % len(actions))] + '_t-%d' % (i // len(actions) + 1)
    else:
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

    # Save results
    for i, genome in enumerate(best_genomes):
        with open(path + '/genome_%d' % (i+1), 'wb') as f:
            pickle.dump(genome, f)
        visualize.draw_net(configs[i], genome, filename=path + "/nn_%d.svg" % (i+1), node_names=node_names)
        configs[i].save(path + '/config')
        visualize.plot_stats(stats[i], ylog=True, filename=path + "/fitness%d.svg" % (i+1))
        visualize.plot_species(stats[i], filename=path + "/speciation%d.svg" % (i+1))
