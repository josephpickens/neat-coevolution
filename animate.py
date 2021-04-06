import os
import pickle

import neat
import numpy as np

from build import build_ecosystem


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
    steps_per_run = 100
    import time
    while True:
        for i, env in enumerate(ecosystem.envs):
            obs_n = env.reset()
            env.render()
            for _ in range(steps_per_run):

                # query for action from each agent's policy
                act_n = []
                for j, net in enumerate(nets[i]):
                    activation = net.activate(obs_n[j])
                    action_idx = np.argmax(activation)
                    u = np.zeros(5)
                    u[action_idx] = 1.0
                    act_n.append(np.concatenate([u, np.zeros(env.world.dim_c)]))
                # step environment
                obs_n, _, _, _ = env.step(act_n)
                # render all agent views
                env.render()
                time.sleep(0.05)


if __name__ == '__main__':
    play_winners('results/2_spread/20210404_110630/60')
