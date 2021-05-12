import os
import pickle
import numpy as np

import neat

from build import build_ecosystem
from world import DirectionalWorld


class Animator():
    def __init__(self, results_path, directional):
        self.results_path = results_path
        self.directional = directional

    def load_nets(self):
        ecosystem_type = self.results_path.split('/')[1].strip()
        # Watch the winners play
        ecosystem = build_ecosystem(ecosystem_type, self.directional)
        num_genomes = len(ecosystem.pops)
        winners = []
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, self.results_path + '/config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        for i in range(num_genomes):
            winner = pickle.load(open(self.results_path + '/genome_%d' % (i + 1), 'rb'))
            winners.append(winner)
        env_assign_map = {}
        for i, p in enumerate(ecosystem.pops):
            env_assign_map[p] = winners[i]
        nets = [[] for _ in ecosystem.envs]
        for i, env_pops in enumerate(ecosystem.assigned_pops):
            for p in env_pops:
                genome = env_assign_map[p]
                nets[i].append(neat.nn.RecurrentNetwork.create(genome, config))
        return ecosystem.envs, nets

    def play_nets(self):
        envs, nets = self.load_nets()
        directional = isinstance(envs[0].world, DirectionalWorld)
        steps_per_run = 100
        import time
        while True:
            for i, env in enumerate(envs):
                obs_n = env.reset()
                total_reward_n = [0] * len(env.world.agents)
                env.render(mode=None)
                for _ in range(steps_per_run):
                    # query for action from each agent's policy
                    act_n = []
                    for j, net in enumerate(nets[i]):
                        activation = net.activate(obs_n[j])
                        u = np.zeros(len(activation))
                        move_idx = np.argmax(activation[0:5])
                        u[move_idx] = 1.0
                        if directional:
                            turn_idx = np.argmax(activation[5:]) + 5
                            u[turn_idx] = 1.0
                        act_n.append(np.concatenate([u, np.zeros(env.world.dim_c)]))
                        agent = env.world.agents[j]
                        agent.color = env.world.color_dict[env.world.agent_colors[j]]
                        # if agent.hunger > 1:
                        #     agent.color = agent.color / agent.hunger
                    # step environment
                    obs_n, reward_n, _, _ = env.step(act_n)
                    total_reward_n = [sum(x) for x in zip(total_reward_n, reward_n)]
                    # render all agent views
                    env._reset_render()
                    env.render(mode=None)
                    time.sleep(0.05)


if __name__ == '__main__':
    # path = 'results/2_competitive/20210512_103111/100'    # global observations, competitive results
    # path = 'results/2_cooperative/20210406_173302/30'     # global observations, cooperative results
    # path = 'results/3_mixed/20210413_123800/55'             # global observations, 3-mixed, evader plays both games
    # path = 'results/3_mixed/20210413_135354/85'             # global observations, 3-mixed, pursuer plays both games
    path = 'results/2_competitive/20210512_122056/100'        # local observations, walls, recurrent connections, competitive results
    # path = 'results/2_cooperative/20210427_081631/490'      # local observations, walls, recurrent connections, cooperative results
    animator = Animator(path, True)
    animator.play_nets()
