import os
import pickle
import numpy as np

import neat

from build import build_ecosystem


class Animator():
    def __init__(self, results_path, plot_hunger=False):
        self.results_path = results_path
        self.plot_hunger = plot_hunger

    def load_nets(self):
        ecosystem_type = self.results_path.split('/')[1].strip()
        # Watch the winners play
        ecosystem = build_ecosystem(ecosystem_type)
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
                        move_idx = np.argmax(activation[0:5])
                        turn_idx = np.argmax(activation[5:]) + 5
                        u = np.zeros(len(activation))
                        u[move_idx] = 1.0
                        u[turn_idx] = 1.0
                        act_n.append(np.concatenate([u, np.zeros(env.world.dim_c)]))
                        agent = env.world.agents[j]
                        agent.color = env.world.color_dict[env.world.agent_colors[j]]
                        if agent.hunger > 1:
                            agent.color = agent.color / agent.hunger
                    # step environment
                    obs_n, reward_n, _, _ = env.step(act_n)
                    total_reward_n = [sum(x) for x in zip(total_reward_n, reward_n)]
                    # render all agent views
                    env._reset_render()
                    env.render(mode=None)
                    time.sleep(0.05)


if __name__ == '__main__':
    path = 'results/2_cooperative/20210504_223832/305'
    animator = Animator(path)
    animator.play_nets()
