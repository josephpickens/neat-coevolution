import numpy as np

# from multiagent.environment import MultiAgentEnv
from directional_agent_env import DirectionalMultiAgentEnv
from cooperative_scenario import CooperativeScenario
from fitness_function import pareto_ranking


def test_action():
    scenario = CooperativeScenario(agent_colors=['green', 'blue'])
    world = scenario.make_world()
    env = DirectionalMultiAgentEnv(world, scenario.reset_world, scenario.reward,
                                   scenario.observation, done_callback=scenario.done)
    act_n = []
    move_idx = [0, 0]
    turn_idx = [3, 3]
    u = [np.zeros(7) for _ in range(2)]
    for i in range(2):
        u[i][move_idx[i]] = 1.0
        u[i][turn_idx[i]] = 1.0
        act_n.append(np.concatenate([u[i], np.zeros(env.world.dim_c)]))
    k = 0
    # env.world.agents[1].state.p_pos = np.ones(2)
    # env.world.landmarks[1].state.p_pos = np.ones(2) * -1
    # env.world.agents[0].state.p_pos = np.zeros(2)
    # env.world.agents[0].state.p_ang_pos = 3.14 / 2
    # env.world.landmarks[0].state.p_pos = np.array([0, 0.3])
    while True:
        env._reset_render()
        env.render()
        obs_n, reward_n, _, _ = env.step(act_n)
        print(obs_n)
        k += 1


def test_pareto():
    population = ['a', 'b', 'c', 'd', 'e']
    objectives = [{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'b': 1, 'a': 2, 'd': 3, 'c': 4, 'e': 5}]
    ranks = pareto_ranking(population, objectives)
    print(ranks)


if __name__ == '__main__':
    test_action()