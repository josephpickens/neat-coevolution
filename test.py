import numpy as np

from multiagent.environment import MultiAgentEnv
from competitive_scenario import CompetitiveScenario
from fitness_function import pareto_ranking


def test_action():
    scenario = CompetitiveScenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        done_callback=scenario.done)
    act_n = [np.concatenate([[1.0, 0, 0, 0, 0], np.zeros(world.dim_c)])]
    k = 0
    while k < 50:
        env.render()
        obs_n, reward_n, done_n, _ = env.step(act_n)
        k += 1


def test_pareto():
    population = ['a', 'b', 'c', 'd', 'e']
    objectives = [{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'b': 1, 'a': 2, 'd': 3, 'c': 4, 'e': 5}]
    ranks = pareto_ranking(population, objectives)
    print(ranks)


if __name__ == '__main__':
    test_pareto()
