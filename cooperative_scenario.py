#!/usr/bin/env python

import numpy as np

from multiagent.core import World, Agent, Landmark
from eval_scenario import EvalScenario


class CooperativeScenario(EvalScenario):
    def __init__(self, agent_colors=None, directional=True):
        super().__init__(2, 2, agent_colors, directional)

    def make_world(self):
        world = super().make_world()
        world.collaborative = True
        for agent in world.agents:
            agent.hunger = 0
        self.reset_world(world)
        return world

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        dists = [self.get_distance(agent, lm) for lm in world.landmarks]
        rew -= min(dists)

        for a in world.agents:
            if self.is_collision(a, agent):
                rew -= 1
        feeding = False
        for lm in world.landmarks:
            if self.is_collision(agent, lm):
                other_at_landmark = False
                for a in world.agents:
                    if a != agent and self.is_collision(a, lm):
                        other_at_landmark = True
                if other_at_landmark:
                    rew += 1
                else:
                    rew += 10
                feeding = True
                break
        # agent.prev_hunger = agent.hunger
        if feeding:
            agent.hunger = max(agent.hunger - 0.1, 0)
        else:
            agent.hunger += 0.1
            if np.sqrt(np.sum(np.square(agent.state.p_vel))) < 0.001:
                rew -= 1
        rew -= agent.hunger

        return rew

    # def reward(self, agent, world):
    #     # food trading game
    #     rew = 0
    #     lm_dists = [self.get_distance(agent, lm) for lm in world.landmarks]
    #     other_dists = [self.get_distance(agent, a) for a in world.agents if a is not agent]
    #     rew -= min(lm_dists)
    #     rew -= min(other_dists)
    #     agent_index = int(agent.name.split()[1])
    #     other_index = (agent_index + 1) % len(world.agents)
    #     # gather food
    #     for i, lm in enumerate(world.landmarks):
    #         if self.is_collision(agent, lm):
    #             if i == other_index:
    #                 agent.food[other_index] += 5
    #
    #     # exchange food
    #     for a in world.agents:
    #         if self.is_collision(a, agent):
    #             if a.food[agent_index] > 0:
    #                 amount = min(a.food[agent_index], 5)
    #                 agent.food[agent_index] += amount
    #                 a.food[agent_index] -= amount
    #             if agent.food[other_index] > 0:
    #                 amount = min(agent.food[other_index], 5)
    #                 agent.food[other_index] -= amount
    #                 a.food[other_index] += amount
    #
    #     # consume food to combat hunger
    #     agent.hunger += 2
    #     if agent.food[agent_index] > 0:
    #         amount = min(agent.food[agent_index], 5)
    #         agent.hunger = max(agent.hunger - amount, 0)
    #         agent.food[agent_index] -= amount
    #     rew -= agent.hunger
    #     return rew
