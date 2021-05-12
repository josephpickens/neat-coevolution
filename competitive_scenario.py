#!/usr/bin/env python

import numpy as np

from multiagent.core import World, Agent, Landmark
from eval_scenario import EvalScenario


class CompetitiveScenario(EvalScenario):
    def __init__(self, agent_colors=None, directional=True):
        super().__init__(2, 2, agent_colors, directional)

    def make_world(self):
        world = super().make_world()
        world.agents[0].pursuer = True
        world.agents[1].pursuer = False
        for agent in world.agents:
            agent.hunger = 0
        self.reset_world(world)
        return world

    # return all evaders
    def evaders(self, world):
        return [agent for agent in world.agents if not agent.pursuer]

    # return all pursuers
    def pursuers(self, world):
        return [agent for agent in world.agents if agent.pursuer]

    def reward(self, agent, world):
        if agent.pursuer:
            return self.pursuer_reward(agent, world)
        else:
            return self.evader_reward(agent, world)

    # def reward(self, agent, world):
    #     rew = 0
    #     other = [a for a in world.agents if a is not agent][0]
    #     agent_lm_dists = [self.get_distance(agent, lm) for lm in world.landmarks]
    #     other_lm_dists = [self.get_distance(other, lm) for lm in world.landmarks]
    #     min_other_lm_index = min(range(len(other_lm_dists)), key=lambda k: other_lm_dists[k])
    #     a_dist = self.get_distance(agent, other)
    #     if agent is world.agents[0]:
    #         rew -= a_dist
    #         rew -= agent_lm_dists[min_other_lm_index]
    #     else:
    #         rew += a_dist
    #         rew -= agent_lm_dists[(min_other_lm_index + 1) % len(agent_lm_dists)]
    #     for i, lm in enumerate(world.landmarks):
    #         if self.is_collision(agent, lm):
    #             if self.is_collision(other, lm):
    #                 if agent is world.agents[0]:
    #                     rew += 10
    #                 else:
    #                     rew -= 10
    #             elif self.is_collision(other, world.landmarks[(i + 1) % len(world.landmarks)]):
    #                 if agent is world.agents[0]:
    #                     rew -= 10
    #                 else:
    #                     rew += 10
    #     return rew

    def evader_reward(self, agent, world):
        # evaders are rewarded for reaching landmark and penalized for being caught by pursuer
        rew = 0
        dists = [self.get_distance(agent, lm) for lm in world.landmarks]
        rew -= min(dists)
        if any(self.is_collision(agent, lm) for lm in world.landmarks):
            rew += 10
        elif any([self.is_collision(agent, p) for p in self.pursuers(world)]):
            rew -= 10
        return rew

    def pursuer_reward(self, agent, world):
        # pursuers are rewarded for collisions with unprotected evaders
        rew = 0
        evaders = self.evaders(world)
        dists = [self.get_distance(agent, e) for e in evaders]
        rew -= min(dists)
        if any([self.is_collision(agent, lm) for lm in world.landmarks]):
            rew -= 10
        elif any([self.is_collision(agent, e) for e in evaders]):
            rew += 10
        return rew
