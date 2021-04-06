#!/usr/bin/env python

import numpy as np

from eval_scenario import EvalScenario


class CooperativeScenario(EvalScenario):
    def __init__(self):
        super().__init__()

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for lm in world.landmarks:
            dists = [self.get_distance(a, lm) for a in world.agents]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            found_home = False
            for landmark in world.landmarks:
                if self.is_collision(agent, landmark):
                    rew += 10
                    found_home = True
            if not found_home and np.sqrt(np.sum(np.square(agent.state.p_vel))) < 0.001:
                rew -= 1

        return rew
