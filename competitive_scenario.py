#!/usr/bin/env python

import numpy as np

from multiagent.core import World, Agent, Landmark
from eval_scenario import EvalScenario


class CompetitiveScenario(EvalScenario):
    def __init__(self):
        super().__init__()
        self.num_pursuers = 1
        self.num_landmarks = 1

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.pursuer = True if i < self.num_pursuers else False
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # properties for agents
        red = np.array([0.85, 0.35, 0.35])
        green = np.array([0.35, 0.85, 0.35])
        black = np.array([0.25, 0.25, 0.25])
        for i, agent in enumerate(world.agents):
            if agent.pursuer:
                agent.color = red
            else:
                agent.color = green
        # properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = black
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # return all evaders
    def evaders(self, world):
        return [agent for agent in world.agents if not agent.pursuer]

    # return all pursuers
    def pursuers(self, world):
        return [agent for agent in world.agents if agent.pursuer]

    def reward(self, agent, world):
        # Agents are rewarded based on whether they are pursuer or evader
        return self.pursuer_reward(agent, world) if agent.pursuer else self.evader_reward(agent, world)

    def evader_reward(self, agent, world):
        # evaders are rewarded for reaching landmark before being caught
        rew = 0
        shape = True
        for lm in world.landmarks:
            dist = self.get_distance(agent, lm)
            if self.is_collision(agent, lm, distance=dist):
                rew += 10
            else:
                rew -= dist
                if any([self.is_collision(agent, p) for p in self.pursuers(world)]):
                    rew -= 10
        return rew

    def pursuer_reward(self, agent, world):
        # pursuers are rewarded for collisions with evaders
        rew = 0
        evaders = self.evaders(world)
        if any([self.is_collision(agent, lm) for lm in world.landmarks]):
            rew -= 10
        else:
            for e in evaders:
                dist = np.sqrt(np.sum(np.square(agent.state.p_pos - e.state.p_pos)))
                if dist < agent.size + e.size:
                    rew += 10
                else:
                    rew -= dist
        return rew
