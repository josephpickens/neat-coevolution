#!/usr/bin/env python

import numpy as np
from math import pi, atan2
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class CompetitiveScenario(BaseScenario):
    def __init__(self, num_pursuers=1, num_evaders=1, num_landmarks=1):
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.num_landmarks = num_landmarks

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.discrete_action = False
        num_agents = self.num_pursuers + self.num_evaders
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.pursuer = True if i < self.num_pursuers else False
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 0.25
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
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
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

    # return all evaders
    def evaders(self, world):
        return [agent for agent in world.agents if not agent.pursuer]

    # return all pursuers
    def pursuers(self, world):
        return [agent for agent in world.agents if agent.pursuer]

    def reward(self, agent, world):
        # Agents are rewarded based on whether they are pursuer or evader
        main_reward = self.pursuer_reward(agent, world) if agent.pursuer else self.evader_reward(agent, world)
        return main_reward

    def evader_reward(self, agent, world):
        # evaders are rewarded for reaching landmark before being caught
        rew = 0
        shape = True
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from landmark)
            rew -= 0.1 * min(np.sqrt(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
                             for lm in world.landmarks)
        if any([self.is_collision(agent, lm) for lm in world.landmarks]):
            rew += 1
        if any([self.is_collision(agent, p) and not self.is_collision(agent, lm)
                for p in self.pursuers(world) for lm in world.landmarks]):
            rew -= 1
        # # boundary penalty
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= self.boundary_penalty(x)
        return rew

    def pursuer_reward(self, agent, world):
        # pursuers are rewarded for collisions with evaders
        rew = 0
        shape = True
        evaders = self.evaders(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from evaders)
            rew -= 0.1 * min([np.sqrt(np.sum(np.square(agent.state.p_pos - e.state.p_pos)))
                              for e in evaders])
        if any([self.is_collision(agent, e) and not self.is_collision(e, lm)
                for e in evaders for lm in world.landmarks]):
            rew += 1
        # # boundary penalty
        # for p in range(world.dim_p):
        #     x = abs(agent.state.p_pos[p])
        #     rew -= self.boundary_penalty(x)
        return rew

    # agent penalty for exiting the screen
    def boundary_penalty(self, x):
        if x < 0.9:
            return 0
        return 10

    def observation(self, agent, world):
        entity_headings = [0] * 16
        for entity in world.entities:
            if entity is agent:
                continue
            if isinstance(entity, Agent):
                i = 0
            else:
                i = 1
            pos = entity.state.p_pos - agent.state.p_pos
            direction = atan2(pos[1], pos[0])
            if direction < 0:
                direction += 2 * pi
            entity_found = False
            j = 1
            while j < 8 and not entity_found:
                if (2 * j - 1) * pi / 8 <= direction < (2 * j + 1) * pi / 8:
                    entity_headings[2 * j + i] = 1
                    entity_found = True
                j += 1
            if not entity_found:
                entity_headings[i] = 1
        return entity_headings

    def done(self, agent, world):
        if agent.pursuer and any([self.is_collision(agent, e) for e in self.evaders(world)]):
            return True
        elif any([self.is_collision(agent, lm) for lm in world.landmarks]):
            return True
        return False
