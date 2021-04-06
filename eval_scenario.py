#!/usr/bin/env python

import numpy as np
from math import pi, atan2

from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class EvalScenario(BaseScenario):
    def __init__(self):
        self.num_agents = 2
        self.num_landmarks = 2

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
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
        green = np.array([0.35, 0.85, 0.35])
        blue = np.array([0.35, 0.35, 0.85])
        black = np.array([0.25, 0.25, 0.25])
        colors = [green, blue]
        for i, agent in enumerate(world.agents):
            agent.color = colors[i % 2]
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

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def get_distance(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        return dist

    def is_collision(self, entity1, entity2, distance=None):
        if entity1 == entity2:
            return False
        if distance is None:
            distance = self.get_distance(entity1, entity2)
        dist_min = entity1.size + entity2.size
        return True if distance < dist_min else False

    def reward(self, agent, world):
        return None

    def observation(self, agent, world):
        # num_dir_sense = 8
        # num_agent_attr = 4
        # num_lm_attr = 2
        # input_size = num_dir_sense * (num_agent_attr + num_lm_attr)
        # obs = [0] * input_size
        # obs_radius = 0.5
        # for entity in world.entities:
        #     if entity is agent:
        #         continue
        #     if isinstance(entity, Agent):
        #         i = 0
        #     else:
        #         i = 1
        #     pos = entity.state.p_pos - agent.state.p_pos
        #     vel = entity.state.p_vel - agent.state.p_vel
        #     dist = np.sqrt(np.sum(np.square(pos)))
        #     if pos[1] < 0 or dist > obs_radius:
        #         continue
        #     direction = atan2(pos[1], pos[0]) + pi / 2
        #     j = 2
        #     while j < 10:
        #         if j * pi / 12 <= direction < (j + 1) * pi / 12:
        #             other_pos = np.array((obs[j - 2 + i], obs[j - 1 + i]))
        #             other_dist = np.sqrt(np.sum(np.square(other_pos)))
        #             if (other_pos[0] == 0 and other_pos[1] == 0) or dist < other_dist:
        #                 k = (num_agent_attr + num_lm_attr) * (j - 2)
        #                 if i == 0:
        #                     obs[k:k+num_agent_attr] = list(pos) + list(vel)
        #                 else:
        #                     obs[k+num_agent_attr:k+num_agent_attr+num_lm_attr] = list(pos)
        #         j += 1
        # return obs

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        # comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_pos.append(other.state.p_vel)
        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)  # + comm)

        # x-vel, y-vel, x-pos, y-pos, x-lm1-vec, y-lm1-vec, x-lm2-vec, y-lm2-vec, x-other-vec, y-other-vec, x-other-vel, y-other-vel
        return obs

    def done(self, agent, world):
        return False
