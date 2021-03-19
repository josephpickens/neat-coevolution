#!/usr/bin/env python

import numpy as np
from math import pi, atan2
from itertools import permutations, repeat
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class CooperativeScenario(BaseScenario):
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.num_landmarks = num_agents

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.discrete_action = False
        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 0.25
        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
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
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def is_collision(self, entity1, entity2):
        delta_pos = entity1.state.p_pos - entity2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity1.size + entity2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        rew = 0
        all_agent_landmark_pairings = list(list(zip(r, p)) for (r, p) in zip(repeat(world.landmarks),
                                                                             permutations(world.agents)))
        # agent_landmark_distances = [[a1/lm1 dist, a2/lm2 dist], [a2/lm1 dist, a1/lm2 dist]]
        agent_landmark_distances = [[np.sqrt(np.sum(np.square(lm.state.p_pos - a.state.p_pos)))
                                     for (lm, a) in pairing] for pairing in all_agent_landmark_pairings]
        # index of optimal agent/landmark pairing (pairing with minimum total distance between agent/landmark)
        optimal_pairing_index = np.argmin([sum(ds) for ds in agent_landmark_distances])
        # index of this agent within the optimal pairing
        agent_index = [i for i in range(self.num_agents)
                       if all_agent_landmark_pairings[optimal_pairing_index][i][1] is agent][0]
        # negative payoff proportional to this agent's distance from the landmark within the optimal
        # agent/landmark pairing
        rew -= 0.1 * agent_landmark_distances[optimal_pairing_index][agent_index]

        # collective positive payoff if both landmarks occupied by distinct agents
        if any([all([self.is_collision(lm, a) for (lm, a) in pairing]) for pairing in all_agent_landmark_pairings]):
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
        # agent_landmark_pairs = list(
        #     list(zip(r, p)) for (r, p) in zip(repeat(world.landmarks), permutations(world.agents)))
        # if any([all([self.is_collision(lm, a) for (lm, a) in p]) for p in agent_landmark_pairs]):
        #     return True
        return False
