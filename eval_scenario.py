import numpy as np
from math import pi, atan2, cos, sin, acos

from multiagent.core import Agent, Landmark, World
from world import DirectionalWorld, Wall
from multiagent.scenario import BaseScenario


class EvalScenario(BaseScenario):
    def __init__(self, num_agents, num_landmarks, agent_colors, directional):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        if not agent_colors:
            agent_colors = ['red', 'green']
        self.agent_colors = agent_colors
        self.directional = directional

    def make_world(self):
        if self.directional:
            world = DirectionalWorld()
        else:
            world = World()
        # set any world properties first
        world.dim_c = 2
        world.color_dict = {'red': np.array([1, 0, 0]),
                            'orange': np.array([1, 0.5, 0]),
                            'yellow': np.array([1, 1, 0]),
                            'green': np.array([0, 1, 0]),
                            'blue': np.array([0, 0, 1]),
                            'purple': np.array([1, 0, 1]),
                            'gray': np.array([0.85, 0.85, 0.85]),
                            'black': np.array([0.25, 0.25, 0.25])}
        world.agent_colors = self.agent_colors
        world.stack_obs = False
        world.stack_actions = False
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.max_speed = 1
            agent.max_ang_vel = 2
            if world.stack_obs:
                agent.past_obs = [[0] * 24 for _ in range(3)]
            if world.stack_actions:
                agent.prev_actions = [[0, 0, 0] for _ in range(2)]
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        landmark_colors = ['black', 'black']
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.color = world.color_dict[landmark_colors[i % len(landmark_colors)]]
            landmark.collide = False
            landmark.movable = False
        # create boundary walls
        if self.directional:
            vl = Wall(orient='V', axis_pos=-1)
            vr = Wall(orient='V', axis_pos=1)
            ht = Wall(axis_pos=1)
            hb = Wall(axis_pos=-1)
            world.walls = [vl, vr, ht, hb]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.color = world.color_dict[world.agent_colors[i % len(world.agent_colors)]]
            agent.state.p_pos = np.random.uniform(-1 + agent.size, 1 - agent.size, world.dim_p)
            agent.state.p_ang_pos = np.random.uniform(-pi, pi)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_ang_vel = 0
            agent.state.c = np.zeros(world.dim_c)
            if hasattr(agent, 'hunger'):
                agent.hunger = 0
            if hasattr(agent, 'food'):
                agent.food = [0] * len(world.agents)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1 + landmark.size, 1 - landmark.size, world.dim_p)
            landmark.state.p_ang_pos = 0
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_ang_vel = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [self.get_distance(a, l) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
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
        raise NotImplementedError()

    # biologically-inspired observation scheme
    def observation_organism(self, agent, world):
        if not self.directional:
            raise Exception('Attempting to use directional observation scheme in non-directional scenario')
        window_angle = 2*pi/3
        num_wedges = 8
        num_agent_attr = 1
        num_lm_attr = 1
        num_wall_attr = 1
        num_total_attr = num_agent_attr + num_lm_attr + num_wall_attr
        input_size = num_wedges * num_total_attr
        obs = [0] * input_size
        obs_radius = 0.5

        # keep agent orientation between -pi and pi (measured counterclockwise from north)
        p_ang_pos = agent.state.p_ang_pos % (2 * pi)
        if p_ang_pos < -pi:
            p_ang_pos += 2 * pi
        elif p_ang_pos > pi:
            p_ang_pos -= 2 * pi

        # get position of each entity relative to agent coordinate frame
        for entity in world.entities + world.walls:
            if entity is agent:
                continue
            if isinstance(entity, Wall):
                i = 2
                if entity.orient == 'V':
                    dim = 0
                else:
                    dim = 1
                dist = abs(entity.axis_pos - agent.state.p_pos[dim])
                # check if wall is within observable range
                if dist > obs_radius:
                    continue
                # find direction of wall-perpendicular line, relative to agent
                if dim == 0:                    # vertical wall
                    if entity.axis_pos > 0:     # right wall
                        ang = -agent.state.p_ang_pos - pi / 2
                    else:                       # left wall
                        ang = pi / 2 - agent.state.p_ang_pos
                else:                           # horizontal wall
                    if entity.axis_pos > 0:     # top wall
                        ang = -agent.state.p_ang_pos
                    else:                       # bottom wall
                        ang = pi - agent.state.p_ang_pos
                intersect_ang = abs(acos(dist / obs_radius))
                # find directions of intersect points of obs_radius circle with wall, relative to agent
                intersect_angs = [ang - intersect_ang, ang + intersect_ang]
                for a in range(2):
                    if intersect_angs[a] < -pi:
                        intersect_angs[a] += 2 * pi
                    elif intersect_angs[a] > pi:
                        intersect_angs[a] -= 2 * pi
            else:
                if isinstance(entity, Agent):
                    i = 0
                else:
                    i = 1
                pos = entity.state.p_pos - agent.state.p_pos
                pos = [pos[0] * cos(p_ang_pos) + pos[1] * sin(p_ang_pos),
                       pos[1] * cos(p_ang_pos) - pos[0] * sin(p_ang_pos)]
                dist = np.sqrt(np.sum(np.square(pos)))
                direction = atan2(pos[1], pos[0]) - pi/2
                # check entity observability (sufficiently close within agent's observation window)
                if abs(direction) > window_angle / 2 or dist - entity.size > obs_radius:
                    continue
            # assign entity dist to appropriate observation wedges
            j = 0
            while j < num_wedges:
                wedge_min = (j - num_wedges/2) * window_angle / num_wedges
                wedge_max = (j - num_wedges/2 + 1) * window_angle / num_wedges
                if ((i != 2 and wedge_min <= direction < wedge_max)
                        or (i == 2 and (intersect_angs[0] <= wedge_min <= intersect_angs[1]
                                        or intersect_angs[0] <= wedge_max <= intersect_angs[1]))):
                    # check for multiple entities within this window and observe closest entity only
                    closest = True
                    w = num_total_attr * j
                    for e in range(w, w + num_total_attr):
                        if obs[e] != 0:
                            if dist < obs[e]:
                                obs[e] = 0
                            else:
                                closest = False
                    if closest:
                        obs[num_total_attr * j + i] = dist
                j += 1
        # obs.append(agent.hunger - agent.prev_hunger)
        # stack frames
        if world.stack_obs:
            current_frame = [o for o in obs]
            obs += [o for frame in agent.past_obs for o in frame]
            agent.past_obs[1:] = agent.past_obs[:-1]
            agent.past_obs[0] = current_frame
        if world.stack_actions:
            for a in agent.prev_actions:
                obs += a
        return obs

    # global observation scheme with relative positions and velocities of all entities
    def observation(self, agent, world):
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

        # x-vel, y-vel, x-pos, y-pos, x-lm0-vec, y-lm1-vec, x-lm2-vec, y-lm2-vec,
        # x-other-vec, y-other-vec, x-other-vel, y-other-vel
        return obs

    def done(self, agent, world):
        return False

