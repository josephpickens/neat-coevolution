import numpy as np

from multiagent.core import World


class DirectionalWorld(World):
    def __init__(self):
        super().__init__()
        self.walls = []

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        p_torque = [None] * len(self.entities)
        # apply agent physical controls
        p_force, p_torque = self.apply_action_force(p_force, p_torque)
        # apply environment forces
        p_force, p_torque = self.apply_environment_force(p_force, p_torque)
        # integrate physical state
        self.integrate_state(p_force, p_torque)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force, p_torque):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                noisy_action = agent.action.u + noise
                p_force[i] = noisy_action[0:self.dim_p]
                p_torque[i] = noisy_action[self.dim_p]
        return p_force, p_torque

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force, p_torque):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force, p_torque

    # integrate physical state
    def integrate_state(self, p_force, p_torque):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if p_torque[i] is not None:
                entity.state.p_ang_vel += (p_torque[i] / (0.5 * entity.mass * np.square(entity.size))) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            if hasattr(entity, 'max_ang_vel') and entity.max_ang_vel is not None:
                if abs(entity.state.p_ang_vel) > entity.max_ang_vel:
                    entity.state.p_ang_vel = entity.state.p_ang_vel / abs(entity.state.p_ang_vel) * entity.max_ang_vel
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_ang_pos += entity.state.p_ang_vel * self.dt

    # from shariqiqbal2810: get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
              ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force


# Wall class from shariqiqbal2810
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                 hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])
