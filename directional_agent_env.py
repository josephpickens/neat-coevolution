import numpy as np
from math import sin, cos, pi

from multiagent.environment import MultiAgentEnv
from multiagent.multi_discrete import MultiDiscrete
from multiagent import rendering


class DirectionalMultiAgentEnv(MultiAgentEnv):
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p + 1)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
                if action[0] == 5: agent.action.u[2] = -1.0
                if action[0] == 6: agent.action.u[2] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    right = action[0][1] - action[0][2]
                    forward = action[0][3] - action[0][4]
                    turn = action[0][6] - action[0][7]
                    agent.action.u[0] = right * cos(agent.state.p_ang_pos) - forward * sin(agent.state.p_ang_pos)
                    agent.action.u[1] = forward * cos(agent.state.p_ang_pos) + right * sin(agent.state.p_ang_pos)
                    agent.action.u[2] = turn
                    if hasattr(agent, 'prev_actions'):
                        agent.prev_actions[1:] = agent.prev_actions[:-1]
                        agent.prev_actions[0] = [right, forward, turn]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                xform = rendering.Transform()
                geom = rendering.make_circle(entity.size)
                obs_window = None
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                    obs_window = make_obs_window(0.5, entity.state.p_ang_pos, 2*pi/3)
                    obs_window.set_color(0.5, 0.5, 0.5, alpha=0.1)
                    obs_window.add_attr(xform)
                    self.render_geoms.append(obs_window)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range,
                                       pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results


def make_obs_window(r, theta, window_ang):
    theta = theta % (2 * pi)
    if theta < -pi:
        theta += 2 * pi
    elif theta > pi:
        theta -= 2 * pi
    points = [(0, 0)]
    for i in range(10 + 1):
        ang = (theta - window_ang/2) + i * window_ang / 10
        points.append((-r * sin(ang), r * cos(ang)))
    return rendering.FilledPolygon(points)
