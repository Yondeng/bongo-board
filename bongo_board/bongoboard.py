import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

class bongo_board(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.base_ball_radian = 25
        self.board_lenth = 125
        self.board_width = 4
        self.node_radian = 4
        self.pendulum_pole_lenth = 110
        self.pendulum_pole_width = 4
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.action_space = spaces.Discrete(2)
        self.thetalimit()
        # Angle at which to fail the episode
        self.theta, self.alpha = 0., 0.
        self.viewer = None
        self.seed()
    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        if action == 0:
            self.theta += 0.02
            self.alpha += 0.01
        elif action == 1:
            self.theta -= 0.02
            self.alpha -= 0.01
        if self.theta > self.max_theta:
            self.theta = self.max_theta
        elif self.theta < self.min_theta:
            self.theta = self.min_theta


        self.y, self.x = (self.base_ball_radian/2)*math.cos(self.theta),\
            (self.base_ball_radian/2)*math.sin(self.theta)
    def thetalimit(self):
        _theta = math.atan(self.node_radian/(self.board_lenth/2))
        self.max_theta = 2.5 * (math.pi - 2 * ((math.pi/2)-_theta))
        self.min_theta = -self.max_theta
        
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        self.fix_point = rendering.Transform(translation=(self.x, self.y))
        self.pole_translation = rendering.Transform(rotation=math.pi/2 - self.alpha,translation=(self.x, self.y))
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.orignal_point = rendering.Transform(translation=(0, 0))
            self.remapping = rendering.Transform()
            self.node = rendering.make_circle(self.node_radian)
            self.node.set_color(.5, .5, .8)
            self.node.add_attr(self.fix_point)
            self.node.add_attr(self.remapping)
            l, r, t, b = -self.board_lenth / 2, self.board_lenth / 2, self.board_width / 2, -self.board_width / 2
            self.board = rendering.FilledPolygon([(l+self.x, b+self.y), (l+self.x, t+self.y),
                                                 (r+self.x, t+self.y), (r+self.x, b+self.y)])
            self.board.add_attr(self.remapping)
            self.planeline = rendering.Line((0,142-self.node_radian),\
                                            (screen_width,142-self.node_radian))
            self.planeline.set_color(0., 0., 0.)
            self.base_ball = rendering.make_circle(self.base_ball_radian/2)
            self.base_ball.add_attr(self.orignal_point)
            self.base_ball.add_attr(self.remapping)
            self.base_ball.set_color(0., 0., 0.8)
            self.viewer.add_geom(self.base_ball)
            self.viewer.add_geom(self.board)
            self.viewer.add_geom(self.node)
            self.viewer.add_geom(self.planeline)
            
            self.pendulum_pole = rendering.make_capsule(110, 4)
            self.pendulum_map = rendering.Transform()
            self.pendulum_pole.set_color(.8, .3, .3)
            self.pendulum_pole.add_attr(self.pole_translation)
            self.pendulum_pole.add_attr(self.remapping)
            self.viewer.add_geom(self.pendulum_pole)
        if self.state is None:
            return None
        # Edit the pole polygon vertex
        self.remapping.set_translation(300, 150)
        self.remapping.set_rotation(self.theta)
        self.pendulum_map.set_translation(300 , 150)
        self.pendulum_map.set_rotation(math.pi/2)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

