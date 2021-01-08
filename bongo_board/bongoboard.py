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
        self.screen_width = 600
        self.screen_height = 400
        self.theta_step = 0.1
        self.alpha_step = 0.1
        self.base_ball_radian = 25
        self.board_lenth = 125
        self.board_width = 4
        self.node_radian = 4
        self.pendulum_pole_lenth = 110
        self.pendulum_pole_width = 4
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.action_space = spaces.Discrete(4)
        self.thetalimit()
        self.center_x, self.center_y = 
        # Angle at which to fail the episode
        self.theta, self.alpha = 0.0001, 0.
        self.viewer = None
        self.seed()
    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        if action == 0:
            self.theta += self.theta_step
            
        elif action == 1:
            self.theta -= self.theta_step
        elif action == 2:
            self.alpha += self.alpha_step
        elif action == 3:
            self.alpha -= self.alpha_step
        if self.theta > self.max_theta:
            self.theta = self.max_theta
        elif self.theta < self.min_theta:
            self.theta = self.min_theta
        # if self.alpha >=  self.theta:
        #     self.alpha = self.theta
        # elif self.alpha <=  - self.theta:
        #     self.alpha = -self.theta
        self.y, self.x = (self.base_ball_radian/2)*math.cos(self.theta),\
            (self.base_ball_radian/2)*math.sin(self.theta)
        # print(self.alpha, self.theta)
    def thetalimit(self):
        _theta = math.atan(self.node_radian/(self.board_lenth/2))
        self.max_theta = 2.5 * (math.pi - 2 * ((math.pi/2)-_theta))
        self.min_theta = -self.max_theta
        
    def render(self, mode='human'):
        
        
        if self.viewer is None:
            self.fix_point = rendering.Transform(translation=(self.x, self.y))
            self.pole_translation = rendering.Transform(translation=(0,0))
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.orignal_point = rendering.Transform(translation=(0, 0))
            self.remapping = rendering.Transform()
            self.pendulum_map = rendering.Transform()
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
            
            self.pendulum_pole.set_color(.8, .3, .3)
            self.pendulum_pole.add_attr(self.pole_translation)
            self.pendulum_pole.add_attr(self.pendulum_map)
            # self.pendulum_pole.add_attr(self.remapping)
            
            self.viewer.add_geom(self.pendulum_pole)
        if self.state is None:
            return None
        # Edit the pole polygon vertex
        
        
        # self.pole_translation.set_rotation(math.pi/2)
        self.pendulum_map.set_translation(300 + self.x , 150 + self.y)
        self.pendulum_map.set_rotation(self.alpha )
        self.remapping.set_translation(300, 150)
        self.remapping.set_rotation(self.theta)
        
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

