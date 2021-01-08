import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class bongo_board(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.base_ball_radian = 25
        self.board_lenth = 125
        self.board_width = 10
        self.node_radian = 10
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.action_space = spaces.Discrete(2)
        # Angle at which to fail the episode
        self.theta = 0.
        self.viewer = None
        self.seed()
    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        if action == 0:
            self.theta += 0.1
        elif action == 1:
            self.theta -= 0.1

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.orignal_point = rendering.Transform(translation=(0, 0))
            self.remapping = rendering.Transform()
            y, x = self.base_ball_radian*math.cos(self.theta), self.base_ball_radian*math.sin(self.theta)
            self.fix_point = rendering.Transform(translation=(x, y))
            self.node = rendering.make_circle(self.node_radian)
            self.node.set_color(.5, .5, .8)
            self.node.add_attr(self.fix_point)
            self.node.add_attr(self.remapping)
            l, r, t, b = -self.board_lenth / 2, self.board_lenth / 2, self.board_width / 2, -self.board_width / 2
            self.board = rendering.FilledPolygon([(l+x, b+y), (l+x, t+y), (r+x, t+y), (r+x, b+y)])
            self.board.add_attr(self.remapping)
            # print(l, r, t, b)
            # axleoffset = cartheight / 4.0
            # cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # self.carttrans = rendering.Transform()
            # cart.add_attr(self.carttrans)
            
            # l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            # pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            # pole.set_color(.8, .6, .4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5, .5, .8)
            
            self.base_ball = rendering.make_circle(25)
            
            self.base_ball.add_attr(self.orignal_point)
            self.base_ball.add_attr(self.remapping)
            self.base_ball.set_color(0., 0., 0.8)
            self.viewer.add_geom(self.base_ball)
            self.viewer.add_geom(self.board)
            self.viewer.add_geom(self.node)
            # self.viewer.add_geom(cart)
            # self.viewer.add_geom(self.axle)



            

        if self.state is None:
            return None

        # Edit the pole polygon vertex

        self.remapping.set_translation(300, 150)
        self.remapping.set_rotation(self.theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

