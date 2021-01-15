import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
import random

class bongo_board(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self):
        self.screen_width, self.screen_height = 600, 400
        self.theta_step, self.alpha_step = 0.1, 0.1
        self.base_ball_radian, self.board_lenth, self.board_width = 25, 125, 4
        self.node_radian = 4
        self.pendulum_pole_lenth, self.pendulum_pole_width, self.pendulum_mass = 110, 4, 1
        self.gravity = 0.98 * 10 /6
        self.force_mag = 1.0
        self.tau = 0.01  # seconds between state updates
        self.thetalimit()
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.max_theta,
                         np.finfo(np.float32).max,
                         math.pi/2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        self.center_x, self.center_y = 300, 150
        # Angle at which to fail the episode
        self.theta, self.alpha = 0., 0.
        self.y, self.x = (self.base_ball_radian/2)*math.cos(self.theta),\
            (self.base_ball_radian/2)*math.sin(self.theta)
        self.observation_space = spaces.Box(-high, high,\
                                             dtype=np.float32)
        self.viewer, self.state = None, None
        self.steps_beyond_done = None
        self.seed()
    def seed(self,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def step(self,action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        alpha, alpha_dot, theta, theta_dot = self.state
        if action == 0:
            force = 1
        elif action == 1:
            force = -1
        
        # alphacc = 1 * self.gravity * math.sin(alpha)
        # thetaacc = ((1 * self.gravity * math.cos(alpha) * math.sin(alpha + theta) * \
        #         (self.base_ball_radian/2)) )/((1 * (self.base_ball_radian/2)**2)/2)
        alphacc = force# + alphacc
        thetaacc = (force) #+ thetaacc
        alpha = alpha + self.tau * alpha_dot
        alpha_dot = alpha_dot + self.tau * alphacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.theta = theta
        self.alpha = alpha + math.pi/2
        # self.alpha = theta + math.pi/2
        
        self.y, self.x = (self.base_ball_radian/2)*math.cos(self.theta),\
            (self.base_ball_radian/2)*math.sin(self.theta)
        # self.pendulum_coordinate_y, self.pendulum_coordinate_x = (self.pendulum_pole_lenth)*math.cos(self.alpha),\
        #                                                         (self.pendulum_pole_lenth)*math.sin(self.alpha)
        # print("max theta bool:",theta > self.max_theta)
        # print("min theta bool:",theta < self.min_theta)
        # print("touch plane:",150 - abs(self.pendulum_coordinate_y) < 141-self.node_radian)
        # print(150 - abs(self.pendulum_coordinate_y), 141-self.node_radian)
        done = bool(
            theta < self.min_theta /1
            or theta > self.max_theta /1
            or alpha > math.pi/2
            or alpha < - math.pi/2
        )
        self.state = (alpha, alpha_dot, theta, theta_dot)
        reward = 0
        
        if theta < self.min_theta or theta > self.max_theta:
            reward = -10
        elif alpha > math.pi/4 or alpha < - math.pi/4:
            reward = -10
        else:
            r1 = 1 - abs(alpha)
            r2 = 1 - abs(self.x)
            reward = r1 + r2
        # if not done:
        #     if abs(self.x) <= 1:
        #         reward = 1
        #     else:
        #         reward = 0
        # # elif self.steps_beyond_done is None:
        # #     # Pole just fell!
        # #     self.steps_beyond_done = 0
        # #     reward = 1.
        # else:
        #     # if self.steps_beyond_done == 0:
        #     #     logger.warn(
        #     #         "You are calling 'step()' even though this "
        #     #         "environment has already returned done = True. You "
        #     #         "should always call 'reset()' once you receive 'done = "
        #     #         "True' -- any further steps are undefined behavior."
        #     #     )
        #     # self.steps_beyond_done += 1
        #     reward = 0
        return np.array(self.state), reward, done, {}
    def thetalimit(self):
        _theta = math.atan((self.base_ball_radian/2)/(self.board_lenth/2))
        self.max_theta = 1 * (math.pi - 2 * ((math.pi/2)-_theta))
        self.min_theta = -self.max_theta
        
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            print(self.x, self.y)
            self.fix_point = rendering.Transform(translation=(self.x, self.y))
            self.pole_translation = rendering.Transform(translation=(0,0))
            self.orignal_point = rendering.Transform(translation=(0, 0))
            self.remapping = rendering.Transform()
            self.pendulum_map = rendering.Transform()
            self.node = rendering.make_circle(self.node_radian)
            self.node.set_color(.5, .5, .8)
            self.node.add_attr(self.fix_point)
            self.mass_ball = rendering.make_circle(self.base_ball_radian/2)
            self.mass_ball.set_color(0.3,0.5,0.8)
            self.mass_ball.add_attr(rendering.Transform(translation=(self.pendulum_pole_lenth,0)))
            self.mass_ball.add_attr(self.pendulum_map)
            
            
            l, r, t, b = -self.board_lenth / 2, self.board_lenth / 2, self.board_width / 2, -self.board_width / 2
            self.board = rendering.FilledPolygon([(l, b), (l, t),
                                                 (r, t), (r, b)])
            self.board.add_attr(self.remapping)
            self.planeline = rendering.Line((0,141-self.node_radian),\
                                            (self.screen_width,141-self.node_radian))
            self.planeline.set_color(0., 0., 0.)
            self.base_ball = rendering.make_circle(self.base_ball_radian/2)
            self.base_ball.add_attr(rendering.Transform(translation = (self.center_x,self.center_y)))
            self.base_ball.set_color(0., 0., 0.8)
            
            self.viewer.add_geom(self.base_ball)
            self.viewer.add_geom(self.board)
            self.viewer.add_geom(self.node)
            self.viewer.add_geom(self.planeline)
            self.viewer.add_geom(self.mass_ball)
            
            self.pendulum_pole = rendering.make_capsule(110, 4)
            self.pendulum_pole.set_color(.8, .3, .3)
            self.pendulum_pole.add_attr(self.pendulum_map)

            
            self.viewer.add_geom(self.pendulum_pole)
        if self.state is None:
            return None
        
        self.fix_point.set_translation(self.center_x + self.x, self.center_y + self.y)
        self.remapping.set_translation(self.center_x + self.x, self.center_y + self.y)
        self.remapping.set_rotation(-self.theta)
        self.pendulum_map.set_translation(self.center_x + self.x, self.center_y + self.y)
        self.pendulum_map.set_rotation(self.alpha)
        # pi/2 + theta = alpha
        # theta = alpha - pi/2

        
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

