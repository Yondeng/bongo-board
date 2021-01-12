import math
import gym
import time
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnv1(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.ctheta = 0
        self.alpha = 0
        self.d_cylinder = 25 #0.25m
        self.cx = 300
        self.cy = 80
        
        self.gravity = 9.8
        self.masscart = 0.1
        self.masspole = 5.0        
        self.total_mass = (self.masspole + self.masscart)
        self.length = 1.1  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 90 * math.pi / 180
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        
        T = self.ctheta * math.pi / 180
        #point C
        self.cx = self.d_cylinder * math.cos(T + 90* math.pi / 180) + 300
        self.cy = self.d_cylinder * math.sin(T + 90* math.pi / 180) + 80
        
        #board Endpoint
        Endpoint = 125 * math.sin(T)

        theta, theta_dot = self.state
        
        if theta > 0 * math.pi / 180:
            self.ctheta -= 1
        elif theta < 0 * math.pi / 180:
            self.ctheta += 1
        
        if self.ctheta < 0:
            force = self.force_mag
        elif self.ctheta == 0:
            force = 0
        else:
            force = -self.force_mag
        
        # force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        
        if self.kinematics_integrator == 'euler':
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (theta, theta_dot)
        done = bool(
            self.cy + Endpoint < 80 - self.d_cylinder
            or self.cy - Endpoint < 80 - self.d_cylinder
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0


        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(2,))
        self.steps_beyond_done = None
        self.ctheta = 0
        self.cx = 300
        self.cy = 80
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            #cylinder
            cylinder = rendering.make_circle(self.d_cylinder)
            cylinderpos = rendering.Transform(translation=(300,80))
            cylinder.set_color(1,0.9,0)
            cylinder.add_attr(cylinderpos)
            self.viewer.add_geom(cylinder)
            
            #board
            l, r, t, b = -125 , 125 ,4 ,-4
            board = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            board.set_color(.8, .6, .4)
            self.boardpos = rendering.Transform(translation=(0, 0))
            board.add_attr(self.boardpos)
            self.viewer.add_geom(board)
            
            #point C
            C = rendering.make_circle(4)
            self.Cpos = rendering.Transform(translation=(0,0))
            C.set_color(255,0,0)
            C.add_attr(self.Cpos)
            self.viewer.add_geom(C)
            
            #horizon
            horizon = rendering.Line((0 , 80 - self.d_cylinder),(600 , 80 - self.d_cylinder))
            horizon.set_color(0, 0, 0)
            self.viewer.add_geom(horizon)
            
            #stick
            t, b, l, r = 220 ,0 ,4 ,-4
            stick = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            stick.set_color(0, 0, 1)
            self.stickpos = rendering.Transform(translation=(0, 0))
            stick.add_attr(self.stickpos)
            self.viewer.add_geom(stick)
            
            #Centroid
            Centroid = rendering.make_circle(25)
            self.Centroidpos = rendering.Transform(translation=(0,0))
            Centroid.set_color(0.5,2.5,5)
            Centroid.add_attr(self.Centroidpos)
            self.viewer.add_geom(Centroid)
        
        if self.state is None:
            return None
            
        T = self.ctheta * math.pi / 180
        
        angle = self.state
        self.Cpos.set_translation(self.cx ,self.cy)
        self.boardpos.set_translation(self.cx ,self.cy)
        self.boardpos.set_rotation(T)
        self.stickpos.set_translation(self.cx ,self.cy)
        self.stickpos.set_rotation(angle[0])
        
        x = self.cx - 220 * math.sin(angle[0])
        y = self.cy + 220 * math.cos(angle[0])
        self.Centroidpos.set_translation(x ,y)
        
        time.sleep(self.tau)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None