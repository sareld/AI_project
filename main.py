import math, sys, random
import argparse

import os

from Cart import *

import pygame
from pygame.locals import *
from pygame.color import *
import time
import matplotlib.pyplot as plt
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

import pickle

MODEL_FILE = ''

SCREEN_SIZE = (1200, 600)

ANGLE_RANGE = 0.02
TOP_VEL = 200

GOOD_REWORD = 5
BAD_REWORD = -1
DEFAULT_REWORD = -0.1

PENDULUM_NUM = 1
PENDULUM_LEN = 200


FPS = 25
DT = 25
EPISODE_LENGTH = 500


ARGS = None


class CarEnvironment:
    def __init__(self):
        global contact
        global shape_to_remove

        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.running = True
        self.clock = pygame.time.Clock()
        ### Physics stuff
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -900.0)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        # disable the build in debug draw of collision point since we use our own code.
        self.draw_options.flags = self.draw_options.flags ^ pymunk.pygame_util.DrawOptions.DRAW_COLLISION_POINTS
        self.accu_rewards = []
        self.time = []
        self.cart = Cart(ARGS.DISCOUNT, ARGS.ALPHA, ARGS.EPSILON, self.space,
                         PENDULUM_LEN, PENDULUM_NUM, ARGS.CYCLIC_SCREEN, ARGS.NO_SWING,ARGS.Q_MODEL)
        try:
            if(ARGS.MODEL_FILE!=''):
                (self.cart.myQ,(self.time,self.accu_rewards)) = pickle.load(open(ARGS.MODEL_FILE, "rb"))
                print("model loaded")
        except:
            print("no model found")

    def main(self):
        episode_num = 1
        sum = 0
        if(len(self.time)>0):
            t_start = self.time[-1]
        else:
            t_start = 0

        t0 = time.time()
        while self.running:
            accu_reward = 0
            i = 0
            while i < EPISODE_LENGTH and self.running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                        break
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            self.running = False
                            break
                        if event.key == K_a:
                            self.cart.balls[0].velocity += (-100, 0)
                        if event.key == K_d:
                            self.cart.balls[0].velocity += (100, 0)

                state = self.cart.getState()
                if ARGS.EXPLORE_EXPLOIT == 'sf':
                    action = self.cart.getSoftMaxAction(state)
                if ARGS.EXPLORE_EXPLOIT == 'eg':
                    action = self.cart.getAction(state)

                next_state, reward = self.doAction(action)

                if (ARGS.USE_GUI):
                    self.draw_screen()
                    self.clock.tick(FPS)
                    pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

                accu_reward += reward
                if (ARGS.CYCLIC_SCREEN):
                    if self.cart.body.position[0] < 0:
                        self.cart.add_position(SCREEN_SIZE[0], 0)

                    if self.cart.body.position[0] > SCREEN_SIZE[0]:
                        self.cart.add_position(-SCREEN_SIZE[0], 0)

                self.cart.update(state, action, next_state, reward)
                i += 1

            self.cart.reset()
            print("episode " + str(episode_num) + ": " + str(accu_reward))

            if episode_num % 5 == 0:
                if ARGS.GRAPHS:
                    try:
                        fig = plt.figure("HeatMap: " + str(sys.argv[1:]))
                        plt.clf()
                        plt.title("Heatmap")

                        img = plt.imshow(self.cart.myQ.heatmap,
                                   interpolation='none', aspect='equal',extent=[-7.5,7.5,-math.pi, math.pi])

                        cbar = plt.colorbar(img)
                        cbar.set_label('Q value', rotation=270)

                        plt.xlabel("angular velocity (rad)")
                        plt.ylabel("angle (rad)")
                        plt.pause(0.000000001)
                    except:
                        plt.close()

            sum += accu_reward
            if (episode_num % 100) == 0 :
                t1 = time.time()
                self.accu_rewards.append(sum / 100)
                self.time.append(t1 - t0 + t_start)
                if ARGS.GRAPHS:
                    try:
                        plt.figure("Learning rate " + str(sys.argv))
                        plt.cla()
                        plt.title("Learning rate " + str(sys.argv))
                        plt.title("Epsilon-Greedy: discount = 0.99, alpha = 0.5, no walls, no swing")
                        plt.ylabel("reward")
                        plt.xlabel("time (sec)")
                        plt.plot(self.time, self.accu_rewards)
                        plt.show(block=False)
                        plt.pause(0.000000001)
                    except:
                        plt.close()

                sum = 0
            if episode_num == ARGS.MAX_EPISODES:
                 self.running = False
            if len(self.accu_rewards)>0 and self.accu_rewards[-1] >= ARGS.MAX_REWARD:
                self.running = False
            episode_num += 1

        if ARGS.MODEL_FILE!='':
            pickle.dump((self.cart.myQ,(self.time,self.accu_rewards)), open(ARGS.MODEL_FILE, "wb"))


    def draw_screen(self):
        ### Clear screen
        self.screen.fill(THECOLORS["white"])
        ### Draw stuff
        self.space.debug_draw(self.draw_options)
        ### Flip screen
        pygame.display.flip()

    def doAction(self, action):
        if action == LEFT:
            self.cart.body.velocity += (-Cart.CART_VELOCITY, 0)
        elif action == RIGHT:
            self.cart.body.velocity += (Cart.CART_VELOCITY, 0)
        else:
            self.cart.body.velocity = (0, 0)

        ### Update physics
        dt = 1.0 / DT

        self.space.step(dt)
        next_state = self.cart.getState()
        reward = 0
        for i in range(len(next_state.angles)):
            if next_state.angles[i] > -math.pi / 2 - ANGLE_RANGE and next_state.angles[i] < -math.pi / 2 + ANGLE_RANGE:
                reward += BAD_REWORD
            elif next_state.angles[i] > math.pi / 2 - ANGLE_RANGE and next_state.angles[i] < math.pi / 2 + ANGLE_RANGE:
                reward += GOOD_REWORD - abs(next_state.line_vel[i]) * 0.001
            else:
                reward += DEFAULT_REWORD
        return next_state, reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cart-Pole stabilizing simulation')
    parser.add_argument('--ee',choices=['sf','eg'],default='eg',dest = 'EXPLORE_EXPLOIT',help="The exploration- expoitation method")
    parser.add_argument('-q',choices=['DeepQ','linearQ','Q'],default='Q',dest = 'Q_MODEL', help="The Q has to be used")
    parser.add_argument('--gui',action='store_true' ,dest='USE_GUI',help="Use graphics")
    parser.add_argument('--graphs',action='store_true', dest='GRAPHS',help="Show graphs")
    parser.add_argument('-m',dest = 'MODEL_FILE',default='',help="The model file")
    parser.add_argument('-d',dest='DISCOUNT',type=float,default=0.99,help="The discount parameter")
    parser.add_argument('-a',dest='ALPHA',type=float,default=0.5,help="The alpha parameter")
    parser.add_argument('-e',dest='EPSILON',type=float,default=0.002,help="The epsilon parameter")
    parser.add_argument('--cyclic',dest='CYCLIC_SCREEN',action='store_true',help="Cyclic screen")
    parser.add_argument('--noswing',dest='NO_SWING',action='store_true',help="Pole starts up")
    parser.add_argument('--maxreward',dest='MAX_REWARD',help="maximum accumulative reward to get",type=int,default=math.inf)
    parser.add_argument('--maxepisodes',dest='MAX_EPISODES',help="maximum episodes to run",type=int,default=math.inf)
    parser.add_argument('--maxtime',dest='MAX_TIME',help="maximum time to run in seconds",type=int,default=math.inf)
    args = sys.argv

    ARGS = parser.parse_args(args[1:])


    if ARGS.Q_MODEL == 'DeepQ':
        import testQDeepCart
        testQDeepCart.run()
        exit()
    else:
        env = CarEnvironment()
        sys.exit(env.main())




