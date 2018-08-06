import math, sys, random

from Cart import *

import pygame
from pygame.locals import *
from pygame.color import *

import matplotlib.pyplot as plt
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

import pickle

QDICT_PICKLE_FILE = "q_dict_softmax_noncyc.pkl"
TRAIN_PICKLE_FILE = "train_dict_softmax_noncyc.pkl"


SCREEN_SIZE = (1200,600)

ANGLE_RANGE = 0.02
TOP_VEL = 200


GOOD_REWORD = 1
BAD_REWORD = -1
DEFAULT_REWORD = -0.1


PENDULUM_NUM = 1
PENDULUM_LEN = 200


FPS = 25
DT = 25
EPISODE_LENGTH = 500

USE_GUI = True
CYCLIC_SCREEN = False


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
        self.cart = Cart(self.space, PENDULUM_LEN, PENDULUM_NUM)
        try:
            self.cart.myQ = pickle.load(open(QDICT_PICKLE_FILE, "rb"))
            self.accu_rewards = pickle.load(open(TRAIN_PICKLE_FILE,"rb"))

        except:
            pass

    def main(self):
        episode_num = 0
        while self.running:
            accu_reward = 0
            i=0
            while i < EPISODE_LENGTH and self.running:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                        break
                    elif event.type == KEYDOWN:
                        if  event.key == K_ESCAPE:
                            self.running = False
                            break
                        if event.key == K_a:
                            self.cart.balls[0].velocity += (-100,0)
                        if event.key == K_d:
                            self.cart.balls[0].velocity += (100, 0)

                state = self.cart.getState()

                action = self.cart.getSoftMaxAction(state)
                #action = self.cart.getAction(state)

                next_state, reward = self.doAction(action)

                if(USE_GUI):
                    self.draw_screen()
                    self.clock.tick(FPS)
                    pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

                accu_reward += reward
                if(CYCLIC_SCREEN):
                    if self.cart.body.position[0] < 0:
                        self.cart.add_position(SCREEN_SIZE[0], 0)

                    if self.cart.body.position[0] > SCREEN_SIZE[0]:
                        self.cart.add_position(-SCREEN_SIZE[0], 0)

                for vel in self.cart.getAnglVelocities():
                    if abs(vel) > 5.5:
                        reward = -1
                        break

                #print(reward)
                self.cart.update(state,action,next_state,reward)
                i+=1

            self.cart.reset()
            print("episode "+str(episode_num)+": "+str(accu_reward))
            if episode_num%2 == 0:
                plt.figure(1)
                plt.clf()
                plt.imshow(self.cart.myQ.heatmap,
                           interpolation='none', aspect='equal')
                plt.pause(0.000000001)
            self.accu_rewards.append(accu_reward)
            episode_num += 1


        fig = plt.figure(2)
        plt.plot(self.accu_rewards)
        plt.show()
        pickle.dump(self.cart.myQ, open(QDICT_PICKLE_FILE, "wb"))
        pickle.dump(self.accu_rewards, open(TRAIN_PICKLE_FILE, "wb"))


    def draw_screen(self):
        ### Clear screen
        self.screen.fill(THECOLORS["white"])
        ### Draw stuff
        self.space.debug_draw(self.draw_options)
        ### Flip screen
        pygame.display.flip()

    def doAction(self,action):
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
            if next_state.angles[i] > -math.pi/2-ANGLE_RANGE and next_state.angles[i] < -math.pi/2+ANGLE_RANGE:
                reward += BAD_REWORD
            elif next_state.angles[i] > math.pi/2-ANGLE_RANGE and next_state.angles[i] < math.pi/2+ANGLE_RANGE:
                    #and abs(next_state.line_vel[i]) < TOP_VEL:
                    # and abs(next_state.angular_vel[i]) < TOP_VEL \
                reward += GOOD_REWORD - abs(next_state.line_vel[i])*0.001
                #print(reward)
            else:
                reward += DEFAULT_REWORD
        return next_state, reward


if __name__ == '__main__':
    env = CarEnvironment()
    sys.exit(env.main())