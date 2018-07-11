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


PICKLE_FILE = "pickle_q.pkl"

CART_VELOCITY = 60
CART_FRICTION = 1.3

CART_POS = 300

FIRST_POLE_LENGTH = 150
SECOND_POLE_LENGTH = 150

SCREEN_SIZE = (1200,600)

ANGLE_RANGE = 0.1

GOOD_REWORD = 2
BAD_REWORD = -1

FPS = 100

DT = 25

EPISODE_LENGTH = 1000

USE_GUI = False

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

        self.cart = Cart(self.space, 200, 1)
        try:
            self.cart.myQ = pickle.load(open(PICKLE_FILE, "rb"))
        except:
            pass

    def main(self):
        episode_num = 0
        accu_rewards = []
        while self.running:
            accu_reward = 0
            for i in range(EPISODE_LENGTH):
                for event in pygame.event.get():
                    if event.type == QUIT:
                        self.running = False
                    elif event.type == KEYDOWN:
                        if  event.key == K_ESCAPE:
                            self.running = False

                state = self.cart.getState()

                action = self.cart.getAction(state)

                next_state, reward = self.doAction(action)

                if(USE_GUI):
                    self.draw_screen()

                accu_reward += reward
                if self.cart.body.position[0] < 0:
                    self.cart.add_position(SCREEN_SIZE[0], 0)

                if self.cart.body.position[0] > SCREEN_SIZE[0]:
                    self.cart.add_position(-SCREEN_SIZE[0], 0)

                for vel in self.cart.getVelocities():
                    if abs(vel) > 5.5:
                        reward = -1
                        break

                #print(reward)
                self.cart.update(state,action,next_state,reward)

                self.clock.tick(FPS)
                pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

            self.cart.reset()
            print("episode "+str(episode_num)+": "+str(accu_reward))
            accu_rewards.append(accu_reward)
            episode_num += 1
        plt.plot(accu_rewards)
        plt.show()
        pickle.dump(self.cart.myQ,open(PICKLE_FILE, "wb" ))


    def draw_screen(self):
        ### Clear screen
        self.screen.fill(THECOLORS["white"])
        ### Draw stuff
        self.space.debug_draw(self.draw_options)
        ### Flip screen
        pygame.display.flip()



    def doAction(self,action):
        state = self.cart.getState()
        if action == LEFT:
            self.cart.body.velocity += (-CART_VELOCITY, 0)
        elif action == RIGHT:
            self.cart.body.velocity += (CART_VELOCITY, 0)
        else:
            self.cart.body.velocity = (0, 0)



        ### Update physics
        dt = 1.0 / DT

        self.space.step(dt)
        next_state = self.cart.getState()
        reward = 0
        for i in range(len(next_state)):
            if next_state[i][0] > -math.pi/2-ANGLE_RANGE and next_state[i][0] < -math.pi/2+ANGLE_RANGE:
                reward = BAD_REWORD
            elif next_state[i][0] > math.pi/2-ANGLE_RANGE and next_state[i][0] < math.pi/2+ANGLE_RANGE \
                    and abs(next_state[i][1]) < 1:
                reward = GOOD_REWORD
        return next_state, reward


if __name__ == '__main__':
    env = CarEnvironment()
    sys.exit(env.main())