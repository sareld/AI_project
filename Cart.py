from State import State
from Qlearner import *
import numpy as np

import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
from main import SCREEN_SIZE
from main import CYCLIC_SCREEN


# RIGHT = "right"
# LEFT = "left"
# STAY = "stay"


RIGHT = 100
LEFT = -100
STAY = 0

CART_ACTIONS = [LEFT,RIGHT,STAY]

class Cart(Qlearner):

    CART_VELOCITY = 100
    CART_FRICTION = 1.3

    INIT_RAND_RANGE = (-1,1)

    INIT_SIDE = -1

    CART_POS = (SCREEN_SIZE[0]/2, SCREEN_SIZE[1]/2)

    FIRST_POLE_LENGTH = 150
    SECOND_POLE_LENGTH = 150

    CART_POINTS = [(50, -20), (-50, -20), (-50, 20), (50, 20)]


    #CART_ACTIONS = [-100,100,0]

    def getLegalActions(self):
        legalActions = CART_ACTIONS.copy()
        if(CYCLIC_SCREEN):
            return legalActions
        if(self.body.position.x <0):
            if LEFT in legalActions: legalActions.remove(LEFT)
        if (self.body.position.x > SCREEN_SIZE[0]):
            if RIGHT in legalActions: legalActions.remove(RIGHT)

        return legalActions


    def create_cart(self):
        cp = Cart.CART_POINTS
        self.cart_mass = 0.1
        cart_inertia = pymunk.moment_for_poly(self.cart_mass, cp)
        self.body = pymunk.Body(self.cart_mass, cart_inertia, body_type=pymunk.Body.KINEMATIC)
        self.body.position = Cart.CART_POS
        self.shape = pymunk.Poly(self.body, cp)
        self.shape.friction = 4
        self.space.add(self.body, self.shape)

        self.balls = []
        self.joints = []

        for i in range(self.pend_num):
            # ball
            mass = 1
            radius = 10
            moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            ball_body = pymunk.Body(mass, moment)

            ball_body.position = (Cart.CART_POS[0] +
                                  (Cart.INIT_RAND_RANGE[1] - Cart.INIT_RAND_RANGE[0]) * np.random.random() +
                                  Cart.INIT_RAND_RANGE[0],
                                  Cart.CART_POS[1] + Cart.INIT_SIDE * (i + 1) * self.pend_length)
            ball_body.start_position = Vec2d(ball_body.position)
            shape = pymunk.Circle(ball_body, radius)
            self.balls.append(ball_body)
            self.space.add(ball_body, shape)
            if (i == 0):
                j = pymunk.PinJoint(self.balls[0], self.body, (0, 0), (0, 0))
            else:
                j = pymunk.PinJoint(self.balls[i], self.balls[i - 1], (0, 0), (0, 0))
            self.joints.append(j)
            self.space.add(j)

    def __init__(self, space, pend_length, pend_num):
        super(Cart,self).__init__()
        self.space = space
        self.pend_num = pend_num
        self.pend_length = pend_length
        self.create_cart()
        # Cart body


    def getAngles(self):
        angles = []
        for i in range(self.pend_num):
            if (i == 0):
                v = Vec2d(self.balls[0].position.x - self.body.position.x,
                          self.balls[0].position.y - self.body.position.y)
            else:
                v = Vec2d(self.balls[i].position.x - self.balls[i - 1].position.x,
                          self.balls[i].position.y - self.balls[i - 1].position.y)
            angles.append(v.angle)
        return angles

    def getAnglVelocities(self):
        ang_vels = []
        for i in range(self.pend_num):
            if (i == 0):
                r = Vec2d(self.balls[0].position.x - self.body.position.x,
                          self.balls[0].position.y - self.body.position.y)
                v = self.balls[0].velocity - self.body.velocity
            else:
                r = Vec2d(self.balls[i].position.x - self.balls[i - 1].position.x,
                          self.balls[i].position.y - self.balls[i - 1].position.y)
                v = self.balls[i].velocity - self.balls[i-1].velocity
            ang_vel = r.cross(v)/(r.get_length()**2)
            ang_vels.append(ang_vel)
        return ang_vels

    def getLineVelocities(self):
        line_vel = []
        for i in range(self.pend_num):
            line_vel.append(self.balls[i].velocity.x)
        return line_vel

    def getState(self):
        angles = self.getAngles()
        vels = self.getAnglVelocities()
        line_vel = self.getLineVelocities()
        return State(self.body.position.x,self.body.velocity.x,angles,vels,line_vel)

    def remove_cart(self):
        self.space.remove(self.body)
        for j in self.joints:
            self.space.remove(j)
        for ball in self.balls:
            self.space.remove(ball)

    def reset(self):
        self.body.position = Cart.CART_POS
        self.body.velocity = (0,0)
        for i in range(self.pend_num):
            self.balls[i].position = (Cart.CART_POS[0] +
                                  (Cart.INIT_RAND_RANGE[1] - Cart.INIT_RAND_RANGE[0]) * np.random.random() +
                                  Cart.INIT_RAND_RANGE[0],
                                  Cart.CART_POS[1] + Cart.INIT_SIDE * (i + 1) * self.pend_length)

            self.balls[i].velocity = Vec2d(0,0)

    def add_position(self, x, y):
        self.body.position = (self.body.position.x + x,self.body.position.y + y)
        for i in range(self.pend_num):
            self.balls[i].position = (self.balls[i].position.x + x,self.balls[i].position.y+y)




