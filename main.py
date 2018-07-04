import math, sys, random

from Cart import *

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
from pymunk import Vec2d
import pymunk.pygame_util




CART_VELOCITY = 60
CART_FRICTION = 1.3

CART_POS = 300

FIRST_POLE_LENGTH = 150
SECOND_POLE_LENGTH = 150

FPS = 25


def draw_collision(arbiter, space, data):
    for c in arbiter.contact_point_set.points:
        r = max(3, abs(c.distance * 5))
        r = int(r)

        p = pymunk.pygame_util.to_pygame(c.point_a, data["surface"])
        pygame.draw.circle(data["surface"], THECOLORS["black"], p, r, 1)


def main():
    global contact
    global shape_to_remove

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    clock = pygame.time.Clock()
    running = True

    ### Physics stuff
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    # disable the build in debug draw of collision point since we use our own code.
    draw_options.flags = draw_options.flags ^ pymunk.pygame_util.DrawOptions.DRAW_COLLISION_POINTS
    ## Balls
    balls = []

    ### walls
    static_lines = []#[pymunk.Segment(space.static_body, (10, 280.0), (10, 350.0), 0.0),
                    #pymunk.Segment(space.static_body, (590, 280.0), (590, 350.0), 0.0)]
    for l in static_lines:
        l.friction = 0.5
    space.add(static_lines)

    ticks_to_next_ball = 10

    ch = space.add_collision_handler(0, 0)
    ch.data["surface"] = screen
    ch.post_solve = draw_collision

    cart = Cart(space,200,1)

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if  event.key == K_ESCAPE:
                    running = False

        action = cart.getAction(cart.getState())
        if action == LEFT:
            cart.body.velocity += (-CART_VELOCITY,0)
        elif action == RIGHT:
            cart.body.velocity += (CART_VELOCITY,0)
        else:
            cart.body.velocity = (0, 0)

        ### Clear screen
        screen.fill(THECOLORS["white"])

        ### Draw stuff
        space.debug_draw(draw_options)

        ### Update physics
        dt = 1.0 / FPS

        space.step(dt)

        ### Flip screen
        pygame.display.flip()
        clock.tick(FPS)
        pygame.display.set_caption("fps: " + str(clock.get_fps()))
        #print(cart.getAngles())
        print(cart.getState())


if __name__ == '__main__':
    sys.exit(main())