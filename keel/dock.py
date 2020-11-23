from dataclasses import dataclass, field

import threading

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from .ship import Ship

from typing import List

@dataclass
class Dock:
    ships:List[Ship] = field(default_factory=list)

    translate_vel=0.1
    rotete_radian_vel=np.pi/180

    def draw(self):
        if self.ships is None or len(self.ships) == 0:
            return
        for ship in self.ships:
            ship.draw()

    def start_display(self):
        self.sanitize_dock()

        pygame.init()
        pygame.display.set_caption('KEEL')

        display=(800, 800)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        self.init_gl_view(display)

        model_view_matrix = np.matrix(np.identity(4), copy=False, dtype='float32')
        glGetFloatv(GL_MODELVIEW_MATRIX, model_view_matrix)

        rotate_matlix = np.matrix(np.identity(4), copy=False, dtype='float32')
        translate_matlix = self.generate_translate_matrix()

        glMultMatrixf(rotate_matlix)
        glMultMatrixf(translate_matlix)
        
        clock = pygame.time.Clock()
        while True:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            keys = pygame.key.get_pressed()
            if len(keys) != 0:
                if keys[pygame.K_LEFT]:
                    rotate_matlix = np.dot(rotate_matlix, self.rotate_matrix_y(-self.rotete_radian_vel))
                if keys[pygame.K_RIGHT]:
                    rotate_matlix = np.dot(rotate_matlix, self.rotate_matrix_y(self.rotete_radian_vel))
                if keys[pygame.K_UP]:
                    rotate_matlix = np.dot(rotate_matlix, self.rotate_matrix_x(-self.rotete_radian_vel))
                if keys[pygame.K_DOWN]:
                    rotate_matlix = np.dot(rotate_matlix, self.rotate_matrix_x(self.rotete_radian_vel))
                if keys[pygame.K_g]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(0, 0, self.translate_vel))
                if keys[pygame.K_t]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(0, 0, -self.translate_vel))
                if keys[pygame.K_w]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(0, self.translate_vel, 0))
                if keys[pygame.K_s]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(0, -self.translate_vel, 0))
                if keys[pygame.K_a]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(-self.translate_vel, 0, 0))
                if keys[pygame.K_d]:
                    translate_matlix = np.dot(translate_matlix, self.translate_matrix(self.translate_vel, 0, 0))
                if keys[pygame.K_SPACE]:
                    rotate_matlix = np.identity(4)
                    translate_matlix = self.generate_translate_matrix()
                self.init_gl_view(display)
                glMultMatrixf(translate_matlix)
                glMultMatrixf(rotate_matlix)
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            self.draw()
            pygame.display.flip()

    def quit_display(self):
        pygame.quit()

    def init_gl_view(self, display):
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 500)

    def translate_matrix(self, x, y, z):
        return np.array([\
            [1., 0., 0., 0.],\
            [0., 1., 0., 0.],\
            [0., 0., 1., 0.],\
            [x, y, z, 1.]])

    def rotate_matrix_x(self, theta):
        return np.array([\
            [1., 0., 0., 0.],\
            [0., np.cos(theta), np.sin(theta), 0.],\
            [0., -np.sin(theta), np.cos(theta), 0.],\
            [0., 0., 0., 1.]])

    def rotate_matrix_y(self, theta):
        return np.array([\
            [np.cos(theta), 0., -np.sin(theta), 0.],\
            [0., 1., 0., 0.],\
            [np.sin(theta), 0., np.cos(theta), 0.],\
            [0., 0., 0., 1.]])

    def generate_ship(self):
        ship = Ship()

        self.ships.append(ship)
        ship.init_keel()
        return ship

    def rotate_keel(self, ship, y_axis_rotate=0., z_axis_rotate=0.):
        ship.keel.rotation(y_axis_rotate, z_axis_rotate)
        self.make_translation_dirty_recursively(ship)
    
    def resize_keel(self, ship, keel_length):
        ship.keel.length = keel_length
        self.make_translation_dirty_recursively(ship)
    
    def set_parent(self, ship ,parent, position=1.):
        ship.set_parent(parent, position)
        self.make_parents_dirty_recursively(ship)
        self.make_translation_dirty_recursively(ship)

    def set_parents_position(self, ship, position=1.):
        ship.set_parent(ship.parent, position)
        self.make_translation_dirty_recursively(ship)

    def make_parents_dirty_recursively(self, ship):
        for target_ship in self.ships:
            for parent in target_ship.parents:
                if ship is parent:
                    target_ship.parents_dirty_flag = True

    def make_translation_dirty_recursively(self, ship):
        for target_ship in self.ships:
            for parent in target_ship.parents:
                if ship is parent:
                    target_ship.translation_dirty_flag = True

    def sanitize_dock(self, force=False):
        for target_ship in reversed(self.ships):
            if force or target_ship.parents_dirty_flag:
                target_ship.get_parents()
        self.ships.sort(key=lambda ship: len(ship.parents))

        for target_ship in self.ships:
            if target_ship.parent != None and (force or target_ship.translation_dirty_flag):
                target_ship.sanitize_keel()
        
        for target_ship in self.ships:
            if 0 != len(target_ship.subtracts):
                target_ship.apply_subtructions()

    def generate_translate_matrix(self):
        translate_matlix = np.matrix(np.identity(4), copy=False, dtype='float32')
        jack_up_ratio = 2.

        object_areas = self.get_object_areas()
        max_area_length = object_areas[0]-object_areas[1]
        if (max_area_length < object_areas[2]-object_areas[3]):
            max_area_length = object_areas[2]-object_areas[3]
        if (max_area_length < object_areas[4]-object_areas[5]):
            max_area_length = object_areas[4]-object_areas[5]
        self.translate_vel = max_area_length/20
        translate_matlix = np.dot(\
            translate_matlix, \
            self.translate_matrix(\
                -(object_areas[0]+object_areas[1])/2, \
                -(object_areas[2]+object_areas[3])/2, \
                -((object_areas[4]+object_areas[5])/2 + max_area_length * jack_up_ratio)))
        return translate_matlix

    def get_object_areas(self):
        right_max = 0.
        left_max = 0.
        top_max = 0.
        bottom_max = 0.
        front_max = 0.
        back_max = 0.
        for target_ship in self.ships:
            keel_translation = target_ship.keel.translation(1.)
            if (keel_translation[3][0] < 0.):
                if keel_translation[3][0] < left_max:
                    left_max = keel_translation[3][0]
            else:
                if keel_translation[3][0] > right_max:
                    right_max = keel_translation[3][0]

            if (keel_translation[3][1] < 0.):
                if keel_translation[3][1] < bottom_max:
                    bottom_max = keel_translation[3][1]
            else:
                if keel_translation[3][1] > top_max:
                    top_max = keel_translation[3][1]
            
            if (keel_translation[3][2] < 0.):
                if keel_translation[3][2] < back_max:
                    back_max = keel_translation[3][2]
            else:
                if keel_translation[3][2] > front_max:
                    front_max = keel_translation[3][2]
            
        return (right_max, left_max, top_max, bottom_max, front_max, back_max)

    def write_stl(self, f):
        self.sanitize_dock()
        if self.ships is None or len(self.ships) == 0:
            return
        for ship in self.ships:
            ship.write_stl(f)
