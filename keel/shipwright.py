from dataclasses import dataclass, field
import numpy as np
import os

from .ship import Ship
from .dock import Dock

@dataclass
class Shipwright:
    dock:Dock
    cached_parent:Ship = field(default=None)
    cached_parent_position:float = field(default=1.)
    cached_rotate_y_axis:float = field(default=0.)
    cached_rotate_z_axis:float = field(default=0.)

    def get_dock(self):
        return self.dock
    
    def start_display(self):
        self.dock.start_display()

    def parent(self, ship, position=1.):
        self.cached_parent = ship
        self.cached_parent_position = position
        return self
    
    def rotate(self, rotate_y_axis=0., rotate_z_axis=0.):
        self.cached_rotate_y_axis = rotate_y_axis
        self.cached_rotate_z_axis = rotate_z_axis
        return self

    def set_parent(self, ship, parent, position=1.):
        self.dock.set_parent(ship, parent, position)
    
    def set_cached_parameter(self, ship):
        if self.cached_parent != None:
            self.set_parent(ship, self.cached_parent, self.cached_parent_position)
            self.cached_parent_position = 1.
            self.cached_parent = None
        
        if self.cached_rotate_y_axis != 0. \
            or self.cached_rotate_z_axis != 0.:
            self.dock.rotate_keel(ship, \
                self.cached_rotate_y_axis, self.cached_rotate_z_axis)
            self.cached_rotate_y_axis = 0.
            self.cached_rotate_z_axis = 0.
        
        return ship

    def set_smoothing(self, smoothing, smoothing_from, smoothing_to=None):
        smoothing.set_smoothing(smoothing_from, smoothing_to)
    
    def set_smoothing_to(self, smoothing, smoothing_to):
        smoothing.set_smoothing_to(smoothing_to)

    def void(self, length=0.):
        void = self.dock.generate_ship()
        self.dock.resize_keel(void, length)
        return self.set_cached_parameter(void)
    
    def cube(self, length):
        return self.rectangular(length, length, length)

    def rectangular(self, width, height, depth):
        ship = self.dock.generate_ship()
        ship.add_rib(0, [(width/2, height/2), (width/2, -height/2), (-width/2, -height/2), (-width/2, height/2)])
        ship.add_rib(1, [(width/2, height/2), (width/2, -height/2), (-width/2, -height/2), (-width/2, height/2)])
        self.dock.resize_keel(ship, depth)
        return self.set_cached_parameter(ship)
    
    def rib_edges_circular(self, radius, arc_central_angle, division, closed=False):
        division = division if closed else division - 1
        edges = []
        for i in range(division):
            theta = i * arc_central_angle / division
            edges.append((np.cos(theta) * radius, np.sin(theta) * radius))
        if closed:
            edges.append((np.cos(arc_central_angle) * radius, np.sin(arc_central_angle) * radius))
        return edges
    
    def pole(self, depth, radius, arc_central_angle, division, closed=False):
        ship = self.dock.generate_ship()
        rib_edges = self.rib_edges_circular(radius, arc_central_angle, division, closed)
        ship.add_rib(0., rib_edges)
        ship.add_rib(1., rib_edges)
        self.dock.resize_keel(ship, depth)
        return self.set_cached_parameter(ship)
    
    def sphere(self, radius, equatorial_division, step, pole_visibility=False):
        return self.spheroid(radius * 2, radius, equatorial_division, step, pole_visibility)

    def spheroid(self, depth, \
        radius, equatorial_division, \
        step, pole_visibility=False):
        step = step - 1 if pole_visibility else step + 1
        ship = self.dock.generate_ship()
        for i in range(step):
            if not pole_visibility and i == 0:
                continue
            ship.add_rib(\
                i * 1. / step , \
                self.rib_edges_circular(\
                    radius * np.sqrt(1 - np.square(i * 2. / step - 1)), \
                    2 * np.pi, equatorial_division, True))
        if pole_visibility:
            ship.add_rib(1., self.rib_edges_circular(\
                0, 2 * np.pi, equatorial_division, True))
        self.dock.resize_keel(ship, depth)
        return self.set_cached_parameter(ship)

    # todo: 1回転しない場合(半回転など)は未実装
    def spin(self, edges, radius, division):
        base = self.dock.generate_ship()
        self.dock.resize_keel(base, 0.)
        keel_length = 2* radius * np.sin(2. * np.pi / (division * 2) / 2.)

        rims = []
        for i in range(division):
            goto_rim = self.dock.generate_ship()
            self.dock.rotate_keel(goto_rim, np.pi/2, 2. * np.pi *  -i / division)
            self.dock.resize_keel(goto_rim, radius)
            self.dock.set_parent(goto_rim, base)

            rotate_z_axis_joint = self.dock.generate_ship()
            self.dock.rotate_keel(rotate_z_axis_joint, 0., np.pi/2)
            self.dock.resize_keel(rotate_z_axis_joint, 0.)
            self.dock.set_parent(rotate_z_axis_joint, goto_rim)
            
            rim = self.dock.generate_ship()
            rim.add_rib(0., edges)
            rim.add_rib(1., edges)
            self.dock.rotate_keel(rim, -np.pi * (1. + 1 / (division * 2)) / 2., 0.)
            self.dock.resize_keel(rim, keel_length)
            self.dock.set_parent(rim, rotate_z_axis_joint)
            rims.append(rim)
        
        for i in range(len(rims)):
            rim_smoothing = self.dock.generate_ship()
            rim_smoothing.set_smoothing(rims[i])
            self.dock.set_parent(rim_smoothing, rims[i])
            if i == len(rims) - 1:
                rim_smoothing.set_smoothing_to(rims[0])
            else:
                rim_smoothing.set_smoothing_to(rims[i+1])

        return self.set_cached_parameter(base)

    
    def generate_stl(self, path, fname):
        file_full_name = os.path.join(path, fname)
        f = open(file_full_name, "w", encoding="ascii")
        f.write("solid \n")
        self.dock.write_stl(f)
        f.write("endsolid ")
        f.close()




