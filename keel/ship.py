from dataclasses import dataclass, field

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from .keel import Keel
from .rib import Rib
from .util import EndSurface, MonocoqueShell
from .util import calc_util, model_util

from typing import List, Any
from functools import reduce

@dataclass
class Ship:
    keel:Keel = field(default=None)
    ribs:List[Rib] = field(default_factory=list)
    parent:any = field(default=None)
    parents:List[Any] = field(default_factory=list)
    parents_dirty_flag:bool = field(default=False)
    translation_dirty_flag:bool = field(default=False)
    parent_keels_position:float =  field(default=1.)

    smoothing:bool = field(default=False)
    smoothing_from:any = field(default=None)
    smoothing_to:any = field(default=None)

    monocoque_shell:MonocoqueShell = field(default=None)

    subtracts:List[Any] = field(default_factory=list)

    is_visible:bool = field(default=True)

    def length(self):
        return self.keel.length()

    def set_parent(self, parent, position=1.):
        self.parent_keels_position = position
        self.parent = parent
        self.sanitize_keel()
        self.parents = []
        self.get_parents()
        return self

    def get_parents(self):
        self.parents_dirty_flag = False
        if self.parent == None:
            return []
        self.parents = []
        self.parents.extend(self.parent.get_parents())
        self.parents.append(self.parent)
        return self.parents

    def end(self, child):
        child.end = self
        return self

    def relative_vector(self):
        if self.keel is None:
            return None
        return self.keel.end - self.keel.start

    def draw(self):
        if not self.is_visible:
            return
        if self.keel is None:
            return
        if None != self.monocoque_shell:
            self.monocoque_shell.draw(self.keel)
            return
        if self.smoothing:
            if self.smoothing_from is None\
                or self.smoothing_from.ribs is None or len(self.smoothing_from.ribs) == 0\
                or self.smoothing_to is None\
                or self.smoothing_to.ribs is None or len(self.smoothing_to.ribs) == 0:
                return
            rib_from = self.get_rib_end(self.smoothing_from)
            translated_edges_from = []
            for edge in rib_from.edges:
                translated_edge_from = np.dot(np.array([edge[0], edge[1], 0., 1.]), self.smoothing_from.keel.translation(rib_from.position))
                translated_edges_from.append(translated_edge_from)

            rib_to = self.get_rib_start(self.smoothing_to)
            translated_edges_to = []
            for edge in rib_to.edges:
                translated_edge_to = np.dot(np.array([edge[0], edge[1], 0., 1.]), self.smoothing_to.keel.translation(rib_to.position))
                translated_edges_to.append(translated_edge_to)
            Rib.draw_beam(translated_edges_from, translated_edges_to)
        else:
            former_rib_edges = None
            for rib in self.ribs:
                former_rib_edges = rib.draw(self.keel, former_rib_edges)
    
    def write_stl(self, f):
        if not self.is_visible:
            return
        if self.keel is None:
            return
        if None != self.monocoque_shell:
            self.monocoque_shell.write_stl(self.keel, f)
            return
        if self.smoothing:
            pass
            if self.smoothing_from is None\
                or self.smoothing_from.ribs is None or len(self.smoothing_from.ribs) == 0\
                or self.smoothing_to is None\
                or self.smoothing_to.ribs is None or len(self.smoothing_to.ribs) == 0:
                return
            rib_from = self.get_rib_end(self.smoothing_from)
            translated_edges_from = []
            for edge in rib_from.edges:
                translated_edge_from = np.dot(np.array([edge[0], edge[1], 0., 1.]), self.smoothing_from.keel.translation(rib_from.position))
                translated_edges_from.append(translated_edge_from)

            rib_to = self.get_rib_start(self.smoothing_to)
            translated_edges_to = []
            for edge in rib_to.edges:
                translated_edge_to = np.dot(np.array([edge[0], edge[1], 0., 1.]), self.smoothing_to.keel.translation(rib_to.position))
                translated_edges_to.append(translated_edge_to)
            
            end_surface = EndSurface().rib_to_vectors(rib_to)
            Rib.write_stl_inter_edges(translated_edges_from, translated_edges_to, end_surface.is_clockwise, f)
        else:
            if len(self.ribs) == 0:
                return
            rib_start = self.get_rib_start(self)
            rib_start.write_stl_start(self.keel, f)
            rib_end = self.get_rib_end(self)
            rib_end.write_stl_end(self.keel, f)
            former_rib_edges = None
            for rib in self.ribs:
                former_rib_edges = rib.write_stl_beam(self.keel, former_rib_edges, f)

    def init_keel(self):
        self.keel = Keel()
        return self.keel
    
    def sanitize_keel(self):
        self.keel.set_start(self.parent.keel.translation(self.parent_keels_position))

    def add_rib(self, position=0., edges=None):
        new_rib = Rib()
        new_rib.position = position
        new_rib.edges = edges
        self.ribs.append(new_rib)

    def order_ribs(self):
        self.ribs.sort(key=lambda rib: rib.position)
    
    def set_smoothing(self, ship_smoothing_from, ship_smoothing_to=None):
        self.smoothing = True
        self.smoothing_from = ship_smoothing_from
        self.smoothing_to = ship_smoothing_to
    
    def set_smoothing_to(self, ship_smoothing_to):
        self.smoothing = True
        self.smoothing_to = ship_smoothing_to

    def get_rib_end(self, ship):
        rib_end = ship.ribs[-1]
        for rib in ship.ribs:
            if rib.position > rib_end.position:
                rib_end = rib
        return rib_end

    def get_rib_start(self, ship):
        rib_start = ship.ribs[0]
        for rib in ship.ribs:
            if rib.position < rib_start.position:
                rib_start = rib
        return rib_start

    def convert_to_monocoque(self):
        if self.keel is None:
            return
        # smoothingへは未対応
        if len(self.ribs) == 0:
            return
        self.order_ribs()

        self.monocoque_shell = MonocoqueShell()
        
        rib_start = self.ribs[0]
        rib_end = self.ribs[-1]
        
        former_rib_positions = None
        for index, rib in enumerate(self.ribs):
            if index == 0:
                continue
            positions_end = None
            if index == 1:
                former_rib_positions = rib_start.generate_monocoque_shell_start(
                    self.monocoque_shell, self.keel.length * rib_start.position)
            if index == len(self.ribs) - 1:
                positions_end = rib_end.generate_monocoque_shell_end(
                    self.monocoque_shell, self.keel.length * rib_end.position)

            former_rib_positions = rib.generate_monocoque_shells_beam(
                self.monocoque_shell, self.keel, former_rib_positions, positions_end)
    
    def is_monocoque(self):
        return None != self.monocoque_shell
    
    def apply_subtructions(self):
        if not self.is_monocoque():
            self.convert_to_monocoque()
        for subtract_ship in self.subtracts:
            subtract_ship.is_visible = False
            if not subtract_ship.is_monocoque():
                subtract_ship.convert_to_monocoque()
            relative_translation = subtract_ship.keel.relative_translation.copy()
            if not reduce(lambda x, y: x or y, map(lambda x: x is self, subtract_ship.parents)):
                relative_translation_to_self = self.keel.relative_translation.copy()
                common_ancestors = [x for x in self.parents if x in subtract_ship.parents]
                if 0 == len(common_ancestors):
                    self.calc_relative_translation_to_ancestor(subtract_ship, None, relative_translation)
                    self.calc_relative_translation_to_ancestor(self, None, relative_translation_to_self)
                else:
                    common_ancestors.sort(key=lambda ship: len(ship.parents))
                    common_ancestor = common_ancestors[-1]
                    self.calc_relative_translation_to_ancestor(subtract_ship, common_ancestor, relative_translation)
                    self.calc_relative_translation_to_ancestor(self, common_ancestor, relative_translation_to_self)
                relative_translation = np.dot(relative_translation, np.linalg.inv(relative_translation_to_self))
            else:
                self.calc_relative_translation_to_ancestor(subtract_ship, self, relative_translation)
            self.subtract(subtract_ship, relative_translation)
    
    def calc_relative_translation_to_ancestor(self, ship, ship_ancestor, translation):
        if ship.parent is None:
                return
        translation[:] = np.dot(translation, np.dot(ship.parent.keel.translation_z(ship.parent_keels_position), ship.parent.keel.relative_translation))
        if ship.parent is ship_ancestor:
            return
        else:
            self.calc_relative_translation_to_ancestor(ship.parent, ship_ancestor, translation)

    def subtract(self, subtraction, translation):
        subtraction.monocoque_shell.translate(translation)
        subtraction.monocoque_shell.copy_translated_to_default()

        self_line_segments = self.monocoque_shell.generate_line_segments()
        subtraction_line_segments = subtraction.monocoque_shell.generate_line_segments()

        penetrations_self = model_util.calc_penetration(self.monocoque_shell.triangles, subtraction_line_segments)
        penetrations_subtrantion =  model_util.calc_penetration(subtraction.monocoque_shell.triangles, self_line_segments)

        penetrating_triangles_self = model_util.fetch_penetrating_triangles(penetrations_subtrantion)
        penetrating_triangles_subtrantion = model_util.fetch_penetrating_triangles(penetrations_self)

        subdivided_self = model_util.subdivide_penetrated_faces(penetrations_self, penetrations_subtrantion)
        subdivided_subtraction = model_util.subdivide_penetrated_faces(penetrations_subtrantion, penetrations_self)

        self.monocoque_shell.triangles = list(x for x in self.monocoque_shell.triangles if not any(x is y for y in penetrating_triangles_self))
        subtraction.monocoque_shell.triangles = list(x for x in subtraction.monocoque_shell.triangles if not any(x is y for y in penetrating_triangles_subtrantion))
        
        self.monocoque_shell.triangles.extend(subdivided_self)
        subtraction.monocoque_shell.triangles.extend(subdivided_subtraction)

        add_penetration_positions = [x.position for x in penetrations_self]#for test
        add_penetration_positions.extend([x.position for x in penetrations_subtrantion])
        self.monocoque_shell.positions.extend(add_penetration_positions)

        self_inside_positions, self_outside_positions = model_util.fetch_positions_inout(penetrations_subtrantion)

        self_inside_positions.extend(
            model_util.fetch_positions_not_affected_penetrations(
                self_inside_positions, self_line_segments, penetrations_subtrantion))
        self_outside_positions.extend(
            model_util.fetch_positions_not_affected_penetrations(
                self_outside_positions, self_line_segments, penetrations_subtrantion))

        inner_triangles_self = []
        outer_triangles_self = []
        inner_triangles_self.extend(
            model_util.fetch_triangles_contains_two_positions(self_inside_positions, self.monocoque_shell.triangles))
        outer_triangles_self.extend(
            model_util.fetch_triangles_contains_two_positions(self_outside_positions, self.monocoque_shell.triangles))

        for self_inside_position in self_inside_positions:
            inner_triangles_self.extend(
                list(x for x in model_util.fetch_triangles_contains_position(self_inside_position, self.monocoque_shell.triangles, exclude=outer_triangles_self)\
                    if not any(y is x for y in inner_triangles_self)))
        for self_outside_position in self_outside_positions:
            outer_triangles_self.extend(
                list(x for x in model_util.fetch_triangles_contains_position(self_outside_position, self.monocoque_shell.triangles, exclude=inner_triangles_self)\
                    if not any(y is x for y in outer_triangles_self)))


        self_inside_pair, self_outside_pair = model_util.fetch_position_pair_inout(penetrations_subtrantion)
        
        for self_inside in self_inside_pair:
            inner_triangles_self.extend(
                list(x for x in model_util.fetch_triangles_contains_positions_pair(
                    self_inside[0], self_inside[1], self.monocoque_shell.triangles, exclude=outer_triangles_self)\
                        if not any(y is x for y in inner_triangles_self)))
        for self_outside in self_outside_pair:
            outer_triangles_self.extend(
                list(x for x in model_util.fetch_triangles_contains_positions_pair(
                    self_outside[0], self_outside[1], self.monocoque_shell.triangles, exclude=inner_triangles_self)\
                        if not any(y is x for y in outer_triangles_self)))
        
        inner_triangles_self_touched = []
        inner_triangles_self_touched_previous_count = 0
        inner_triangles_self_previous_count = 0
        while inner_triangles_self_previous_count != len(inner_triangles_self) \
            or inner_triangles_self_touched_previous_count != len(inner_triangles_self_touched):
            inner_triangles_self_touched_previous_count = len(inner_triangles_self_touched)
            inner_triangles_self_previous_count = len(inner_triangles_self)
            touched = list(x for x in self.monocoque_shell.triangles if any(y.is_touched_side(x) for y in inner_triangles_self))
            touched_exclude_outer = list(x for x in touched if not any(y is x for y in outer_triangles_self))
            inner_triangles_self_touched.extend(list(x for x in touched_exclude_outer if not any(y is x for y in inner_triangles_self_touched)))
            inner_triangles_self.extend(list(x for x in inner_triangles_self_touched if not any(y is x for y in inner_triangles_self)))
        
        self.monocoque_shell.triangles = list(x for x in self.monocoque_shell.triangles if not any(y is x for y in inner_triangles_self))

        subtraction_inside_positions, subtraction_outside_positions = model_util.fetch_positions_inout(penetrations_self)

        self_inside_positions.extend(
            model_util.fetch_positions_not_affected_penetrations(
                subtraction_inside_positions, subtraction_line_segments, penetrations_self))
        self_outside_positions.extend(
            model_util.fetch_positions_not_affected_penetrations(
                subtraction_outside_positions, subtraction_line_segments, penetrations_self))

        inner_triangles_subtraction = []
        outer_triangles_subtraction = []
        inner_triangles_subtraction.extend(
            model_util.fetch_triangles_contains_position(subtraction_inside_positions, subtraction.monocoque_shell.triangles))
        outer_triangles_subtraction.extend(
            model_util.fetch_triangles_contains_position(subtraction_outside_positions, subtraction.monocoque_shell.triangles))
        
        for subtraction_inside_position in subtraction_inside_positions:
            inner_triangles_subtraction.extend(
                list(x for x in model_util.fetch_triangles_contains_position(subtraction_inside_position, subtraction.monocoque_shell.triangles, exclude=outer_triangles_subtraction)\
                    if not any(y is x for y in inner_triangles_subtraction)))
        for subtraction_outside_position in subtraction_outside_positions:
            outer_triangles_subtraction.extend(
                list(x for x in model_util.fetch_triangles_contains_position(subtraction_outside_position, subtraction.monocoque_shell.triangles, exclude=inner_triangles_subtraction)\
                    if not any(y is x for y in outer_triangles_subtraction)))

        subtraction_inside_pair, subtraction_outside_pair = model_util.fetch_position_pair_inout(penetrations_self)
        
        for subtraction_inside in subtraction_inside_pair:
            inner_triangles_subtraction.extend(
                list(x for x in model_util.fetch_triangles_contains_positions_pair(
                    subtraction_inside[0], subtraction_inside[1], subtraction.monocoque_shell.triangles, exclude=outer_triangles_subtraction)\
                        if not any(y is x for y in inner_triangles_subtraction)))
        
        for subtraction_outside in subtraction_outside_pair:
            outer_triangles_subtraction.extend(
                list(x for x in model_util.fetch_triangles_contains_positions_pair(
                    subtraction_outside[0], subtraction_outside[1], subtraction.monocoque_shell.triangles, exclude=inner_triangles_subtraction)\
                        if not any(y is x for y in outer_triangles_subtraction)))

        inner_triangles_subtraction_touched = []
        inner_triangles_subtraction_touched_previous_count = 0
        inner_triangles_subtraction_previous_count = 0
        while inner_triangles_subtraction_previous_count != len(inner_triangles_subtraction) \
            or inner_triangles_subtraction_touched_previous_count != len(inner_triangles_subtraction_touched):
            inner_triangles_subtraction_touched_previous_count = len(inner_triangles_subtraction_touched)
            inner_triangles_subtraction_previous_count = len(inner_triangles_subtraction)
            touched = list(x for x in subtraction.monocoque_shell.triangles if any(y.is_touched_side(x) for y in inner_triangles_subtraction))
            touched_exclude_outer = list(x for x in touched if not any(y is x for y in outer_triangles_subtraction))
            inner_triangles_subtraction_touched.extend(list(x for x in touched_exclude_outer if not any(y is x for y in inner_triangles_subtraction_touched)))
            inner_triangles_subtraction.extend(list(x for x in inner_triangles_subtraction_touched if not any(y is x for y in inner_triangles_subtraction)))
        
        for inner_triangle_subtraction in inner_triangles_subtraction:
            inner_triangle_subtraction.inverse()
        
        self.monocoque_shell.triangles.extend(inner_triangles_subtraction)
        
        #temp
        self.monocoque_shell.positions.extend(subtraction.monocoque_shell.positions)
        # positionの追加を整理して完成positionの追加を整理して完成

        
        

        




    

        
