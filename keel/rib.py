from dataclasses import dataclass, field

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from typing import List

from .facet import Facet
from .util import EndSurface

@dataclass
class Rib:

    edges: list = field(default_factory=list)
    position: float = field(default=0.)

    def draw(self, keel, former_rib_edges):
        if self.edges is None or len(self.edges) == 0:
            return None
        glLineWidth(1)
        glBegin(GL_LINE_STRIP)
        return_edges = []
        glColor3f(0,1,0)
        for edge in self.edges:
            translated_edge = np.dot(np.array([edge[0], edge[1], 0., 1.]), keel.translation(self.position))
            glVertex3fv((translated_edge[0], translated_edge[1], translated_edge[2]))
            return_edges.append(translated_edge)
        glVertex3fv((return_edges[0][0], return_edges[0][1], return_edges[0][2]))
        glEnd()
        if former_rib_edges is not None \
            and len(former_rib_edges) != 0 \
            and len(return_edges) != 0:
            Rib.draw_beam(former_rib_edges, return_edges)
        return return_edges
    
    def write_stl_beam(self, keel, former_rib_edges, f):
        edges_count = len(self.edges)
        if self.edges is None or edges_count <= 2:
            return None
        return_edges = []
        for edge in self.edges:
            translated_edge = np.dot(np.array([edge[0], edge[1], 0., 1.]), keel.translation(self.position))
            return_edges.append(translated_edge)
        if former_rib_edges is not None \
            and len(former_rib_edges) != 0 \
            and len(return_edges) != 0:
            end_surface = EndSurface().rib_to_vectors(self.edges)
            Rib.write_stl_inter_edges(former_rib_edges, return_edges, end_surface.is_clockwise, f)
        return return_edges

    def write_stl_start(self, keel, f):
        edges_count = len(self.edges)
        if edges_count <= 2:
            return
        else:
            end_surface = EndSurface().rib_to_vectors(self.edges)
            if end_surface.is_clockwise:
                for i in range(edges_count - 2):
                    facet = Facet()
                    facet.vertex_1 = np.array([self.edges[0][0], self.edges[0][1], 0., 1.])
                    facet.vertex_2 = np.array([self.edges[i+2][0], self.edges[i+2][1], 0., 1.])
                    facet.vertex_3 = np.array([self.edges[i+1][0], self.edges[i+1][1], 0., 1.])
                    facet.translation(keel.translation(self.position))
                    facet.calc_normal()
                    facet.write(f)
            else:
                for i in range(edges_count - 2):
                    facet = Facet()
                    facet.vertex_1 = np.array([self.edges[0][0], self.edges[0][1], 0., 1.])
                    facet.vertex_2 = np.array([self.edges[i+1][0], self.edges[i+1][1], 0., 1.])
                    facet.vertex_3 = np.array([self.edges[i+2][0], self.edges[i+2][1], 0., 1.])
                    facet.translation(keel.translation(self.position))
                    facet.calc_normal()
                    facet.write(f)

    def write_stl_end(self, keel, f):
        edges_count = len(self.edges)
        if edges_count <= 2:
            return
        else:
            end_surface = EndSurface().rib_to_vectors(self.edges)
            if end_surface.is_clockwise:
                for i in range(edges_count - 2):
                    facet = Facet()
                    facet.vertex_1 = np.array([self.edges[0][0], self.edges[0][1], 0., 1.])
                    facet.vertex_2 = np.array([self.edges[i+1][0], self.edges[i+1][1], 0., 1.])
                    facet.vertex_3 = np.array([self.edges[i+2][0], self.edges[i+2][1], 0., 1.])
                    facet.translation(keel.translation(self.position))
                    facet.calc_normal()
                    facet.write(f)
            else:
                for i in range(edges_count - 2):
                    facet = Facet()
                    facet.vertex_1 = np.array([self.edges[0][0], self.edges[0][1], 0., 1.])
                    facet.vertex_2 = np.array([self.edges[i+2][0], self.edges[i+2][1], 0., 1.])
                    facet.vertex_3 = np.array([self.edges[i+1][0], self.edges[i+1][1], 0., 1.])
                    facet.translation(keel.translation(self.position))
                    facet.calc_normal()
                    facet.write(f)

    @staticmethod
    def draw_beam(former_rib_edges, edges):
        edge_max = len(edges)
        if edge_max < len(former_rib_edges):
            edge_max = len(former_rib_edges)
        glLineWidth(1)
        glBegin(GL_LINES)
        glColor3f(0.5, 0.5, 0)
        for i in range(edge_max):
            index_former_rib_edge = i if i < len(former_rib_edges) - 1 else len(former_rib_edges) - 1
            index_edge = i if i < len(edges) - 1 else len(edges) - 1
            glVertex3fv((former_rib_edges[index_former_rib_edge][0], former_rib_edges[index_former_rib_edge][1], former_rib_edges[index_former_rib_edge][2]))
            glVertex3fv((edges[index_edge][0], edges[index_edge][1], edges[index_edge][2]))
        glEnd()
    
    @staticmethod
    def write_stl_inter_edges(former_rib_edges, edges, is_cockwise, f):
        edges_count = len(edges)
        if is_cockwise:
            for i in range(edges_count):
                facet = Facet()
                facet.vertex_1 = edges[i]
                facet.vertex_2 = edges[i-1]
                facet.vertex_3 = former_rib_edges[i-1]#i==0の時も成立
                facet.calc_normal()
                facet.write(f)

                facet = Facet()
                facet.vertex_1 = edges[i]
                facet.vertex_2 = former_rib_edges[i-1]
                facet.vertex_3 = former_rib_edges[i]
                facet.calc_normal()
                facet.write(f)
        else:
            for i in range(edges_count):
                facet = Facet()
                facet.vertex_1 = edges[i]
                facet.vertex_2 = former_rib_edges[i-1]#i==0の時も成立
                facet.vertex_3 = edges[i-1]
                facet.calc_normal()
                facet.write(f)

                facet = Facet()
                facet.vertex_1 = edges[i]
                facet.vertex_2 = former_rib_edges[i]
                facet.vertex_3 = former_rib_edges[i-1]
                facet.calc_normal()
                facet.write(f)
