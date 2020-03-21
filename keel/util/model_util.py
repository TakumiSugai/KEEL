from dataclasses import dataclass, field

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

from typing import List

from keel.util import calc_util

@dataclass
class Facet:
    vertex_1: np.array = field(default=None)
    vertex_2: np.array = field(default=None)
    vertex_3: np.array = field(default=None)
    normal: np.array = field(default=None)
    
    def translation(self, translation):
        self.vertex_1 = np.dot(self.vertex_1, translation)
        self.vertex_2 = np.dot(self.vertex_2, translation)
        self.vertex_3 = np.dot(self.vertex_3, translation)

    def calc_normal(self):
        cross_product = np.cross(\
            self.vertex_2[:-1] - self.vertex_1[:-1] ,\
            self.vertex_3[:-1] - self.vertex_1[:-1])
        l2_norm = np.linalg.norm(cross_product, ord=2)
        if l2_norm > 0.: #0除算が発生することがあるので暫定
            cross_product /= l2_norm
        self.normal = np.array([cross_product[0], cross_product[1], cross_product[2], 1.])

    def write(self, f):
        f.write(' facet normal {} {} {}\n'.format(self.normal[0], self.normal[1], self.normal[2]))
        f.write('  outer loop\n')
        f.write('   vertex {} {} {}\n'.format(self.vertex_1[0], self.vertex_1[1], self.vertex_1[2]))
        f.write('   vertex {} {} {}\n'.format(self.vertex_2[0], self.vertex_2[1], self.vertex_2[2]))
        f.write('   vertex {} {} {}\n'.format(self.vertex_3[0], self.vertex_3[1], self.vertex_3[2]))
        f.write('  endloop\n')
        f.write(' endfacet\n')
 
@dataclass
class EndSurface:
    edges:List[np.array] = field(default_factory=list)
    vectors:List[np.array] = field(default_factory=list)
    exterior_angles:List[float] = field(default_factory=list)
    is_clockwise:bool = field(default=None)

    def rib_to_vectors(self, rib):
        for i in range(len(rib.edges)):
            self.edges.append(calc_util.extract_vector_1by2(rib.edges[i]))
        self.vectors = EndSurface.calc_vectors(rib.edges)
        self.exterior_angles = EndSurface.calc_exterior_angles(self.vectors)
        self.is_clockwise = (sum(self.exterior_angles) < 0)
        return self

    def generate_facets(self, is_start_side):
        is_need_inverse = is_start_side ^ self.is_clockwise
        
        facets = []
        edges = self.edges[:]
        vectors = self.vectors[:]
        exterior_angles = self.exterior_angles[:]
        sum_exterior_angles = sum(self.exterior_angles)

        is_concave = True
        while(is_concave):
            is_concave = False
            for i in range(len(exterior_angles)):
                if exterior_angles[i - 2] * sum_exterior_angles <= 0:
                    is_concave = True
                    if 0 < exterior_angles[i - 1] * sum_exterior_angles:
                        facet = Facet()
                        if is_need_inverse:
                            facet.vertex_1 = calc_util.extend_vector_1by4(edges[i-2])
                            facet.vertex_2 = calc_util.extend_vector_1by4(edges[i])
                            facet.vertex_3 = calc_util.extend_vector_1by4(edges[i-1])
                        else:
                            facet.vertex_1 = calc_util.extend_vector_1by4(edges[i-2])
                            facet.vertex_2 = calc_util.extend_vector_1by4(edges[i-1])
                            facet.vertex_3 = calc_util.extend_vector_1by4(edges[i])
                        facets.append(facet)
                        del edges[i-1]
                        vectors = EndSurface.calc_vectors(edges)
                        exterior_angles = EndSurface.calc_exterior_angles(vectors)
                        break
        
        for i in range(len(edges) - 2):
            facet = Facet()
            if is_need_inverse:
                facet.vertex_1 = calc_util.extend_vector_1by4(edges[0])
                facet.vertex_2 = calc_util.extend_vector_1by4(edges[i+2])
                facet.vertex_3 = calc_util.extend_vector_1by4(edges[i+1])
            else:
                facet.vertex_1 = calc_util.extend_vector_1by4(edges[0])
                facet.vertex_2 = calc_util.extend_vector_1by4(edges[i+1])
                facet.vertex_3 = calc_util.extend_vector_1by4(edges[i+2])
            facets.append(facet)
        return facets
    
    def generate_monocoque_shells(self, z_position, is_start_side):
        is_need_inverse = is_start_side ^ self.is_clockwise
        
        monocoque_shells = []
        edges = self.edges[:]
        vectors = self.vectors[:]
        exterior_angles = self.exterior_angles[:]
        sum_exterior_angles = sum(self.exterior_angles)

        is_concave = True
        while(is_concave):
            is_concave = False
            for i in range(len(exterior_angles)):
                if exterior_angles[i - 2] * sum_exterior_angles <= 0:
                    is_concave = True
                    if 0 < exterior_angles[i - 1] * sum_exterior_angles:
                        monocoque_shell = MonocoqueShell(
                            calc_util.extend_vector_1by4(edges[i-2]),
                            calc_util.extend_vector_1by4(edges[i-1]),
                            calc_util.extend_vector_1by4(edges[i]))
                        if is_need_inverse:
                            monocoque_shell.inverse()
                        monocoque_shells.append(monocoque_shell)
                        del edges[i-1]
                        vectors = EndSurface.calc_vectors(edges)
                        exterior_angles = EndSurface.calc_exterior_angles(vectors)
                        break
        
        for i in range(len(edges) - 2):
            monocoque_shell = MonocoqueShell(
                calc_util.extend_vector_1by4(edges[0]),
                calc_util.extend_vector_1by4(edges[i+1]),
                calc_util.extend_vector_1by4(edges[i+2]))
            if is_need_inverse:
                monocoque_shell.inverse()
            monocoque_shells.append(monocoque_shell)
        for shell in monocoque_shells:
            shell.set_z_position(z_position)
        return monocoque_shells

    @staticmethod
    def calc_exterior_angles(vectors):
        exterior_angles = []
        for i in range(len(vectors) - 1):
            exterior_angles.append(
                calc_util.exterior_angle(vectors[i], vectors[i + 1]))
                
        exterior_angles.append(
            calc_util.exterior_angle(vectors[-1], vectors[0]))
        
        return exterior_angles
    
    @staticmethod
    def calc_vectors(edges):
        vectors = []
        for i in range(len(edges)):
            vectors.append(np.array(edges[i]) - np.array(edges[i-1]))
        return vectors

@dataclass
class MonocoqueShell:
    vertex_1: np.array = field(default=None)
    vertex_2: np.array = field(default=None)
    vertex_3: np.array = field(default=None)

    def inverse(self):
        temp = self.vertex_2
        self.vertex_2 = self.vertex_3
        self.vertex_3 = temp

    def draw(self, keel):
        if self.vertex_1 is None or self.vertex_2 is None or self.vertex_3 is None:
            return
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glColor3f(0,1,0.5)

        translation = np.dot(keel.relative_translation, keel.origin_translation)
        
        vertex_1_translated = np.dot(calc_util.extend_vector_1by4(self.vertex_1), translation)
        glVertex3fv((vertex_1_translated[0], vertex_1_translated[1], vertex_1_translated[2]))

        vertex_2_translated = np.dot(calc_util.extend_vector_1by4(self.vertex_2), translation)
        glVertex3fv((vertex_2_translated[0], vertex_2_translated[1], vertex_2_translated[2]))

        vertex_3_translated = np.dot(calc_util.extend_vector_1by4(self.vertex_3), translation)
        glVertex3fv((vertex_3_translated[0], vertex_3_translated[1], vertex_3_translated[2]))

        glEnd()
    
    def write_stl(self, keel, f):
        if self.vertex_1 is None or self.vertex_2 is None or self.vertex_3 is None:
            return
        facet = Facet(self.vertex_1, self.vertex_2, self.vertex_3)
        facet.translation(np.dot(keel.relative_translation, keel.origin_translation))
        facet.calc_normal()
        facet.write(f)

    def set_z_position(self, z_position):
        if self.vertex_1 is None or self.vertex_2 is None or self.vertex_3 is None:
            return
        self.vertex_1[2] = z_position
        self.vertex_2[2] = z_position
        self.vertex_3[2] = z_position