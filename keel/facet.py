from dataclasses import dataclass, field

import numpy as np

from typing import List

@dataclass
class Facet:
    normal: np.array = field(default=None)
    vertex_1: np.array = field(default=None)
    vertex_2: np.array = field(default=None)
    vertex_3: np.array = field(default=None)
    
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
        
