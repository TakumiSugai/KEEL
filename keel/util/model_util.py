from dataclasses import dataclass, field

import numpy as np

from typing import List

from . import calc_util
 
@dataclass
class EndSurface:
    edges:List[np.array] = field(default_factory=list)
    vectors:List[np.array] = field(default_factory=list)
    exterior_angles:List[float] = field(default_factory=list)
    is_clockwise:bool = field(default=None)

    def rib_to_vectors(self, rib):
        self.vectors.clear()
        for i in range(len(rib.edges)):
            self.edges.append(calc_util.extract_vector_1by2(rib.edges[i]))
            self.vectors.append(np.array([
                rib.edges[i][0] - rib.edges[i - 1][0],
                rib.edges[i][1] - rib.edges[i - 1][1]]))
        self.calc_exterior_angles()
        self.is_clockwise = sum(self.calc_exterior_angles) < 0
        return self

    def calc_exterior_angles(self):
        self.exterior_angles.clear()
        for i in range(len(self.vectors) - 2):
            self.exterior_angles.append(
                calc_util.exterior_angle(
                    self.vectors[i],
                    self.vectors[i + 1]))
        self.exterior_angles.append(
            calc_util.exterior_angle(
                self.vectors[-1],
                self.vectors[0]))

