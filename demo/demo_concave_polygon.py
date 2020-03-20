from keel import Dock, Shipwright
import numpy as np
import os

mag_cup_radius = 35.

dock = Dock()
sw = Shipwright(dock)

base = sw.rotate(np.pi/2., 0.).void(0)

center_line = sw.parent(base).void(100.)

edges_star = [(-12.5, -3.5), (-2.5, -3.5), (0., -13.5), (3.5, -3.5), (12.5, -3.5), 
        (5.0, 1.5), (8.5, 11.5), (0, 4.5), (-8.5, 11.5), (-5., 1.5)]

sw.parent(center_line, 0.).pillar(edges_star, 10.)

#reverse
edges_star2 = edges_star[:]
edges_star2.reverse()
sw.parent(center_line, 0.33).pillar(edges_star, 10.)

#start position increment
edges_star3 = edges_star[1:]
edges_star3.append(edges_star[0])
sw.parent(center_line, 0.67).pillar(edges_star3, 10.)

#start position increment + reverse
edges_star4 = edges_star3[:]
edges_star4.reverse()
sw.parent(center_line, 1.).pillar(edges_star4, 10.)

sw.start_display()
# sw.generate_stl(".", "concave_polygon.stl")
