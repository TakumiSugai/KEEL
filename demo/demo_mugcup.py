from keel import Dock, Shipwright
import numpy as np
import os

mag_cup_radius = 35.

dock = Dock()
sw = Shipwright(dock)

base = sw.rotate(np.pi/2., 0.).void(0)

center_line = sw.parent(base).void(100.)

sw.parent(base).pole(10., mag_cup_radius, 2 * np.pi, 32, True)

sw.parent(base).spin(\
    [(0., 0.), (-5., 0.), (-5., 90.), (-3., 100.), \
    (3., 100.), (5., 90.), (5., 0.)], \
    mag_cup_radius, 16)


cup_rim_top_side = sw.parent(center_line, 0.8).rotate(np.pi/2., np.pi/2.).void(mag_cup_radius)
bar_top = sw.parent(cup_rim_top_side).rectangular(10., 20.,  50.)

cup_rim_bottom_side = sw.parent(center_line, 0.2).rotate(np.pi/2., np.pi/2.).void(mag_cup_radius)
sw.parent(cup_rim_bottom_side).rectangular(10., 20.,  50.)

sw.parent(bar_top, 0.8).rotate(np.pi /2, 0.).pole(60., 10., 2 * np.pi, 16, True)

sw.start_display()
# sw.generate_stl(".", "mugcup.stl")
