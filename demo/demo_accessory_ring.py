from keel import Dock, Shipwright
import numpy as np
import os

split_ratio = 3

ring_circumference_length = 67.
ring_radius = ring_circumference_length / (2 * np.pi)

sw = Shipwright(Dock())

base = sw.rotate(np.pi/2., 0.).void()

center_line = sw.parent(base).void(30.)

sw.parent(center_line, 0.).spin(\
    np.array([(0., 0.), (2., 0.), (2., 3.), (0., 3.)]), \
    ring_radius, 16 * split_ratio)

sw.parent(center_line, 0.2).spin(\
    np.array([(0., 0.), (2., 0.), (2., 2.), (0., 2.)]), \
    ring_radius, 16 * split_ratio)

sw.parent(center_line, 0.15).rotate(np.pi /12, 0.).spin(\
    sw.rib_edges_circular(1.1, 2* np.pi, 8 * split_ratio, True), 
    ring_radius + 1.5, 16 * split_ratio)

sw.start_display()
# sw.generate_stl(".", "accessory_ring.stl")
