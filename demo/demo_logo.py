from keel import Dock, Shipwright
import numpy as np
import os

sw = Shipwright(Dock())

base = sw.rotate(np.pi/2., 0.).void(0)

center_line = sw.parent(base).void(3.)

stroke_one_k = sw.parent(center_line, 0.).rotate(np.pi/2., np.pi/2.).rectangular(0.1, 0.1, 1.)
stroke_two_k = sw.parent(stroke_one_k, 0.33).rotate(-np.pi/4., 0.).rectangular(0.1, 0.1, 0.8)
sw.parent(stroke_two_k, 0.3).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.6)

stroke_one_e_1 = sw.parent(center_line, 0.25).rotate(np.pi/2., np.pi/2.).rectangular(0.1, 0.1, 1.)
sw.parent(stroke_one_e_1, 0.05).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)
sw.parent(stroke_one_e_1, 0.5).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)
sw.parent(stroke_one_e_1, 0.95).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)

stroke_one_e_2 = sw.parent(center_line, 0.5).rotate(np.pi/2., np.pi/2.).rectangular(0.1, 0.1, 1.)
sw.parent(stroke_one_e_2, 0.05).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)
sw.parent(stroke_one_e_2, 0.5).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)
sw.parent(stroke_one_e_2, 0.95).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)

stroke_one_l = sw.parent(center_line, 0.75).rotate(np.pi/2., np.pi/2.).rectangular(0.1, 0.1, 1.)
sw.parent(stroke_one_l, 0.05).rotate(-np.pi/2., 0.).rectangular(0.1, 0.1, 0.5)

sw.start_display()
# sw.generate_stl(".", "logo.stl")
