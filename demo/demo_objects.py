from keel import Dock, Shipwright
import numpy as np
import os

sw = Shipwright(Dock())

base = sw.rotate(np.pi/2, 0.).void(0.)

cube = sw.parent(base).cube(2.)

rectangular = sw.parent(cube).rectangular(2., 3., 4.)

pole = sw.parent(rectangular).pole(1., 0.5, 2 * np.pi, 16, True)

pole_half = sw.parent(pole).pole(1., 0.5, np.pi, 16)

sphere = sw.parent(pole_half).sphere(2., 16, 16, True)

sw.parent(sphere).spheroid(4., 1., 16, 32)

sw.start_display()
# sw.generate_stl(".", "objects.stl")
