from keel import Dock, Shipwright
import numpy as np
import os

sw = Shipwright(Dock())

base = sw.rotate(np.pi/2, 0.).void(0.)

cube = sw.parent(base).cube(2.)

beam = sw.parent(base).rotate(0.,np.pi/6).void(1.)
rectangular = sw.parent(beam).rotate(np.pi/4).rectangular(2., 1., 4.)

# next...
# Subtract rectangular from cube

sw.start_display()
# sw.generate_stl(".", "boolean_subtract.stl")
