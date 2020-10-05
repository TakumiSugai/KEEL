from keel import Dock, Shipwright
import numpy as np
import os

sw = Shipwright(Dock())

base = sw.rotate(np.pi/2, 0.).void(5.)

cube1 = sw.parent(base,0.).cube(2.)
cube1.add_rib(0.5, cube1.ribs[0].edges)

beam = sw.parent(base, 0).rotate(0.,np.pi/6).void(1.)
rectangular = sw.parent(beam).rotate(np.pi/4).rectangular(2., 1., 4.)

cube1.subtracts.append(rectangular)

# bug: cannot running both at once
sw.start_display()
# sw.generate_stl(".", "boolean_subtract.stl")