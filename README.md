# nbodies
Newtonian n-body 3D gravitational interactions with graphical display (python3 version)

Based on original work on Atari ST / GFA Basic dating back to 1986/87 that did 2D calculations and used the Euler integration, it has been entirely rewritten in GFA Basic and then into Python3. It features a 3D model with a choice between Euler and the much more accurate Verlet integration, 
variable integration step, planet collision and breakup rules... 

The Python3 version is fully object-oriented and uses the pygame rendering engine. If features classes Universe and Planet. The Planet class has variables and methods relative to each planet behavior, including drawing, while the Universe class has a dynamic table of Planets and performs the physics: interaction calculation, time step adjustment, collision and fragmentation deteciton, etc. Global objects such as the representation surface are managed in the __main__ section and passed down myUniverse and Planet objects.

It has a choice of preset initial conditions (called by the keyboard digits) as well as a few commands to trigger a variety of random initial conditions (few or many bodies). Use the embedded help for info (key ? or h).

Call with: 
> python3 gravity_v3.py
