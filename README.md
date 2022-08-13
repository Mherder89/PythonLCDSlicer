This project is a proof-of-concept slicer for LCD Resin Printers:

The goal is to have the distance to the surface for every illuminated pixel,
so that a physical model for its desired brightness can be applied.
This should lead so more accurate small holes and less overall blooming.

It is clear that this is quite a resource heavy computation, so the first step
is to implement the slicing and distance calculation on a GPU using CUDA.

The second problem is to find a suitable algorithm to calculate the pixel brightness given the
distances to the surface. This is complicated because we need to combine the light of all pixels
to get the intensity at a given surface point.

The third problem is finding a calibration procedure for the variables
(min cure intensity, max cure intensity, light absorption in xy and z direction)
used in the model.

After 1-3 is achieved the slicer should be ported from Python to c# or c++ for better performance
and a more user-friendly interface.
