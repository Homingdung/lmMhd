# https://www.firedrakeproject.org/r-space.html#an-example
from firedrake import *
baseN = 25
mesh = UnitSquareMesh(baseN, baseN)
V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
Z = V * R
(u, r) = TrialFunction(Z)
(v, s) = TestFunction(Z)

a = inner(grad(u), grad(v)) * dx + u * s * dx + v * r * dx 
L = -v * ds(3) + v * ds(4)

z = Function(Z)
solve(a == L, z)
u, s = split(z)

exact = Function(V)
x, y = SpatialCoordinate(mesh)

exact.interpolate(y - 0.5)
print(sqrt(assemble((u - exact)*(u - exact)*dx)))

f = Function(R)
f.assign(0)
print(float(f))
