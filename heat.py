# heat equation

from firedrake import *

baseN = 32
mesh = UnitSquareMesh(baseN, baseN)

(x, y)= SpatialCoordinate(mesh)


Vg = VectorFunctionSpace(mesh, "CG", 2)
R = FunctionSpace(mesh, "R", 0)

# u, lmbda
Z = MixedFunctionSpace([Vg, R])

z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)

(u, lmbda) = split(z)
(up, lmbdap) = split(z_prev)
(ut, lmbdat) = split(z_test)

x, y = SpatialCoordinate(mesh)
u_init = as_vector([exp(-50*((x-0.25)**2 + (y-0.5)**2)), 0.0])

# initialize z and z_prev
z.sub(0).interpolate(u_init)
z.sub(1).interpolate(Constant(0.0))

z_prev.assign(z)

f = as_vector([1, 0])

def energy(u):
    return inner(u, u) * dx 

def dissipation(u):
    return inner(grad(u), grad(ut)) * dx

def work(u):
    return inner(f, u) * dx

dt = Constant(0.001)
t = Constant(0)
T = 1.0

# Conservation law
Conser = energy(u) - energy(up) + dt * (dissipation(u) - work(u))

F=(
    inner((u - up)/dt, ut) * dx
    + inner(grad(u), grad(ut)) * dx
    + lmbda * inner(u, ut) * dx # energy direction law

    # structure want to preserve
    + inner(Conser, lmbdat) * dx
)

problem = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(problem)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    
    z_prev.assign(z)
