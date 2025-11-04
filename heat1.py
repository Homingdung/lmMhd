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
z_prev.sub(0).interpolate(u_init)
z_prev.sub(1).interpolate(Constant(0.0))
z.assign(z_prev)

f_exp = as_vector([0, 0])
f = Function(Vg).interpolate(f_exp)

dt = Constant(0.001)
t = Constant(0)
T = 1.0
nu = Constant(0)

def energy(u):
    return dot(u, u) 

def dissipation(u):
    return nu * inner(grad(u), grad(u))

def work(f):
    return dot(f, u)


def compute_energy(u):
    return assemble(0.5 * inner(u, u) * dx)

F = (
      inner((u - up)/dt, ut) * dx
    + nu * inner(grad(u), grad(ut)) * dx
    - inner(f, ut) * dx
    + lmbda * inner(u, ut) * dx
    + 1/dt * inner(energy(u) - energy(up), lmbdat) * dx
    + inner(dissipation(u), lmbdat) * dx
    - inner(work(f), lmbdat) * dx
)

bcs = [
    DirichletBC(Z.sub(0), 0, "on_boundary"),
]

problem = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(problem)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    energy_ = compute_energy(z.sub(0))
    print(f"energy={energy_}")
    
    z_prev.assign(z)
