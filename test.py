from firedrake import *

# mesh / spaces
baseN = 32
mesh = UnitSquareMesh(baseN, baseN)

V = VectorFunctionSpace(mesh, "CG", 2)  # vector field
R = FunctionSpace(mesh, "R", 0)         # space for scalar Lagrange multiplier

Z = V * R

# functions and tests
z = Function(Z, name="z")         # current unknown (u, lambda)
z_prev = Function(Z, name="z_prev")
(u, lmbda) = split(z)
(up, lmbdap) = split(z_prev)

(ut, lmbdat) = TestFunctions(Z)   # ut is vector test, lmbdat is scalar test

# forcing
f = as_vector([1.0, 0.0])

# time stepping
dt = Constant(0.001)
t = 0.0
T = 1.0

# boundary conditions (example: homogeneous Dirichlet on full boundary)
bc_u = DirichletBC(Z.sub(0), Constant((0.0, 0.0)), "on_boundary")
bcs = [bc_u]

# initial condition for u: for example a gaussian or zero
u0 = Function(V)
# example: initialize with small nonzero field (optional)
x, y = SpatialCoordinate(mesh)
u0_expr = as_vector([exp(-50*((x-0.25)**2 + (y-0.5)**2)), 0.0])
u0.interpolate(u0_expr)

# initialize z and z_prev
z.sub(0).assign(u0)
z.sub(1).assign(Constant(0.0))

z_prev.assign(z)  # copy initial state to previous

# define energy and dissipation (as integral forms)
def Energy(u):
    # E(u) = 1/2 * ∫ |u|^2 dx
    return 0.5 * inner(u, u) * dx

def Dissipation(u):
    # D(u) = ∫ |grad u|^2 dx
    return inner(grad(u), grad(u)) * dx

def Work(u):
    # Work from forcing = ∫ f·u dx
    return inner(f, u) * dx

# Residual of PDE (backward Euler) tested with ut:
R_pde = inner((u - up) / dt, ut) * dx + inner(grad(u), grad(ut)) * dx - inner(f, ut) * dx

# Contribution from Lagrange multiplier to PDE equation:
# derivative of Energy wrt u (direction ut) is ∫ u·ut dx
R_lmb_to_pde = lmbda * inner(u, ut) * dx

# Constraint (global scalar) enforcing discrete energy balance:
# C(u) = E(u) - E(up) + dt * ( D(u) - Work(u) ) = 0
C_u = Energy(u) - Energy(up) + dt * (Dissipation(u) - Work(u))

# Full mixed residual: PDE part + λ-term + constraint tested by scalar test lmbdat
F = R_pde + R_lmb_to_pde + inner(C_u, lmbdat) * dx

# set up nonlinear problem + solver
problem = NonlinearVariationalProblem(F, z, bcs=bcs)
solver = NonlinearVariationalSolver(problem)

# (optional) solver parameters tuning
# solver.parameters['snes_monitor'] = None
# solver.parameters['snes_atol'] = 1e-10

# time loop
while t < T - 1e-12:
    t += float(dt)
    if mesh.comm.rank == 0:
        print(f"Solving for t = {t:.6f} ...", flush=True)
    solver.solve()

    # advance
    z_prev.assign(z)
