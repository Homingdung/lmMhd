# Lagrangian multiplier for heat equation
# only consider energy-preserving


from firedrake import *
from tabulate import tabulate

baseN = 32
mesh = UnitSquareMesh(baseN, baseN)

V = VectorFunctionSpace(mesh, 'CG', 2)  
Q = FunctionSpace(mesh, 'CG', 1)
# scalar lmbda_e
S = FunctionSpace(mesh, "R", 0)

Z = MixedFunctionSpace([V, Q])
(x, y) = SpatialCoordinate(mesh)

z0 = Function(Z)
z_test0 = TestFunction(Z)
z_prev0 = Function(Z)


(u0, p0) = split(z)
(u0t, p0t)= split(z_test)
(u0p, p0p) = split(z_prev)

z1 = Function(Z)
z_test1 = TestFunction(Z)
z_prev1 = Function(Z)


(u1, p1) = split(z1)
(u1t, p1t)= split(z_test1)
(u1p, p1p) = split(z_prev1)

Re = Constant(1) 

u_lid = as_vector([x*(1-x), 0])
u_noslip = Constant((0, 0))
bcs = [
        DirichletBC(Z.sub(0), u_noslip, (1, 2, 3)),
        DirichletBC(Z.sub(0), u_lid, (4,))]

dt = Constant(0.01)
t = Constant(0)
T = 1.0

lu = {
	 "mat_type": "aij",
	 "snes_type": "newtonls",
	 "ksp_type":"preonly",
	 "pc_type": "lu",
	 "pc_factor_mat_solver_type":"mumps"
}
sp = lu

F0 = (
    inner((u0 - u0p)/dt, u0t) * dx
    - inner(div(p0), u0t) * dx
    - inner(div(u0), p0t) * dx
)

F1 = (
    inner((u1 - u1p)/dt, u1t) * dx
    - inner(div(p1), u1t) * dx
    - inner(div(u1), p1t) * dx
)


pb0 = NonlinearVariationalProblem(F0, z0, bcs)
solver0 = NonlinearVariationalSolver(pb0, solver_parameters = sp)

pb1 = NonlinearVariationalProblem(F1, z1, bcs)
solver1 = NonlinearVariationalSolver(pb1, solver_parameters = sp)

# scalar function solver


while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    # solver original system
    solver.solve()
    (u, p) = z.subfunctions
    # corection for structure-preservation
    u.assign(u + lmbda_e * u)
    p.assign(p + lmbda_e * p)
    z_prev.assign(z)

