# (u, p, B, r) 
# Case-LRW-2010
# Lagrangian multiplier method for energy and cross helicity

from firedrake import *
import ufl.algorithms
from mpi4py import MPI
import petsc4py

def scurl(x):
    return x[1].dx(0) - x[0].dx(1)


def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])

baseN = 32
mesh = PeriodicRectangleMesh(baseN, baseN, 1, 1, direction="both")

(x, y)= SpatialCoordinate(mesh)


Vg = VectorFunctionSpace(mesh, "CG", 4)
Q = FunctionSpace(mesh, "DG", 3)
R = FunctionSpace(mesh, "R", 0)

Vb = VectorFunctionSpace(mesh, "CG", 2)
Vn = FunctionSpace(mesh, "CG", 1)
Vc = FunctionSpace(mesh, "CG", 1)
# Mixed function space
# (u, p, B, r, lmbda_e, lmbda_c)
Z = MixedFunctionSpace([Vg, Q, Vg, Q, R, R])
z = Function(Z)

z_test = TestFunction(Z)
#(ut, pt, Bt, rt, lmbda_e, lmbda_c) = split(z_test)
z_prev = Function(Z)

#(up, pp, Bp, rp, lmbda_e, lmbda_c) = split(z_prev)
(u, p, B, r, lmbda_e, lmbda_c) = split(z)
(ut, pt, Bt, rt, lmbda_et, lmbda_ct) = split(z_test)
(up, pp, Bp, rp, lmbda_ep, lmbda_ep) = split(z_prev)

# initial condition
u1 = -sin(2*pi * y)
u2 = sin(2 * pi * x)
u_init = as_vector([u1, u2])
B1 = -sin(2 * pi *y) 
B2 = sin(4 * pi * x)
B_init = as_vector([B1, B2])
 
z_prev.sub(0).interpolate(u_init)
z_prev.sub(2).interpolate(B_init)

S = Constant(1)
nu = Constant(0)
eta = Constant(0)
    
#bcs = [DirichletBC(Z.sub(0), 0, (1, 2, 3, )),
#       DirichletBC(Z.sub(0), as_vector([1, 0]), (4,)),
#       DirichletBC(Z.sub(2), B_init, "on_boundary"),
#]
       

dt = Constant(0.001)
t = Constant(0)
T = 1.0
gamma = Constant(100)


u_avg = u
p_avg = p
B_avg = B 
r_avg = r 

def energy(u, B):
    return 0.5 * dot(u, u) + 0.5 * float(S) * dot(B, B)

def dissipation(u, B):
    return nu * inner(grad(u), grad(u)) + S * eta * inner(grad(B), grad(B))

def work(f, g):
    return dot(f, u) + dot(g, B)


def cHelicity(u, B):
    return dot(u, B)

def cdissipation(u, B):
    return (nu + eta) * inner(grad(u), grad(B)) 

F = (
#u
  inner((u - up)/dt, ut) * dx
+ nu * inner(grad(u_avg), grad(ut)) * dx
+ inner(dot(grad(u_avg), u_avg), ut)*dx # advection term
- inner(p_avg, div(ut)) * dx
- S * inner(dot(grad(B_avg), B_avg), ut) * dx
+ lmbda_e * inner(u, ut) * dx
+ lmbda_c * inner(B, ut) * dx

#p
- inner(div(u), pt) * dx

#B
+ inner((B - Bp)/dt, Bt) * dx
+ eta * inner(grad(B_avg), grad(Bt)) * dx
+ inner(dot(grad(B_avg), u_avg), Bt) * dx
- inner(dot(grad(u_avg), B_avg), Bt) * dx
+ inner(r_avg, div(Bt)) * dx
+ S * lmbda_e * inner(B, Bt) * dx
+ lmbda_c* inner(u, Bt) * dx
#r
+ inner(div(B), rt) * dx

# conservation law
## energy
+ 1/dt * inner(energy(u, B) - energy(up, Bp), lmbda_et) * dx 
+ inner(dissipation(u, B), lmbda_et) * dx
#- inner(work(f, g), lmbda_et) * dx

## helicity
+ 1/dt * inner(cHelicity(u, B) - cHelicity(up, Bp), lmbda_ct) * dx 
+ inner(cdissipation(u, B), lmbda_ct) * dx
)

lu = {
#"mat_type": "aij",
	 "snes_type": "newtonls",
	 "ksp_type":"preonly",
	 "pc_type": "lu",
	 "pc_factor_mat_solver_type":"mumps"
}

sp = {
    "mat_type": "matfree",
    #"snes_monitor": None, 
    "ksp_type": "fgmres",
#"ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_0_fields": "0, 1, 2, 3",
    "pc_fieldsplit_1_fields": "4, 5",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "python",
#"ksp_monitor": None,
        "pc_python_type": "firedrake.AssembledPC",
        "assembled_pc_type": "lu",
        "assembled_pc_factor_mat_solver_type": "mumps",
    },
    "fieldsplit_1": {
        "ksp_type": "gmres",
#"ksp_monitor": None,
        "pc_type": "none",
        "ksp_max_it": 2, 
        "ksp_convergence_test": "skip",

    },

}






bcs = None
problem = NonlinearVariationalProblem(F, z, bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters = sp)

(u, p, B, r, lmbda_e, lmbda_c) = z.subfunctions
B.rename("MagneticField")
r.rename("LagrangeMultiplier")
u.rename("Velocity")
p.rename("Pressure")
#pvd = VTKFile("output/reb-helicity-mhd.pvd")
z.assign(z_prev)
#pvd.write(*z.subfunctions)


#def compute_helicity(B):
#    A = Function(Vc)
#    v = TestFunction(Vc)
#    F_curl  = inner(curl(A), curl(v)) * dx - inner(B, curl(v)) * dx
#    sp = {  
#           "ksp_type":"gmres",
#           "pc_type": "ilu",
#    }
#    bcs_curl = [DirichletBC(Vc, 0, "on_boundary")]
#    hcurl_pb = NonlinearVariationalProblem(F_curl, A, bcs_curl)
#    solver_hcurl = NonlinearVariationalSolver(hcurl_pb, solver_parameters = sp, options_prefix = "solver_curlcurl")
#    solver_hcurl.solve()
#    return assemble(inner(A, B)*dx)

def compute_div(u):
    return norm(div(u), "L2")

def compute_energy(u, B):
    return assemble(0.5 * inner(u, u) * dx + 0.5 * S *  inner(B, B) * dx)

def compute_cross(u, B):
    return assemble(0.5 * inner(u, B) * dx)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    
    divu = compute_div(z.sub(0))
    divB = compute_div(z.sub(2))
    totalEnergy = compute_energy(z.sub(0), z.sub(2))
    cross = compute_cross(z.sub(0), z.sub(2))
    dofs = Z.dim()
    print(GREEN % f"divu = {divu}, divB={divB}, totalEnergy={totalEnergy}, crossHelicity={cross}, dofs = {dofs}")
    #pvd.write(*z.subfunctions, time=float(t))

    z_prev.assign(z)
