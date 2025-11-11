# helicityhu
# LM method

from firedrake import *
import csv
import os
import sys

baseN = 4
mesh = UnitCubeMesh(baseN, baseN, baseN)
(x, y, z0) = SpatialCoordinate(mesh)

k = 1
nu = Constant(1)
eta = Constant(1)
s = Constant(1)

t = Constant(0)
dt = Constant(0.01)
T = 1.0

Vg = VectorFunctionSpace(mesh, "CG", k)
Vg_ = FunctionSpace(mesh, "CG", k)
Vc = FunctionSpace(mesh, "N1curl", k)
Vd = FunctionSpace(mesh, "RT", k)
Vn = FunctionSpace(mesh, "DG", k-1)
VR = FunctionSpace(mesh, "R", 0)

# Mixed unknowns: [u, P, B, A, j, E, lmbda_e, lmbda_c, lmbda_m]
Z = MixedFunctionSpace([Vc, Vg_, Vd, Vc, Vc, Vc, VR, VR, VR])
z = Function(Z)
z_prev = Function(Z)
z_test = TestFunction(Z)
(u, P, B, A, j, E, lmbda_e, lmbda_c, lmbda_m) = split(z)
(ut, Pt, Bt, At, jt, Et, lmbda_et, lmbda_ct, lmbda_mt) = split(z_test)
(up, Pp, Bp, Ap, jp, Ep, lmbda_ep, lmbda_cp, lmbda_mp) = split(z_prev)

# Convenient references to subfunctions
(u_, P_, B_, A_, j_, E_, lmbda_e_, lmbda_c_, lmbda_m_) = z.subfunctions
B_.rename("MagneticField")
E_.rename("ElectricField")
u_.rename("Velocity")
A_.rename("MagneticPotential")
j_.rename("Current")


def g(x):
    return 32 * x**3 * (x - 1) ** 3

phi_ex = as_vector([y*g(x)*g(y) * g(z0), -x*g(x)*g(y)*g(z0), g(x)*g(y)*g(z0)])
u_ex = curl(phi_ex)
A_ex = as_vector([10 * y*g(x) * g(y) * g(z0), -10 * x*g(x)*g(y)*g(z0), 10 * g(x)*g(y)*g(z0)])
P_ex = sin(2*pi*x) * sin(2*pi*y) * sin(2*pi*z0)
B_ex = curl(A_ex)

z_prev.sub(0).interpolate(u_ex)    
z_prev.sub(1).interpolate(P_ex)    
z_prev.sub(2).interpolate(B_ex)
z_prev.sub(3).interpolate(A_ex)

z.assign(z_prev)

# source term
f_exp = as_vector([0, 0, 0])
f = Function(Vc).interpolate(f_exp)

def form_energy(B, u):
    return 0.5 * s * dot(B, B) + 0.5 * dot(u, u)

def form_dissipation_e(u, j):
    return nu * inner(curl(u), curl(u)) + s * eta * dot(j, j)


def form_helicity_c(u, B):
    return dot(u, B)

def form_dissipation_c(j, u):
    return (nu + eta) * inner(j, curl(u))

def form_helicity_m(A, B):
    return dot(A, B)

def form_dissipation_m(B, j):
    return 2 * eta * inner(B, j) 

def form_work(f, x):
    return dot(f, x)


F = (

    #u
    inner((u - up)/dt, ut) * dx
    - inner(cross(u, curl(u)), ut) * dx
    + nu * inner(curl(u), curl(ut)) * dx
    - s * inner(cross(j, B), ut) * dx
    + inner(grad(P), ut) * dx
    + lmbda_e * inner(u, ut) * dx # LM for energy_u
    + 2 * lmbda_c * inner(B, ut) * dx # LM for cross helicity
    #- inner(f, ut) * dx

    #P
    + inner(u, grad(Pt)) * dx
    
    #B
    + inner((B - Bp)/dt, Bt) * dx
    + inner(curl(E), Bt) * dx
    + lmbda_e * inner(B, Bt) * dx # LM for energy_B
    + 2 * lmbda_c * inner(u, Bt) * dx # LM for cross helicity
    # A
    + inner((A - Ap)/dt, At) * dx
    + inner(E, At) * dx
    + 2 * lmbda_m * inner(B, At) * dx # LM for magnetic helicity
    # j 
    + inner(j, jt) * dx
    - inner(B, curl(jt)) * dx
    # E
    + inner(E, Et) * dx
    + inner(cross(u, B), Et) * dx
    - eta * inner(j, Et) * dx
    
    # energy law 
    + 1/dt * inner(form_energy(B, u) - form_energy(Bp, up), lmbda_et) * dx
    + inner(form_dissipation_e(u, j), lmbda_et) * dx
    #- inner(form_work(f, u), lmbda_et) * dx

    # cross helicity law
    + 1/dt * inner(form_helicity_c(u, B) - form_helicity_c(up, Bp), lmbda_ct) * dx
    + inner(form_dissipation_c(j, u), lmbda_ct) * dx
    #- inner(form_work(f, B), lmbda_ct) * dx

    # helicity 
    + 1/dt * inner(form_helicity_m(A, B) - form_helicity_m(Ap, Bp), lmbda_mt) * dx
    + inner(form_dissipation_m(B, j), lmbda_mt) * dx
)

dirichlet_ids = ("on_boundary", )
bcs = [DirichletBC(Z.sub(index), 0, subdomain) for index in range(len(Z)-3) for subdomain in dirichlet_ids]

fs = {
    "mat_type": "matfree",
    "snes_monitor": None, 
    "ksp_type": "fgmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_0_fields": "0, 1, 2, 3, 4, 5",
    "pc_fieldsplit_1_fields": "6, 7, 8",
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
sp = fs
pb = NonlinearVariationalProblem(F, z, bcs=bcs)
solver = NonlinearVariationalSolver(pb, solver_parameters = sp)

while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    z_prev.assign(z)
