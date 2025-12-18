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
nu = Constant(0)
eta = Constant(0)
s = Constant(1)

t = Constant(0)
dt = Constant(0.01)
T = 1.0

dirichlet_ids = ("on_boundary", )

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


def project_ic(B_init):
    Zp = MixedFunctionSpace([Vd, Vn])
    zp = Function(Zp)
    (Bp_proj, p) = split(zp)
    test_B, test_p = TestFunctions(Zp)
    
    bcs_proj = [DirichletBC(Zp.sub(0), 0, sub) for sub in dirichlet_ids]
  
    L = (
        0.5*inner(Bp_proj, Bp_proj)*dx
        - inner(B_init, Bp_proj)*dx
        - inner(p, div(Bp_proj))*dx
    )
    Fp = derivative(L, zp, TestFunction(Zp))

    gamma = Constant(1E5)
    Up = 0.5*(inner(Bp_proj, Bp_proj) + inner(div(Bp_proj) * gamma, div(Bp_proj)) + inner(p * (1/gamma), p))*dx
    Jp = derivative(derivative(Up, zp), zp)

    spp = {
        "mat_type": "nest",
        "snes_type": "ksponly",
        "ksp_type": "minres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "fieldsplit_pc_type": "cholesky",
        "fieldsplit_pc_factor_mat_solver_type": "mumps",
        "ksp_atol": 1.0e-5,
        "ksp_rtol": 1.0e-5,
        "ksp_minres_nutol": 1E-8,
        "ksp_convergence_test": "skip",
    }

    solve(Fp == 0, zp, bcs_proj, Jp=Jp, solver_parameters=spp,
          options_prefix="B_init_div_free_projection")
    return zp.subfunctions[0]  # return projected B

def potential(B):
    Afunc = Function(Vc)
    v = TestFunction(Vc)
    F_curl  = inner(curl(Afunc), curl(v)) * dx - inner(B, curl(v)) * dx

    sp_curl = {  
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]
    pb = NonlinearVariationalProblem(F_curl, Afunc, bcs_curl)
    solver = NonlinearVariationalSolver(pb, solver_parameters = sp_curl)
    solver.solve()
    diff = norm(B - curl(Afunc), "L2")
    print(f"||B - curl(A)||_L2 = {diff:.8e}")
    return Afunc

def potential2(B):
    Z_curl = MixedFunctionSpace([Vc, Vn])
    z_curl = Function(Z_curl)
    z_curl_test = TestFunction(Z_curl)
    (A, p) = split(z_curl)
    (At, pt) = split(z_curl_test)
    F_curl  = inner(curl(A), curl(At)) * dx - inner(B, curl(At)) * dx + inner(A, grad(pt)) * dx

    sp_curl = {  
           "ksp_type":"gmres",
           "pc_type": "ilu",
    }
    bcs_curl = [DirichletBC(Vc, 0, sub) for sub in dirichlet_ids]
    pb = NonlinearVariationalProblem(F_curl, z_curl, bcs_curl)
    solver = NonlinearVariationalSolver(pb, solver_parameters = sp_curl)
    solver.solve()
    return z_curl.sub(0)

B_proj = project_ic(B_ex)   
z_prev.sub(0).interpolate(u_ex)    
z_prev.sub(1).interpolate(P_ex)    
z_prev.sub(2).interpolate(B_proj)
z_prev.sub(3).interpolate(potential(B_proj))


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
    + inner(B, Bt) * dx
    - inner(curl(A), Bt) * dx
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


pvd = VTKFile("output/3dmhd.pvd")
pvd.write(*z.subfunctions, time=float(t))

# store data
data_filename = "output/data.csv"
fieldnames = ["t", "energy", "helicity_c", "helicity_m", "divB"]
if mesh.comm.rank == 0:
    with open(data_filename, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# monitor
def compute_div(B):
    return norm(div(B), "L2")

def compute_helicity(x, y):
    return assemble(inner(x, y) * dx)

def compute_energy(u, B):
    return assemble(0.5 * inner(u, u) * dx + 0.5 * s * inner(B, B) * dx)


energy = compute_energy(z.sub(0), z.sub(2)) # u, B
helicity_m = compute_helicity(z.sub(3), z.sub(2))  # A, B
helicity_c = compute_helicity(z.sub(0), z.sub(2)) # u, B
divB = compute_div(z.sub(2)) # B
if mesh.comm.rank == 0:
    row = {
        "t": float(t),
        "energy": float(energy),
        "helicity_c": float(helicity_c),
        "helicity_m": float(helicity_m),
        "divB": float(divB),
    }
    with open(data_filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


while (float(t) < float(T-dt) + 1.0e-10):
    t.assign(t+dt)    
    if mesh.comm.rank==0:
        print(f"Solving for t = {float(t):.4f} .. ", flush=True)
    solver.solve()
    
    # monitor
    energy = compute_energy(z.sub(0), z.sub(2)) # u, B
    helicity_m = compute_helicity(z.sub(3), z.sub(2))  # A, B
    helicity_c = compute_helicity(z.sub(0), z.sub(2)) # u, B
    divB = compute_div(z.sub(2)) # B

    if mesh.comm.rank == 0:
        row = {
            "t": float(t),
            "energy": float(energy),
            "helicity_c": float(helicity_c),
            "helicity_m": float(helicity_m),
            "divB": float(divB),
        }
        with open(data_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    pvd.write(*z.subfunctions, time=float(t))
    print(RED % f"t={float(t)}, energy={energy}, helicity_c={helicity_c}, helicity_m={helicity_m}, divB={divB}")
    z_prev.assign(z)

