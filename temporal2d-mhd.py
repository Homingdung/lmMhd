# (u, p, B, r) 
# Case-LRW-2010
# Lagrangian multiplier method for energy and cross helicity
# temporal convergence study

from firedrake import *
import ufl.algorithms
from mpi4py import MPI
import petsc4py
import csv
from tabulate import tabulate


# Define cross and curl operations
def scross(x, y):
    return x[0]*y[1] - x[1]*y[0]

def vcross(x, y):
    return as_vector([x[1]*y, -x[0]*y])

def scurl(x):
    return x[1].dx(0) - x[0].dx(1)

def vcurl(x):
    return as_vector([x.dx(1), -x.dx(0)])

# Storage for errors
errors_u = []
errors_p = []
errors_B = []
rates_u = []
rates_p = []
rates_B = []

# Time stepping parameters
T = 1.0
dt_values = [1/16, 1/32]  # Different time steps for convergence test

baseN = 24
mesh = UnitSquareMesh(baseN, baseN)

(x, y)= SpatialCoordinate(mesh)


Vg = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)

Vb = VectorFunctionSpace(mesh, "CG", 2)
Vn = FunctionSpace(mesh, "CG", 1)
Vc = FunctionSpace(mesh, "CG", 1)

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

S = Constant(1)
nu = Constant(1)
eta = Constant(1)
    

for dt_val in dt_values:
    dt = Constant(dt_val)
    t = Constant(0)
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


    # Exact solution
    u_exact = as_vector([y * exp(-t), x * cos(t)])
    B_exact = as_vector([y * cos(t), x * exp(-t)])
    p_exact = Constant(0)
    u_ex_t = as_vector([-y * exp(-t), - x * sin(t)])
    B_ex_t = as_vector([-y * sin(t), -x * exp(-t)])

    # Compute forcing terms
    f =  u_ex_t - nu * div(grad(u_exact)) + dot(grad(u_exact), u_exact) + S * dot(grad(B_exact), B_exact) 
    g =  B_ex_t - eta * div(grad(B_exact)) - dot(grad(B_exact), u_exact) - dot(grad(u_exact), B_exact)
 
    # initial condition
    z_prev.sub(0).interpolate(u_exact)
    z_prev.sub(2).interpolate(B_exact)
    z.assign(z_prev)

    F = (
    #u
      inner((u - up)/dt, ut) * dx
    + nu * inner(grad(u_avg), grad(ut)) * dx
    + inner(dot(grad(u_avg), u_avg), ut)*dx # advection term
    - inner(p_avg, div(ut)) * dx
    - S * inner(dot(grad(B_avg), B_avg), ut) * dx
    + lmbda_e * inner(u, ut) * dx
    + lmbda_c * inner(B, ut) * dx
    - inner(f, ut) * dx
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
    - inner(work(f, g), lmbda_et) * dx

    ## helicity
    + 1/dt * inner(cHelicity(u, B) - cHelicity(up, Bp), lmbda_ct) * dx 
    + inner(cdissipation(u, B), lmbda_ct) * dx
    - inner(work(f, g), lmbda_ct) * dx
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


    bcs = [
       DirichletBC(Z.sub(0), u_exact, "on_boundary"),
       DirichletBC(Z.sub(2), B_exact, "on_boundary"),
    ]

    problem = NonlinearVariationalProblem(F, z, bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters = sp)

    (u, p, B, r, lmbda_e, lmbda_c) = z.subfunctions
    B.rename("MagneticField")
    r.rename("LagrangeMultiplier")
    u.rename("Velocity")
    p.rename("Pressure")
    #pvd = VTKFile("output/helicity-mhd.pvd")
    #pvd.write(u, p, B, r, time=float(t))

    def compute_div(u):
        return norm(div(u), "L2")

    def compute_energy(u, B):
        return assemble(0.5 * inner(u, u) * dx + 0.5 * S *  inner(B, B) * dx)

    def compute_cross(u, B):
        return assemble(0.5 * inner(u, B) * dx)


    data_filename = "data.csv"
    if mesh.comm.rank == 0:
        with open(data_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "energy", "helicity_c", "divu", "divB"])


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
        #pvd.write(u, p, B, r, time=float(t))
        if mesh.comm.rank == 0:
            with open(data_filename, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{float(t):.4f}", f"{totalEnergy}", f"{cross}", f"{divu}", f"{divB}"])		
        
        z_prev.assign(z)

    # Compute L2 error
    u_error = errornorm(u_exact, z.sub(0), "L2")
    p_error = errornorm(p_exact, z.sub(1), "L2")
    B_error = errornorm(B_exact, z.sub(2), "L2")
    print(f"dt: {dt_val}, u_error: {u_error}, p_error: {p_error}, B_error: {B_error}")

    errors_u.append(u_error)
    errors_p.append(p_error)
    errors_B.append(B_error)

# Print results
headers = ["dt", "Error (u)", "Rate (u)", "Error (p)", "Rate(p)", "Error (B)", "Rate (B)"]
table_data = []
for i in range(len(dt_values)):
    if i == 0:
        table_data.append([dt_values[i], errors_u[i], "-", errors_p[i], "-", errors_B[i], "-"])
    else:
        table_data.append([dt_values[i], errors_u[i], rates_u[i-1], errors_p[i], rates_p[i-1], errors_B[i], rates_B[i-1]])

print("\nTemporal Convergence Results:")
print(tabulate(table_data, headers=headers, floatfmt=".4e"))
print(tabulate(table_data, headers=headers, tablefmt="latex"))
