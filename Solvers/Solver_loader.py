# Solver load
from Solvers.Standard_solver import Standard_solver
from Solvers.subject_harmonization_solver import subject_harmonization_solver
from Solvers.temporal_solver import temporal_solver
from Solvers.temporal_sequence_harmonization_adversarial_solver import temporal_sequence_harmonization_adversarial_solver

def solver_loader(cfg_proj, cfg_m):
    if cfg_proj.solver == "Standard_solver":
        s = Standard_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "subject_harmonization_solver":
        s = subject_harmonization_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "temporal_solver":
        s = temporal_solver(cfg_proj, cfg_m)
    elif cfg_proj.solver == "temporal_sequence_harmonization_adversarial_solver":
        s = temporal_sequence_harmonization_adversarial_solver(cfg_proj, cfg_m)
    return s

