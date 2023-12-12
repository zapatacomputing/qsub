from qsub.quantum_algorithms.gaussian_filtering_gsee import GF_LD_GSEE

# Initialize the highest-level subroutine
gf_ld_gsee = GF_LD_GSEE()


# Define paramters that will be used to set the requirements for this subroutine
alpha = 0.3
energy_gap = 0.5
square_overlap = 0.9
precision = 1e-3
failure_tolerance = 1e-1
hamiltonian = None
gf_ld_gsee.set_requirements(
    alpha=alpha,
    energy_gap=energy_gap,
    square_overlap=square_overlap,
    precision=precision,
    failure_tolerance=failure_tolerance,
    hamiltonian=hamiltonian,
)


# Run the profile for this subroutine
gf_ld_gsee.run_profile()
gf_ld_gsee.print_profile()
print(gf_ld_gsee.count_subroutines())
