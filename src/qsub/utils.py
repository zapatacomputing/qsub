# Module of helper functions
import numpy as np

def consume_fraction_of_error_budget(consumed_fraction, current_error_budget):
    # Takes in the current error budget and the amount to be consumed and outputs the amount consumed and the remaining error budget
    consumed_error_budget = consumed_fraction * current_error_budget
    remaining_error_budget = current_error_budget - consumed_error_budget
    return consumed_error_budget, remaining_error_budget

def calculate_max_of_solution_norm(fluid_nodes, uniform_density_deviation):

    phi_max_squared =0
    for k in range(1,4):
        phi_max_squared +=fluid_nodes**k*(1 + uniform_density_deviation)**(2*k)
    return np.sqrt(phi_max_squared)

