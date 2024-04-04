def consume_fraction_of_error_budget(consumed_fraction, current_error_budget):
    # Takes in the current error budget and the amount to be consumed and outputs the amount consumed and the remaining error budget
    consumed_error_budget = consumed_fraction * current_error_budget
    remaining_error_budget = current_error_budget - consumed_error_budget
    return consumed_error_budget, remaining_error_budget
