from dataclasses import make_dataclass


def consume_fraction_of_error_budget(consumed_fraction, current_error_budget):
    # Takes in the current error budget and the amount to be consumed and outputs the amount consumed and the remaining error budget
    consumed_error_budget = consumed_fraction * current_error_budget
    remaining_error_budget = current_error_budget - consumed_error_budget
    return consumed_error_budget, remaining_error_budget

def create_data_class_from_dict(d):
    # Extract field names and types from the dictionary

    fields = [(key, type(value)) for key, value in d.items()]
    
    # Dynamically create the data class
    DataClass = make_dataclass('DynamicDataClass', fields)
    
    # Instantiate the data class with values from the dictionary
    instance = DataClass(**d)
    
    return instance