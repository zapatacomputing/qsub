from qsub.subroutine_model import SubroutineModel


class DFHamiltonianBlockEncoding(SubroutineModel):
    def __init__(
        self,
        task_name="hamiltonian_block_encoding",
        requirements=None,
        toffoli_gate=SubroutineModel("toffoli_gate"),
    ):
        super().__init__(task_name, requirements, toffoli_gate=toffoli_gate)

    def set_requirements(self, hamiltonian, failure_tolerance):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {k: v for k, v in args.items() if not k.startswith("__")}
        super().set_requirements(**args)

    def populate_requirements_for_subroutines(self):
        # Allocate failure tolerance
        allocation = 0.5
        consumed_failure_tolerance = allocation * self.requirements["failure_tolerance"]
        remaining_failure_tolerance = (
            self.requirements["failure_tolerance"] - consumed_failure_tolerance
        )
        # Convert instance into Toffoli cost
        # Note: The openfermion functions do not naturally take in a failure
        # tolerance so the consumed_failure_tolerance does not get used here
        h1, eri_full = get_integrals_from_hamiltonian_instance(
            self.requirements["hamiltonian"]
        )
        truncation_threshold = choose_threshold_for_df(h1, eri_full)
        toffoli_gate_cost, _ = get_double_factorized_be_toffoli_and_qubit_cost(
            h1,
            eri_full,
            truncation_threshold,
        )
        # Populate requirements
        self.toffoli_gate.number_of_times_called = toffoli_gate_cost
        self.toffoli_gate.set_requirements(
            failure_tolerance=remaining_failure_tolerance / toffoli_gate_cost
        )

    def get_subnormalization(self, hamiltonian_instance):
        h1, eri_full = get_integrals_from_hamiltonian_instance(hamiltonian_instance)
        truncation_threshold = choose_threshold_for_df(h1, eri_full)
        return get_double_factorized_be_subnormalization(
            h1, eri_full, truncation_threshold
        )
