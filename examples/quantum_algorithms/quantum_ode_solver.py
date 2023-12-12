def generate_plot_for_taylor_quantum_ode():
    # Provided data and new mu_P_A values to loop over
    time_list = [(10 ** (i + 3)) for i in range(8)]
    mu_P_A_values = [
        -(0.1**i) for i in range(5)
    ]  # Generates list from -1 to -0.1 in increments of 0.1

    # Initialize the plot
    plt.figure(figsize=(10, 5))

    # Loop over mu_P_A values
    for mu_P_A in mu_P_A_values:
        query_count = []
        for time in time_list:
            evolution_time = time  # Example value
            failure_tolerance = 1e-10  # Example value
            norm_b = 0.0  # Example value
            norm_x_t = 1.0  # Example value
            A_stable = True
            kappa_P = 1

            # Initialize Taylor Quantum ODE Solver with your actual implementation
            taylor_ode = TaylorQuantumODESolver(
                amplify_amplitude=ObliviousAmplitudeAmplification(),
            )
            qlsa_subroutine = TaylorQLSA(
                linear_system_block_encoding=CarlemanBlockEncoding()
            )

            taylor_ode.set_requirements(
                evolution_time,
                mu_P_A,
                kappa_P,
                failure_tolerance,
                norm_b,
                norm_x_t,
                A_stable,
                qlsa_subroutine,
            )

            # Run the solver and get the query count
            taylor_ode.run_profile()
            taylor_ode.print_profile()
            query_count.append(
                taylor_ode.count_subroutines()["block_encode_carleman_linearization"]
            )

        # Plot the current curve
        plt.loglog(time_list, query_count, marker="o", label=f"mu_P_A = {mu_P_A}")

    # Plot the additional curves
    plt.loglog(
        time_list,
        [11900 * np.sqrt(T) * np.log(T) for T in time_list],
        label="11900 * sqrt(T) * log(T)",
        linestyle="--",
    )
    plt.loglog(
        time_list,
        [10300 * T * np.log(T) for T in time_list],
        label="10300 * T * log(T)",
        linestyle="-.",
    )

    # Add title and labels
    plt.title("Query Count vs. Time Taken for Different mu_P_A")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Query Count")
    plt.legend()  # Show legend to identify curves

    # Show grid
    plt.grid(True)  # , which="both", ls="--")

    # Display the plot
    plt.show()


# generate_plot_for_taylor_quantum_ode()


def generate_graphs():
    evolution_time = 10000  # Example value
    failure_tolerance = 1e-10  # Example value
    mu_P_A = -0.001
    norm_b = 0.0  # Example value
    norm_x_t = 1.0  # Example value
    A_stable = True
    kappa_P = 1

    # Initialize Taylor Quantum ODE Solver with your actual implementation
    taylor_ode = TaylorQuantumODESolver(
        amplify_amplitude=ObliviousAmplitudeAmplification(),
    )
    qlsa_subroutine = TaylorQLSA(linear_system_block_encoding=CarlemanBlockEncoding())

    taylor_ode.set_requirements(
        evolution_time,
        mu_P_A,
        kappa_P,
        failure_tolerance,
        norm_b,
        norm_x_t,
        A_stable,
        qlsa_subroutine,
    )

    # Run the solver and get the query count
    taylor_ode.run_profile()
    taylor_ode.print_profile()

    # Add child subroutines to root_subroutine...
    # taylor_ode.create_tree()
    print()
    print("Tree of subtasks and subroutines:")
    taylor_ode.display_tree()

    counts = taylor_ode.count_subroutines()
    print()
    print("Counts of subtasks:")
    for key, value in counts.items():
        print(f"'{key}': {value},")

    # graph = taylor_ode.display_hierarchy()
    # graph.view()  # This will open the generated diagram

    # Add child subroutines to taylor_ode...
    # taylor_ode.plot_graph()


generate_graphs()
