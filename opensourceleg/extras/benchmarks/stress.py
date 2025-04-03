import time

import numpy as np

PER_OPERATION_TIME = 0.00036  # s
ARRAY_SIZE = 64
DELTA_ARRAY_SIZE = 24


def add_compute_stress(frequency: float, max_percentage: float, variable_load: bool = False) -> None:
    """
    Add compute stress to the system by adding a large number of matrix multiplications
    The stress is added by adding matrix multiplications to the main loop

    Args:
        frequency: The frequency of the system
        max_percentage: The maximum percentage of the time that the system should be stressed
        variable_load: Whether the stress should be variable or not
    """
    cycle_time = 1 / frequency
    target_stress_time = cycle_time * max_percentage
    n_operations = int(target_stress_time / PER_OPERATION_TIME)
    counter = 0

    while counter < n_operations:
        if variable_load:
            array_size = np.random.randint(ARRAY_SIZE - DELTA_ARRAY_SIZE, ARRAY_SIZE + DELTA_ARRAY_SIZE)
        else:
            array_size = ARRAY_SIZE

        x = np.random.randn(array_size, array_size)
        y = np.random.randn(array_size, array_size)
        np.dot(x, y)

        counter += 1


if __name__ == "__main__":
    frequency = 100
    max_percentage = 0.0

    tic = time.monotonic()
    add_compute_stress(frequency, max_percentage, variable_load=True)
    toc = time.monotonic()
    print(f"Actual time taken: {(toc - tic):.6f} s, cycle time: {1 / frequency:.6f} s")
