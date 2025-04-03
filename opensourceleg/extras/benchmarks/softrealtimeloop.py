import time

import matplotlib.pyplot as plt
import numpy as np

from opensourceleg.extras.benchmarks.stress import add_compute_stress
from opensourceleg.logging import Logger
from opensourceleg.utilities import SoftRealtimeLoop

DATA_DIRECTORY = "./softrealtimeloop_data"


def run_naive_rt_loop(
    frequency: float,
    duration: float,
    load_percentage: float = 0.0,
    variable_load: bool = False,
):
    """Run a naive rt loop using a variable sleep time"""
    dt = 1.0 / frequency
    timestamps = []

    start_time = time.monotonic()
    elapsed = 0

    while elapsed < duration:
        cycle_start = time.monotonic()
        t = time.monotonic() - start_time
        timestamps.append(t)

        add_compute_stress(frequency, load_percentage, variable_load)

        elapsed = t
        sleep_time = dt - (time.monotonic() - cycle_start)

        if sleep_time > 0:
            time.sleep(sleep_time)

    return timestamps


def run_naive_loop(
    frequency: float,
    duration: float,
    load_percentage: float = 0.0,
    variable_load: bool = False,
):
    """Run a naive loop using a fixed sleep time"""
    dt = 1.0 / frequency
    timestamps = []

    start_time = time.monotonic()
    elapsed = 0

    while elapsed < duration:
        t = time.monotonic() - start_time
        timestamps.append(t)
        elapsed = t

        add_compute_stress(frequency, load_percentage, variable_load)

        time.sleep(dt)

    return timestamps


def run_soft_rt_loop(
    frequency: float,
    duration: float,
    load_percentage: float = 0.0,
    variable_load: bool = False,
):
    """Run using the SoftRealtimeLoop class"""
    timestamps = []

    clock = SoftRealtimeLoop(dt=1.0 / frequency, maintain_original_phase=False)

    for t in clock:
        timestamps.append(t)

        add_compute_stress(frequency, load_percentage, variable_load)

        if t >= duration:
            break

    return timestamps


def plot_benchmark_results(
    naive_loop_timestamps: list[float],
    naive_rt_timestamps: list[float],
    soft_rt_timestamps: list[float],
    frequency: float,
    load_percentage: float = 0.0,
    variable_load: bool = False,
):
    """Plot histograms of cycle times and other performance metrics."""
    naive_loop_intervals = np.diff(naive_loop_timestamps) * 1000
    naive_rt_intervals = np.diff(naive_rt_timestamps) * 1000
    soft_rt_intervals = np.diff(soft_rt_timestamps) * 1000

    ideal_interval_ms = (1.0 / frequency) * 1000

    # Create figure with subplots (3 rows, 3 columns)
    fig, axs = plt.subplots(3, 3, figsize=(20, 12))

    # Histogram of cycle intervals
    axs[0, 0].hist(
        naive_loop_intervals,
        bins=200,
        alpha=0.7,
        label=f"Mean: {np.mean(naive_loop_intervals):.3f}ms\nStd: {np.std(naive_loop_intervals):.3f}ms",
    )
    axs[0, 0].axvline(ideal_interval_ms, color="r", linestyle="dashed", label=f"Ideal: {ideal_interval_ms:.3f}ms")
    axs[0, 0].set_title("Naive Loop - Cycle Intervals")
    axs[0, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend()

    axs[0, 1].hist(
        naive_rt_intervals,
        bins=200,
        alpha=0.7,
        label=f"Mean: {np.mean(naive_rt_intervals):.3f}ms\nStd: {np.std(naive_rt_intervals):.3f}ms",
    )
    axs[0, 1].axvline(ideal_interval_ms, color="r", linestyle="dashed", label=f"Ideal: {ideal_interval_ms:.3f}ms")
    axs[0, 1].set_title("Naive RT Loop - Cycle Intervals")
    axs[0, 1].set_xlabel("Time (ms)")
    axs[0, 1].set_ylabel("Count")
    axs[0, 1].legend()

    axs[0, 2].hist(
        soft_rt_intervals,
        bins=200,
        alpha=0.7,
        label=f"Mean: {np.mean(soft_rt_intervals):.3f}ms\nStd: {np.std(soft_rt_intervals):.3f}ms",
    )
    axs[0, 2].axvline(ideal_interval_ms, color="r", linestyle="dashed", label=f"Ideal: {ideal_interval_ms:.3f}ms")
    axs[0, 2].set_title("SoftRealtimeLoop - Cycle Intervals")
    axs[0, 2].set_xlabel("Time (ms)")
    axs[0, 2].set_ylabel("Count")
    axs[0, 2].legend()

    # Error from ideal interval
    naive_error = naive_loop_intervals - ideal_interval_ms
    naive_rt_error = naive_rt_intervals - ideal_interval_ms
    soft_rt_error = soft_rt_intervals - ideal_interval_ms

    axs[1, 0].plot(naive_loop_timestamps[1:], naive_error, "o-", markersize=2, alpha=0.5)
    axs[1, 0].set_title("Naive Loop - Error from Ideal Interval")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Error (ms)")
    axs[1, 0].axhline(0, color="r", linestyle="dashed")

    axs[1, 1].plot(naive_rt_timestamps[1:], naive_rt_error, "o-", markersize=2, alpha=0.5)
    axs[1, 1].set_title("Naive RT Loop - Error from Ideal Interval")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Error (ms)")
    axs[1, 1].axhline(0, color="r", linestyle="dashed")

    axs[1, 2].plot(soft_rt_timestamps[1:], soft_rt_error, "o-", markersize=2, alpha=0.5)
    axs[1, 2].set_title("SoftRealtimeLoop - Error from Ideal Interval")
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Error (ms)")
    axs[1, 2].axhline(0, color="r", linestyle="dashed")

    # Histogram of error distribution
    axs[2, 0].hist(
        naive_error,
        bins=100,
        alpha=0.7,
        label=f"Mean: {np.mean(naive_error):.2f}ms\n"
        f"Std: {np.std(naive_error):.2f}ms\n"
        f"Max: {np.max(np.abs(naive_error)):.2f}ms",
    )
    axs[2, 0].set_title("Naive Loop - Error Distribution")
    axs[2, 0].set_xlabel("Error (ms)")
    axs[2, 0].set_ylabel("Count")
    axs[2, 0].legend()

    axs[2, 1].hist(
        naive_rt_error,
        bins=100,
        alpha=0.7,
        label=f"Mean: {np.mean(naive_rt_error):.2f}ms\n"
        f"Std: {np.std(naive_rt_error):.2f}ms\n"
        f"Max: {np.max(np.abs(naive_rt_error)):.2f}ms",
    )
    axs[2, 1].set_title("Naive RT Loop - Error Distribution")
    axs[2, 1].set_xlabel("Error (ms)")
    axs[2, 1].set_ylabel("Count")
    axs[2, 1].legend()

    axs[2, 2].hist(
        soft_rt_error,
        bins=100,
        alpha=0.7,
        label=f"Mean: {np.mean(soft_rt_error):.2f}ms\n"
        f"Std: {np.std(soft_rt_error):.2f}ms\n"
        f"Max: {np.max(np.abs(soft_rt_error)):.2f}ms",
    )
    axs[2, 2].set_title("SoftRealtimeLoop - Error Distribution")
    axs[2, 2].set_xlabel("Error (ms)")
    axs[2, 2].set_ylabel("Count")
    axs[2, 2].legend()

    summary_text = (
        f"Mean Error (ms) | {np.mean(np.abs(naive_error)):.4f} "
        f"| {np.mean(np.abs(naive_rt_error)):.4f} "
        f"| {np.mean(np.abs(soft_rt_error)):.4f}\n"
        f"Max Error (ms) | {np.max(np.abs(naive_error)):.4f} "
        f"| {np.max(np.abs(naive_rt_error)):.4f} "
        f"| {np.max(np.abs(soft_rt_error)):.4f}\n"
        f"Std Deviation (ms) | {np.std(naive_error):.4f} "
        f"| {np.std(naive_rt_error):.4f} "
        f"| {np.std(soft_rt_error):.4f}\n"
    )

    fig.suptitle(
        f"SoftRealtimeLoop Performance Benchmark - Target Frequency: {frequency} Hz, "
        f"Compute Stress: {load_percentage * 100}%, "
        f"Variable Load: {variable_load}",
        fontsize=12,
    )
    fig.text(0.5, 0.0, summary_text, ha="center", va="bottom", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(f"sfrt_benchmark_{frequency}Hz_load{load_percentage}_variable{variable_load}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    frequencies = [200, 500, 1000]
    load_percentage = 0.80
    variable_load = True
    duration = 120

    rt_logger = Logger(file_name="rt_logger")

    for freq in frequencies:
        rt_logger.info(f"Running benchmark at {freq} Hz...")

        # Run benchmarks
        rt_logger.info("  Running naive loop...")
        naive_loop_timestamps = run_naive_loop(freq, duration, load_percentage, variable_load)

        rt_logger.info("  Running naive RT loop...")
        naive_rt_timestamps = run_naive_rt_loop(freq, duration, load_percentage, variable_load)

        rt_logger.info("  Running SoftRealtimeLoop...")
        soft_rt_timestamps = run_soft_rt_loop(freq, duration, load_percentage, variable_load)

        rt_logger.info("  Plotting results...")
        plot_benchmark_results(
            naive_loop_timestamps, naive_rt_timestamps, soft_rt_timestamps, freq, load_percentage, variable_load
        )

        rt_logger.info(f"Benchmark at {freq} Hz completed.")

        np.savez(
            f"benchmark_data_{freq}Hz_load_{load_percentage}_variable_{variable_load}.npz",
            naive_loop_timestamps=naive_loop_timestamps,
            naive_rt_timestamps=naive_rt_timestamps,
            soft_rt_timestamps=soft_rt_timestamps,
        )

    rt_logger.info("All benchmarks completed.")
