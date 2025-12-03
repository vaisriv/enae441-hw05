import matplotlib.pyplot as plt
import numpy as np

from textwrap import dedent
import time
import os


###############################################
# HELPER FUNCTIONS
###############################################


def load_numpy_data(file_path):
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data


DT = 10.0 # [s] measurement spacing
V_MEAS = 100 # [m^2] measurement noise


###############################################
# BATCH LEAST SQUARES
###############################################


def _coerce_measurement_array(data):
    arr = np.array(data)

    if arr.ndim == 2:
        # Already (n_trials, n_meas)
        return arr

    if arr.ndim == 1 and isinstance(arr[0], np.ndarray):
        # Likely an object array of per-trial arrays
        return np.vstack(arr)

    # Fallback: try to interpret as 2D
    if arr.ndim == 1:
        return arr[None, :]  # single trial

    raise ValueError(f"Unexpected measurement data shape: {arr.shape}")


def _build_design_matrix(num_meas, dt):
    t = np.arange(num_meas, dtype=float) * dt
    Gamma = np.column_stack((np.ones_like(t), t))
    return Gamma


def run_BLLS(file_path="./data/HW5Measurements.npy", dt=DT, V=V_MEAS):
    raw = load_numpy_data(file_path)
    measurements = _coerce_measurement_array(raw) # shape (n_trials, n_meas)

    n_trials, n_meas = measurements.shape
    Gamma_full = _build_design_matrix(n_meas, dt) # (n_meas, 2)

    # xhat_all[trial, k-1, :] = [z0_hat, zdot0_hat] using first k measurements
    xhat_all = np.zeros((n_trials, n_meas, 2))
    # covariance is the same across trials for a given k, so store per k
    P_all = np.zeros((n_meas, 2, 2))
    # timing per k (averaged over trials)
    avg_time_per_k = np.zeros(n_meas)
    counts_per_k = np.zeros(n_meas, dtype=int)

    for trial in range(n_trials):
        y_trial = measurements[trial, :].astype(float)

        for k in range(1, n_meas + 1):
            Gamma_k = Gamma_full[:k, :]
            y_k = y_trial[:k]

            start = time.perf_counter()
            # Batch LS: x_hat = (Gamma^T Gamma)^(-1) Gamma^T y
            GtG = Gamma_k.T @ Gamma_k
            x_hat = np.linalg.pinv(Gamma_k) @ y_k
            end = time.perf_counter()

            idx = k - 1
            xhat_all[trial, idx, :] = x_hat

            # analytical covariance (same for all trials, so only compute once)
            if trial == 0:
                P_all[idx, :, :] = V * np.linalg.pinv(GtG)

            avg_time_per_k[idx] += (end - start)
            counts_per_k[idx] += 1

    avg_time_per_k /= counts_per_k

    results = {
        "dt": dt,
        "V": V,
        "measurements": measurements,
        "xhat_all": xhat_all, # shape (n_trials, n_meas, 2)
        "P_all": P_all, # shape (n_meas, 2, 2)
        "avg_time_per_k": avg_time_per_k,
    }
    return results


# simple lazy cache
_RESULTS_BLLS_P1 = None
_RESULTS_BLLS_P3 = None

def _get_results_BLLS_P1():
    global _RESULTS_BLLS_P1
    if _RESULTS_BLLS_P1 is None:
        _RESULTS_BLLS_P1 = run_BLLS(file_path="./data/HW5Measurements-P1.npy", dt=DT, V=V_MEAS)
    return _RESULTS_BLLS_P1

def _get_results_BLLS_P3():
    global _RESULTS_BLLS_P3
    if _RESULTS_BLLS_P3 is None:
        _RESULTS_BLLS_P3 = run_BLLS(file_path="./data/HW5Measurements-P1.npy", dt=DT, V=V_MEAS)
    return _RESULTS_BLLS_P3


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


#######################
# Problem 1
#######################


# REQUIRED --- 1b
def plot_batch_least_squares_single_trial():
    res = _get_results_BLLS_P1()
    measurements = res["measurements"]
    xhat_all = res["xhat_all"]

    n_trials, n_meas = measurements.shape
    k_vec = np.arange(1, n_meas + 1)

    z_hat_1 = xhat_all[0, :, 0]
    meas_1 = measurements[0, :]

    fig, ax = plt.subplots()
    ax.plot(k_vec, z_hat_1, label=r"$\hat{z}_0$ (Batch LS estimate)")
    ax.scatter(k_vec, meas_1, s=10, alpha=0.4, label="Noisy Range measurements")

    ax.set_xlabel(r"Measurement Index $k$")
    ax.set_ylabel("Range / Position [m]")
    ax.set_title("Batch LS Estimate of S/C Range (Single Trial)")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1c
def plot_batch_least_squares_all_trials():
    res = _get_results_BLLS_P1()
    xhat_all = res["xhat_all"]
    P_all = res["P_all"]

    n_trials, n_meas, _ = xhat_all.shape
    k_vec = np.arange(1, n_meas + 1)

    # z0 estimates across trials
    z0_all = xhat_all[:, :, 0]
    mean_z0 = z0_all.mean(axis=0)
    sigma_z0 = np.sqrt(P_all[:, 0, 0])

    fig, ax = plt.subplots()

    for i in range(n_trials):
        ax.plot(k_vec, z0_all[i, :], color="0.8", linewidth=0.5)

    # mean estimate
    ax.plot(k_vec, mean_z0, label=r"Mean $\hat{z}_0$ across trials", linewidth=2)

    # +/- 3 sigma bounds
    upper = mean_z0 + 3.0 * sigma_z0
    lower = mean_z0 - 3.0 * sigma_z0
    ax.plot(k_vec, upper, "r--", label=r"$\pm 3\sigma$ bounds")
    ax.plot(k_vec, lower, "r--")

    ax.set_xlabel(r"Measurement Index $k$")
    ax.set_ylabel("Range / Position [m]")
    ax.set_title(r"Batch LS Estimates Over All Trials with $\pm 3\sigma$ Bounds")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1d
def plot_state_estimate_histograms():
    res = _get_results_BLLS_P1()
    xhat_all = res["xhat_all"] # (n_trials, n_meas, 2)
    P_all = res["P_all"]

    n_trials, n_meas, n_states = xhat_all.shape
    ks = [10, 50, 200]

    figs = []

    for k in ks:
        idx = k - 1
        x_k = xhat_all[:, idx, :]
        P_k = P_all[idx, :, :]

        fig, axes = plt.subplots(1, n_states)
        if n_states == 1:
            axes = [axes]

        for s in range(n_states):
            data = x_k[:, s]
            mu_s = np.mean(data)
            var_s = np.var(data, ddof=1)
            sigma_s = np.sqrt(var_s)

            ax = axes[s]
            ax.hist(data, bins=10, density=True, alpha=0.7)
            ax.set_xlabel("Estimate Value")
            ax.set_ylabel("Density")
            if s == 0:
                ax.set_title(rf"$\hat{{z}}_0$ after $k={k}$")
            else:
                ax.set_title(rf"$\hat{{\dot z}}_0$ after $k={k}$")

            # display sample mean and variance
            text = (
                rf"$\hat\mu = {mu_s:.2f}$" + "\n" +
                rf"$\hat\sigma^2 = {var_s:.2f}$" + "\n" +
                rf"$P_{{{s+1},{s+1}}} = {P_k[s, s]:.2f}$"
            )
            ax.text(
                0.98, 0.98, text,
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.suptitle(f"Histograms of state estimates after k={k} measurements")
        fig.tight_layout()
        figs.append(fig)

    description = dedent("""
        For each k ∈ {10, 50, 200}, the histograms show the distribution of the
        batch least-squares state estimates across the 50 trials.

        As k increases, the histograms for ẑ₀ become narrower and more tightly
        clustered around a common mean. The sample variances of ẑ₀ decrease
        and approach the corresponding diagonal entries of the analytical
        covariance Pₖ = (Γₖᵀ R⁻¹ Γₖ)⁻¹, which is expected for a linear
        unbiased least-squares estimator driven by independent Gaussian noise.

        The estimates of ż̂₀ are centered close to zero (since the spacecraft
        is effectively stationary in the chosen frame), and their spread also
        shrinks with increasing k, reflecting the fact that longer time spans
        give more leverage to estimate the slope in the constant-velocity model.
    """).strip()

    return figs, description


# REQUIRED --- 1e
def plot_execution_time_vs_measurements():
    res = _get_results_BLLS_P1()
    avg_time_per_k = res["avg_time_per_k"] # seconds
    n_meas = avg_time_per_k.shape[0]
    k_vec = np.arange(1, n_meas + 1)

    fig, ax = plt.subplots()
    ax.plot(k_vec[1:], avg_time_per_k[1:] * 1e6) # microseconds

    ax.set_xlabel(r"Number of Measurements Used ($k$)")
    ax.set_ylabel("Average Execution Time [µs]")
    ax.set_title("Average Batch LS Execution Time [µs] vs. Number of Measurements")
    ax.grid(True)

    return fig


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2a
def plot_recursive_lease_squares():
    fig = plt.figure()
    return fig


# REQUIRED --- Problem 2b
def plot_and_describe_sample_mean():
    fig = plt.figure()

    description = dedent("""
        Write your answer here.
        Write your answer here.
    """).strip()

    return fig, description


# REQUIRED --- Problem 2c
def plot_and_describe_time():
    fig = plt.figure()

    description = dedent("""
        Write your answer here.
        Write your answer here.
    """).strip()

    return fig, description


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def compute_final_x_and_P():
    x_final = 0
    P_final = 2
    return x_final, P_final

def print_final_x_and_P(x_final, P_final):
    output = dedent(f"""
        x_final = {x_final}
        P_final = {P_final}
    """).strip()

    return output

# REQUIRED --- Problem 3c
def plot_pure_prediction():
    fig = plt.figure()
    return fig


# REQUIRED --- Problem 3d
def plot_with_measurement_updates():
    fig = plt.figure()
    return fig


# REQUIRED --- Problem 3e
def describe_differences():
    description = dedent("""
        Write your answer here.
        Write your answer here.
    """).strip()

    return description


# REQUIRED --- Problem 3f
def plot_and_describe_residuals():
    fig = plt.figure()

    description = dedent("""
        Write your answer here.
        Write your answer here.
    """).strip()

    return fig, description


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    # 1b
    plot_batch_least_squares_single_trial().savefig("./outputs/figures/s01b.png", dpi=300)
    # 1c
    plot_batch_least_squares_all_trials().savefig("./outputs/figures/s01c.png", dpi=300)
    # 1d
    figs1d, desc1d = plot_state_estimate_histograms()
    for i in range(0, len(figs1d)):
        figs1d[i].savefig(("./outputs/figures/s01d" + str(i + 1) + ".png"), dpi=300)
    with open("./outputs/text/s01d.txt", "w", encoding="utf-8") as f:
        f.write(desc1d)
    # 1e
    plot_execution_time_vs_measurements().savefig("./outputs/figures/s01e.png", dpi=300)

    # Problem 2
    # 2a
    plot_recursive_lease_squares().savefig("./outputs/figures/s02a.png", dpi=300)
    # 2b
    fig2b, desc2b = plot_and_describe_sample_mean()
    fig2b.savefig("./outputs/figures/s02b.png", dpi=300)
    with open("./outputs/text/s02b.txt", "w", encoding="utf-8") as f:
        f.write(desc2b)
    # 2c
    fig2c, desc2c = plot_and_describe_time()
    fig2c.savefig(("./outputs/figures/s02c.png"), dpi=300)
    with open("./outputs/text/s02c.txt", "w", encoding="utf-8") as f:
        f.write(desc2c)

    # Problem 3
    # 3b
    x_final, P_final = compute_final_x_and_P()
    with open("./outputs/text/s03b.txt", "w", encoding="utf-8") as f:
        f.write(print_final_x_and_P(x_final, P_final))
    # 3c
    plot_pure_prediction().savefig("./outputs/figures/s03c.png", dpi=300)
    # 3d
    plot_with_measurement_updates().savefig("./outputs/figures/s03d.png", dpi=300)
    # 3e
    with open("./outputs/text/s03e.txt", "w", encoding="utf-8") as f:
        f.write(describe_differences())
    # 3f
    fig3f, desc3f = plot_and_describe_residuals()
    fig3f.savefig(("./outputs/figures/s03f.png"), dpi=300)
    with open("./outputs/text/s03f.txt", "w", encoding="utf-8") as f:
        f.write(desc3f)

    # plt.show()


if __name__ == "__main__":
    main()
