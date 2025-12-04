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
        # already (n_trials, n_meas)
        return arr

    if arr.ndim == 1 and isinstance(arr[0], np.ndarray):
        # per-trial arrays
        return np.vstack(arr)

    if arr.ndim == 1:
        # single trial
        return arr[None, :]

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

            # Batch LS: x_hat = (Gamma^T Gamma)^(-1) Gamma^T y
            start = time.perf_counter()
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


###############################################
# RECURSIVE LEAST SQUARES
###############################################


def _build_rls_design_vector(k, dt):
    t_k = (k - 1) * dt
    w = np.array([0.0, 0.0, 1.0, 0.0, 0.0, t_k], dtype=float)
    return w


def run_RLS(file_path="data/HW5Measurements.npy", dt=DT, V=V_MEAS):
    raw = load_numpy_data(file_path)
    measurements = _coerce_measurement_array(raw) # shape (n_trials, n_meas)

    n_trials, n_meas = measurements.shape
    w_k = _build_rls_design_vector(n_meas, dt) # (6,)

    # initial state
    x0 = np.array([0.0, 0.0, 42e3, 0.0, 0.0, 0.0], dtype=float)

    # initial covariance
    P0 = np.zeros((6, 6), dtype=float)
    P0[:3, :3] = 50.0 * np.eye(3)
    P0[3:, 3:] = 1.0 * np.eye(3)

    # xhat_all[trial, k-1, :] = [0, 0, z0_hat, 0, 0, zdot0_hat] using first k measurements
    xhat_all = np.zeros((n_trials, n_meas, 6))
    # covariance is the same across trials for a given k, so store per k
    P_all = np.zeros((n_meas, 6, 6))
    # timing per k (averaged over trials)
    avg_time_per_update = np.zeros(n_meas)
    counts_per_k = np.zeros(n_meas, dtype=int)

    for trial in range(n_trials):
        x_hat = x0.copy()
        P = P0.copy()

        for k in range(1, n_meas + 1):
            idx = k - 1
            z_meas = measurements[trial, idx]

            # RLS update
            start = time.perf_counter()
            S_k = float(w_k @ (P @ w_k) + V)
            K_k = (P @ w_k) / S_k
            innov = z_meas - w_k @ x_hat
            x_hat = x_hat + K_k * innov
            P = (np.eye(6) - np.outer(K_k, w_k)) @ P
            end = time.perf_counter()

            xhat_all[trial, idx, :] = x_hat

            # analytical covariance (same for all trials, so only compute once)
            if trial == 0:
                P_all[idx, :, :] = P

            avg_time_per_update[idx] += (end - start)
            counts_per_k[idx] += 1

    avg_time_per_update /= counts_per_k

    results = {
        "dt": dt,
        "V": V,
        "measurements": measurements,
        "xhat_all": xhat_all, # shape (n_trials, n_meas, 2)
        "P_all": P_all, # shape (n_meas, 2, 2)
        "avg_time_per_update": avg_time_per_update,
    }
    return results


###############################################
# KALMAN FILTER
###############################################


def run_KF(file_path="data/HW5Measurements.npy"):
    raw = load_numpy_data(file_path)

    y_meas = raw.item()["Y"] # shape (N, 3)
    t_meas = raw.item()["t"]

    n_steps = y_meas.shape[0]
    dt = np.mean(np.diff(t_meas))

    # State Model
    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    # State transition F
    F = np.block([
        [I3, dt * I3],
        [Z3, I3]
    ])  # 6x6

    # Process noise covariance Q
    q = 1e-5 # from W = q * I3
    dt3 = dt**3 / 3.0
    dt2 = dt**2 / 2.0
    Q_pos = dt3 * I3
    Q_cross = dt2 * I3
    Q_vel = dt * I3
    Q = q * np.block([
        [Q_pos,   Q_cross],
        [Q_cross, Q_vel  ]
    ]) # 6x6

    # Measurement matrix H and noise R
    H = np.hstack((I3, Z3)) # 3x6
    R = 1e3 * I3 # 3x3 (R=V)

    # ICs
    x0 = np.array([0.0, 0.0, 500.0, 0.01, 0.0, 0.01]) # km, km/s

    P0 = np.zeros((6, 6))
    P0[0:3, 0:3] = 50.0 * I3 # km^2
    P0[3:6, 3:6] = 0.1 * I3 # km^2 / s^2

    # Store Vals
    x_pred = np.zeros((n_steps, 6)) # μ_k^- (prior / prediction)
    P_pred = np.zeros((n_steps, 6, 6))
    x_filt = np.zeros((n_steps, 6)) # μ_k^+ (posterior / corrected)
    P_filt = np.zeros((n_steps, 6, 6))
    residuals = np.zeros((n_steps, 3))
    innovation = np.zeros((n_steps, 3))

    # Timing
    step_time = np.zeros(n_steps)

    # Initialize
    x_plus = x0.copy()
    P_plus = P0.copy()

    # Filter loop
    for k in range(n_steps):
        y_k = y_meas[k, :]

        t_start = time.perf_counter()

        # Predict
        x_minus = F @ x_plus
        P_minus = F @ P_plus @ F.T + Q

        # Store prediction
        x_pred[k, :] = x_minus
        P_pred[k, :, :] = P_minus

        # Update
        S_k = H @ P_minus @ H.T + R # 3x3
        K_k = P_minus @ H.T @ np.linalg.inv(S_k) # 6x3

        y_pred = H @ x_minus
        innov_k = y_k - y_pred # innovation using prediction
        x_plus = x_minus + K_k @ innov_k
        P_plus = (np.eye(6) - K_k @ H) @ P_minus

        # Store corrected state
        x_filt[k, :] = x_plus
        P_filt[k, :, :] = P_plus

        # Residual using μ_k^+
        residuals[k, :] = y_k - H @ x_plus
        innovation[k, :] = innov_k

        t_end = time.perf_counter()
        step_time[k] = t_end - t_start

    results = {
        "dt": dt,
        "y_meas": y_meas,
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "x_pred": x_pred,
        "P_pred": P_pred,
        "x_filt": x_filt,
        "P_filt": P_filt,
        "residuals_plus": residuals, # y_k - H μ_k^+
        "innovation": innovation, # y_k - H μ_k^-
        "step_time": step_time,
    }
    return results

###############################################
# CACHE
###############################################


_RESULTS_BLLS = None
_RESULTS_RLS = None
_RESULTS_KF = None


def _get_results_BLLS():
    global _RESULTS_BLLS
    if _RESULTS_BLLS is None:
        _RESULTS_BLLS = run_BLLS(file_path="./data/HW5Measurements-P1.npy", dt=DT, V=V_MEAS)
    return _RESULTS_BLLS


def _get_results_RLS():
    global _RESULTS_RLS
    if _RESULTS_RLS is None:
        _RESULTS_RLS = run_RLS(file_path="./data/HW5Measurements-P1.npy", dt=DT, V=V_MEAS)
    return _RESULTS_RLS


def _get_results_KF():
    global _RESULTS_KF
    if _RESULTS_KF is None:
        _RESULTS_KF = run_KF(file_path="./data/HW5Measurements-P3.npy")
    return _RESULTS_KF


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


#######################
# Problem 1
#######################


# REQUIRED --- 1b
def plot_batch_least_squares_single_trial():
    res = _get_results_BLLS()
    measurements = res["measurements"]
    xhat_all = res["xhat_all"]

    n_trials, n_meas = measurements.shape
    k_vec = np.arange(1, n_meas + 1)

    z_hat_1 = xhat_all[0, :, 0]/1e3
    meas_1 = measurements[0, :]/1e3

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_vec, z_hat_1, label=r"$\hat{z}_0$ (Batch LS estimate)")
    ax.scatter(k_vec, meas_1, s=10, alpha=0.4, label="Noisy Range measurements")

    ax.set_xlabel(r"Measurement Index $k$")
    ax.set_ylabel("Range / Position [km]")
    ax.set_title("Batch LS Estimate of S/C Range (Single Trial)")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1c
def plot_batch_least_squares_all_trials():
    res = _get_results_BLLS()
    xhat_all = res["xhat_all"]
    P_all = res["P_all"]

    n_trials, n_meas, _ = xhat_all.shape
    k_vec = np.arange(1, n_meas + 1)

    # z0 estimates across trials
    z0_all = xhat_all[:, :, 0]
    mean_z0 = z0_all.mean(axis=0)
    sigma_z0 = np.sqrt(P_all[:, 0, 0])

    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(n_trials):
        ax.plot(k_vec, z0_all[i, :]/1e3, color="0.8", linewidth=0.5)

    # mean estimate
    ax.plot(k_vec, mean_z0/1e3, label=r"Mean $\mu_k$ across trials", linewidth=2)

    # +/- 3 sigma bounds
    upper = mean_z0/1e3 + 3.0 * sigma_z0/1e3
    lower = mean_z0/1e3 - 3.0 * sigma_z0/1e3
    ax.plot(k_vec, upper, "r--", label=r"$\pm 3\sigma$ bounds")
    ax.plot(k_vec, lower, "r--")

    ax.set_xlabel(r"Measurement Index $k$")
    ax.set_ylabel("Range / Position [km]")
    ax.set_title(r"Batch LS Estimates Over All Trials with $\pm 3\sigma$ Bounds")
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1d
def plot_state_estimate_histograms():
    res = _get_results_BLLS()
    xhat_all = res["xhat_all"] # (n_trials, n_meas, 2)
    P_all = res["P_all"]

    n_trials, n_meas, n_states = xhat_all.shape
    ks = [10, 50, 200]

    figs = []

    for k in ks:
        idx = k - 1
        x_k = xhat_all[:, idx, :]
        P_k = P_all[idx, :, :]

        fig, axes = plt.subplots(n_states, 1, figsize=(4, 8))
        if n_states == 1:
            axes = [axes]

        for s in range(n_states):
            data = x_k[:, s]/1e3
            mu_s = np.mean(data)
            var_s = np.var(data, ddof=1)
            sigma_s = np.sqrt(var_s)

            ax = axes[s]
            ax.hist(data, bins=10, density=True, alpha=0.7)
            ax.set_xlabel("EV [km]")
            ax.set_ylabel("Density")
            if s == 0:
                ax.set_title(rf"$\hat{{z}}_0$ after $k={k}$")
            else:
                ax.set_title(rf"$\hat{{\dot z}}_0$ after $k={k}$")

            # display sample mean and variance
            text = (
                rf"$\hat\mu = {mu_s:.3e}$ [km]" + "\n" +
                rf"$\hat\sigma^2 = {var_s:.2f}$" + "\n" +
                rf"$P_{{{s+1},{s+1}}} = {P_k[s, s]:.2f}$"
            )
            ax.text(
                0.98, 0.98, text,
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        fig.suptitle(f"Histograms of State Estimates\nAfter k={k} Measurements")
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
    res = _get_results_BLLS()
    avg_time_per_k = res["avg_time_per_k"] # seconds
    n_meas = avg_time_per_k.shape[0]
    k_vec = np.arange(1, n_meas + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
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
    res = _get_results_RLS()
    xhat_all = res["xhat_all"] # (n_trials, n_meas, 6)
    n_trials, n_meas, n_states = xhat_all.shape

    k_vec = np.arange(1, n_meas + 1)

    state_labels = [
        r"$x$ [km]",
        r"$y$ [km]",
        r"$z$ [km]",
        r"$\dot x$ [km/s]",
        r"$\dot y$ [km/s]",
        r"$\dot z$ [km/s]",
    ]

    # only display z-states
    idzs = (2, 5)
    # fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    axes = axes.flatten()

    # for i in range(n_states):
    for i in range(2):
        idz = idzs[i]
        ax = axes[i]
        for trial in range(n_trials):
            ax.plot(k_vec, xhat_all[trial, :, idz], color="tab:blue")
        ax.set_ylabel(state_labels[idz])
        ax.grid(True)

    for ax in axes[-2:]:
        ax.set_xlabel(r"Measurement Index $k$")

    fig.suptitle(r"Recursive Least Squares State Estimates vs Measurement Index $k$")
    fig.tight_layout()

    return fig


# REQUIRED --- Problem 2b
def plot_and_describe_sample_mean():
    res = _get_results_RLS()
    xhat_all = res["xhat_all"] # (n_trials, n_meas, 6)
    P_all = res["P_all"] # (n_meas, 6, 6)

    n_trials, n_meas, n_states = xhat_all.shape
    k_vec = np.arange(1, n_meas + 1)

    # sample mean across trials
    mean_states = xhat_all.mean(axis=0) # (n_meas, 6)

    state_labels = [
        r"$x_0$ [km]",
        r"$y_0$ [km]",
        r"$z_0$ [km]",
        r"$\dot{x}_0$ [km/s]",
        r"$\dot{y}_0$ [km/s]",
        r"$\dot{z}_0$ [km/s]",
    ]

    # only display z-states
    idzs = (2, 5)
    # fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    axes = axes.flatten()

    # for i in range(n_states):
    for i in range(2):
        idz = idzs[i]
        ax = axes[i]
        mu_i = mean_states[:, idz]
        sigma_i = np.sqrt(P_all[:, idz, idz])

        ax.plot(k_vec, mu_i, label=r"Mean $\mu_k$ across trials", linewidth=2.5)
        ax.plot(k_vec, mu_i + 3.0 * sigma_i, "r--", label=r"$\mu_k \pm 3\sigma$")
        ax.plot(k_vec, mu_i - 3.0 * sigma_i, "r--")

        for trial in range(n_trials):
            ax.plot(k_vec, xhat_all[trial, :, idz], color="0.8", linewidth=0.5)

        ax.set_ylabel(state_labels[idz])
        ax.grid(True)
        ax.legend()

    for ax in axes[-2:]:
        ax.set_xlabel(r"Measurement Index $k$")

    fig.suptitle(r"RLS Sample Mean State Estimates and $\pm 3 \sigma$ Bounds")
    fig.tight_layout()

    description = dedent("""
        The plots show the sample mean state estimates μₖ across all 50 trials
        together with ±3σ envelopes derived from the RLS covariance Pₖ.

        Only the z-related states (z₀ and ż₀) change significantly with k,
        because the measurements depend only on z. The x, y, ẋ and ẏ
        components remain essentially fixed at their prior means and covariances,
        reflecting the fact that they are unobservable in this simplified setup.

        For the observable components, the behavior mirrors the batch least squares
        results from Problem 1: as k increases, the mean converges toward the true
        value and the ±3σ bounds shrink. This happens because recursive least
        squares is algebraically equivalent to batch least squares for a linear
        Gaussian model: RLS simply builds the same normal-equation solution one
        measurement at a time, using the prior (x̂₀, P₀) as an initial condition.
        Differences between the RLS and batch plots at small k come from the way
        the prior is incorporated and from finite-precision numerical effects,
        but they converge as more data are assimilated.
    """).strip()

    return fig, description


# REQUIRED --- Problem 2c
def plot_and_describe_time():
    # Problem 1 BLLS timings
    results_BLLS = _get_results_BLLS()
    avg_time_blls = results_BLLS["avg_time_per_k"] # (n_meas,)

    # Problem 2 RLS timings
    results_RLS = _get_results_RLS()
    avg_time_rls = results_RLS["avg_time_per_update"] # (n_meas,)

    n_meas = avg_time_blls.shape[0]
    k_vec = np.arange(1, n_meas + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # convert to microseconds
    ax.plot(k_vec, avg_time_blls * 1e6, label="Batch LS (per k)", linewidth=2)
    ax.plot(k_vec, avg_time_rls * 1e6, label="RLS (per update)", linewidth=2)

    ax.set_xlabel(r"Number of Measurements Used ($k$)")
    ax.set_ylabel("Average Execution Time [µs]")
    ax.set_title("Execution Time [µs] vs. Number of Measurements\nComparison: Batch LS vs RLS")
    ax.grid(True)
    ax.legend()

    description = dedent("""
        The batch least squares implementation recomputes the normal equations
        using all k measurements at each step. As a result, its computational
        cost per update grows roughly linearly with k, and the measured average
        execution time increases as more measurements are included.

        In contrast, the recursive least squares implementation updates the
        estimate and covariance using only the new measurement and the previous
        state (x̂ₖ₋₁, Pₖ₋₁). Each RLS update has essentially constant cost
        O(n²) in the state dimension n and does not depend on k, so the
        measured execution time per update remains nearly flat as k increases.

        For this problem the absolute differences in timing are small because
        both the measurement dimension and state dimension are modest. However,
        the scaling behavior is fundamentally different: for long data records
        or higher-dimensional states, RLS is significantly more efficient than
        repeatedly solving the batch least squares problem from scratch.
    """).strip()

    return fig, description


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def compute_final_x_and_P():
    res = _get_results_KF()
    x_filt = res["x_filt"]
    P_filt = res["P_filt"]

    x_final = x_filt[-1, :]
    P_final = P_filt[-1, :, :]
    return x_final, P_final


# REQUIRED --- Problem 3c
def plot_pure_prediction():
    res = _get_results_KF()
    x_pred = res["x_pred"]
    P_pred = res["P_pred"]
    dt = res["dt"]

    n_steps, n_states = x_pred.shape
    t_vec = np.arange(n_steps) * dt

    state_labels = [
        r"$x$ [km]",
        r"$y$ [km]",
        r"$z$ [km]",
        r"$\dot x$ [km/s]",
        r"$\dot y$ [km/s]",
        r"$\dot z$ [km/s]",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True)
    axes = axes.flatten()

    for i in range(n_states):
        ax = axes[i]
        mu_minus = x_pred[:, i]
        sigma_minus = np.sqrt(P_pred[:, i, i])

        ax.plot(t_vec, mu_minus, label=r"$\hat{\mu}_k^-$")
        ax.plot(t_vec, mu_minus + 3.0 * sigma_minus, "r--", label=r"$\pm 3\sigma$")
        ax.plot(t_vec, mu_minus - 3.0 * sigma_minus, "r--")
        ax.set_ylabel(state_labels[i])
        ax.grid(True)

        if i == 5:
            ax.legend(loc="upper right")

    for ax in axes[-2:]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(r"Kalman Filter Pure Predictions")
    fig.tight_layout()

    return fig


# REQUIRED --- Problem 3d
def plot_with_measurement_updates():
    res = _get_results_KF()
    x_filt = res["x_filt"]
    P_filt = res["P_filt"]
    dt = res["dt"]

    n_steps, n_states = x_filt.shape
    t_vec = np.arange(n_steps) * dt

    state_labels = [
        r"$x$ [km]",
        r"$y$ [km]",
        r"$z$ [km]",
        r"$\dot x$ [km/s]",
        r"$\dot y$ [km/s]",
        r"$\dot z$ [km/s]",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex=True)
    axes = axes.flatten()

    for i in range(n_states):
        ax = axes[i]
        mu_plus = x_filt[:, i]
        sigma_plus = np.sqrt(P_filt[:, i, i])

        ax.plot(t_vec, mu_plus, label=r"$\hat{\mu}_k^+$")
        ax.plot(t_vec, mu_plus + 3.0 * sigma_plus, "r--", label=r"$\pm 3\sigma$")
        ax.plot(t_vec, mu_plus - 3.0 * sigma_plus, "r--")
        ax.set_ylabel(state_labels[i])
        ax.grid(True)

        if i == 5:
            ax.legend(loc="upper right")

    for ax in axes[-2:]:
        ax.set_xlabel("Time [s]")

    fig.suptitle(r"Kalman Filter Corrected Estimates")
    fig.tight_layout()

    return fig


# REQUIRED --- Problem 3e
def describe_differences():
    description = dedent("""
        The pure prediction curves (μₖ⁻, Pₖ⁻) show the state evolution and
        uncertainty when only the process model and process noise are applied.
        Between measurements, the covariance Pₖ⁻ grows due to injected
        process noise Q at each step, reflecting increasing uncertainty about
        the spacecraft's position and velocity in the absence of new data.

        The measurement-updated curves (μₖ⁺, Pₖ⁺) show the effect of
        incorporating the noisy position measurements. At each update, the
        position components' covariance drops sharply relative to Pₖ⁻, often
        to values significantly below the measurement noise variance, because
        the filter fuses multiple measurements over time.

        Thus:
        - Pₖ⁻ typically increases between updates (model + process noise).
        - Pₖ⁺ is always less than or equal to Pₖ⁻ after each measurement.
        - μₖ⁺ tends to track the true trajectory more closely than μₖ⁻,
          which drifts when the model is driven only by process noise.
    """).strip()

    return description


# REQUIRED --- Problem 3f
def plot_and_describe_residuals():
    res = _get_results_KF()
    residuals = res["residuals_plus"] # shape (N, 3)
    dt = res["dt"]
    n_steps = residuals.shape[0]
    t_vec = np.arange(n_steps) * dt

    fig, ax = plt.subplots(3, 1, figsize=(8, 4), sharex=True)
    labels = [r"$\delta y_x$", r"$\delta y_y$", r"$\delta y_z$"]

    for i in range(3):
        ax[i].plot(t_vec, residuals[:, i])
        ax[i].axhline(0.0, color="k", linewidth=0.8)
        ax[i].set_ylabel(labels[i])
        ax[i].grid(True)

    ax[-1].set_xlabel("Time [s]")
    fig.suptitle(r"Measurement Residuals: $\delta \mathbf{y}_k = \mathbf{y}_k - H\hat{\mu}_k^+$")
    fig.tight_layout()

    # qualitative interpretation
    sigma_meas = np.sqrt(1e3)
    three_sigma = 3 * sigma_meas

    description = dedent(f"""
        The residuals δyₖ = yₖ - H xₖ⁺ represent the difference between the
        actual measurements and the measurement predicted by the updated state.

        For a well-tuned linear Kalman filter with correctly modeled process and
        measurement noise, these residuals should be approximately zero-mean,
        uncorrelated in time, and have a variance somewhat smaller than the
        raw measurement noise variance (because the filter has already used
        each measurement to refine the state estimate).

        In this problem, the nominal measurement noise standard deviation is
        σₘₑₐₛ ≈ {sigma_meas:.2f} [km], so a ±3σ band is about ±{three_sigma:.2f} [km].
        Thus, most residuals should lie within this range, and there should be no
        obvious deterministic trend over time. Any strong bias or systematic
        trend in the residuals would suggest either a modeling error or a mismatch
        between the assumed initial state/covariance and the actual conditions.
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
        f.write(f"Final filtered state estimate x_N^+ =\n{x_final}\n")
        f.write(f"Final filtered covariance P_N^+ =\n{P_final}")
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
