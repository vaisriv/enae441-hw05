import matplotlib.pyplot as plt
import numpy as np


def load_numpy_data(file_path):
    Y_all = np.load(file_path)
    return Y_all


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


# NOTE:
# If you find yourself re-writing a lot of the code for the different sub-problems, consider writing one larger function at the top of the script that returns a dictionary with the appropriate data for each sub-problem.

# E.g. a function like:
# def run_BLLS()
#    ...
#   results_dict = {
#       'x_all': x_all,
#       'timing_all': timing_all,
#       ...
#   }
#    return results_dict


# Then each required function can just call that larger function and extract the relevant data from the dictionary. This will help avoid code duplication and make it easier to debug.

# results_BLLS = run_BLLS()

# def plot_batch_least_squares_single_trial():
#     x_all = results_BLLS['x_all']
#     ...


#######################
# Problem 1
#######################


# REQUIRED --- 1b
def plot_batch_least_squares_single_trial():
    return fig


# REQUIRED --- 1c
def plot_batch_least_squares_all_trials():
    return fig


# REQUIRED --- 1d
def plot_state_estimate_histograms():
    figs = []
    description = """
        Write your answer here.
    """
    return figs, description


# REQUIRED --- 1e
def plot_execution_time_vs_measurements():
    return fig


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2a
def plot_recursive_lease_squares():
    return fig


# REQUIRED --- Problem 2b
def plot_and_describe_sample_mean():
    fig = None
    description = """
        Write your answer here
    """
    return fig, description


# REQUIRED --- Problem 2c
def plot_and_describe_time():
    fig = None
    description = """
        Write your answer here.
    """
    return fig, description


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def compute_final_x_and_P():
    return x_final, P_final


# REQUIRED --- Problem 3c
def plot_pure_prediction():
    return fig


# REQUIRED --- Problem 3d
def plot_with_measurement_updates():
    return fig


# REQUIRED --- Problem 3e
def describe_differences():
    description = """
        Write your answer here.
    """
    return description


# REQUIRED --- Problem 3f
def plot_and_describe_residuals():
    fig = None
    description = """
        Write your answer here.
        """
    return fig, description


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    plot_batch_least_squares_single_trial()
    plot_batch_least_squares_all_trials()
    plot_state_estimate_histograms()
    plot_execution_time_vs_measurements()

    # Problem 2
    plot_recursive_lease_squares()
    plot_and_describe_sample_mean()
    plot_and_describe_time()

    # Problem 3
    compute_final_x_and_P()
    plot_pure_prediction()
    plot_with_measurement_updates()
    describe_differences()
    plot_and_describe_residuals()

    plt.show()


if __name__ == "__main__":
    main()
