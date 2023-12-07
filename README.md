# Intro
Repository for personal work on a MSc dissertation at Cardiff University. Program Data Science and Analytics.
The purpose is to review and study the current work and publications in the field of image forensics, in particular PRNU strategies and their relation with deep learning techniques.

# Dataset
Dresden and Korus datasets, containing pictures from identified camera devices, and tampered pictures.

# Repository structure
The respository is structured as follows:
- ```classes``` Package containing all wrappers used for this project
- ```scripts/``` one-off scripts for dataset manipulation
- ```docs/``` dissertation report done with RStudio and Markdown
- ```datasets/``` Korus and Dresden dataset pictures

# Test results
- For a detailed description of the experiments and results obtained please refer to our [report](/docs/report/report.pdf).

# Example usage

## Run Workloads

To run workloads, use the following command:

python main.py --instruction run_workloads --wl_state <optional_workload_state> --wl_ID <optional_workload_ID> --experiment <optional_experiment_ID>
wl_state: Optional. Specify the state of the workloads to run (default is 'awaiting').
wl_ID: Optional. Specify the ID of a specific workload to run.
experiment: Optional. Specify the ID of a specific experiment.

Find Workload
To find a specific workload, use the following command:

- python main.py --instruction find_workload --wl_ID <workload_ID>
wl_ID: Required. Specify the ID of the workload to find.

Generate Workloads
To generate new workloads, use the following command:

- python main.py --instruction generate_workloads --experiment <experiment_ID>
experiment: Required. Specify the ID of the experiment for which workloads should be generated.

Delete Workloads by Experiment
To delete workloads based on an experiment, use the following command:

- python main.py --instruction delete_workloads_by_experiment --experiment <experiment_ID>
experiment: Required. Specify the ID of the experiment to delete workloads.

Usage Example
Here's an example to generate workloads:

- python main.py --instruction generate_workloads --experiment 7
This command generates workloads for experiment 7.

# Bibliography















