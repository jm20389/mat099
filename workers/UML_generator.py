# Change working directory to project root
import os
os.chdir('../')
print(os.getcwd())

import subprocess

def run_pyreverse(target_folder, ignore_list, output_folder, output_format, project, colorized, only_classnames):
    # Get a list of all Python files in the target folder and its subfolders
    #python_files = [os.path.join(root, file) for root, dirs, files in os.walk(target_folder) for file in files if file.endswith('.py')]
    python_files = [os.path.join(root, file) for root, dirs, files in os.walk(target_folder) for file in files if file.endswith('.py')]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Build the command with the Python files, ignore list, output folder, format, orientation, and simplified option
    ignore_list = ','.join(ignore_list)
    command = [
                'pyreverse'
                ,'--ignore ' + ignore_list#','.join(ignore_list)
                ,'--output-directory=' + output_folder
                ,'-o'
                ,output_format]
    if project:
        command += ['--project=' + project]
    if colorized:
        command += ['--colorized']
    command += python_files
    if only_classnames:
        command += ['--only-classnames']
    #command += python_files

    try:
        # Run the command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pyreverse: {e}")

# Example usage:
if __name__ == "__main__":
    target_folder = '/home/joseca/mat099/'
    ignore_list = [
        'backup',
        'PRNUProcessor_old.py',
        'PRNUProcessor_old2_091123.py',
        'workloadRunner.py',
        '.gitignore',
        'CMT307.ipynb',
        'UML_generator.py',
        'gan_test02.ipynb',
        'PRNU_test01.ipynb',
        'test01_general.py',
        'EDA_vision.ipynb',
        'datasets',
        'resources',
        'diagrams',
        'ftp_test.py',
        'worker_rename_files.py',
        'style_transfer_test02.ipynb',
        'style_transfer_test01.ipynb',
        'style_transfer_generated_pictures',
        'PRNU_test02.ipynb',
        'workloadGenerator.py',
        'initial_test01.ipynb',
        'download_vision.py',
        'documentation',
        '.git',
        'PRNU_test.py',
        'EDA.ipynb',
        'ftp_test01.ipynb'
    ]

    output_folder = '/home/joseca/mat099/diagrams/'
    output_format = 'jpg'
    project = 'PRNU Processor'
    colorized = True
    only_classnames = True

    run_pyreverse(
                 target_folder   = target_folder
                ,ignore_list     = ignore_list
                ,output_folder   = output_folder
                ,output_format   = output_format
                ,project         = project
                ,colorized       = colorized
                ,only_classnames = only_classnames
                )
