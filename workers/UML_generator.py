# Change working directory to project root
import os
os.chdir('../')
print(os.getcwd())

import subprocess

def run_pyreverse(target_folder, ignore_list, output_folder, output_format, project, colorized, only_classnames, class_keywords):
    #python_files = [os.path.join(root, file) for root, dirs, files in os.walk(target_folder) for file in files if file.endswith('.py')]
    python_files = [os.path.join(root, file) for root, dirs, files in os.walk(target_folder) for file in files if file.endswith('.py')]
    os.makedirs(output_folder, exist_ok=True)

    def keywordMatch(_file, keyword_list):
        for keyword in keyword_list:
            if keyword in _file:
                return True
        return False

    python_files = [file for file in python_files if keywordMatch(file, class_keywords)]
    python_files = [file for file in python_files if file.split('/')[-1] not in ignore_list]

    command = [
                'pyreverse'
                ,'--ignore=' + ','.join(ignore_list)
                ,'--output-directory=' + output_folder
                ,'-o'
                ,output_format]

    if project:
        command += ['--project=' + project]
    if colorized:
        command += ['--colorized']
    if only_classnames:
        command += ['--only-classnames']

    command += python_files

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running pyreverse: {e}")


if __name__ == "__main__":
    target_folder = '/home/joseca/mat099/'
    ignore_list = [
        # Files
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
        # Classes
        ,'StyleContentModel'
    ]

    output_folder   = '/home/joseca/mat099/diagrams/'
    output_format   = 'png'
    project         = 'PRNU Processor'
    colorized       = True
    only_classnames = True
    class_keywords  = ['PRNU', 'Image', 'SQL', 'StyleTransfer', 'Workload', 'Pickle']

    run_pyreverse(
                 target_folder   = target_folder
                ,ignore_list     = ignore_list
                ,output_folder   = output_folder
                ,output_format   = output_format
                ,project         = project
                ,colorized       = colorized
                ,only_classnames = only_classnames
                ,class_keywords  = class_keywords
                )
