import os

output_file_path = "diagrams/output_tree.txt"

# Excluded directories and files
excluded_contents = [
    # , 'db'
    # , 'workers'
    # , 'README.md'
    'documentation'
    # , 'classes'
    # , 'main.py'
    , 'config.ini'
    , 'output'
    , '.gitignore'
    #, 'fingerprints'
    , 'test'
    , 'backup'
    , 'datasets'
    , 'resources'
    #, 'diagrams'
    #, 'notebooks'
    #, 'workloads'
    #, 'data-pickle'
    , 'server.json'
    , 'tmp'
    , '.git'
]
excluded_files = [ #inner files
    'report'
    , 'example_dissertation_cardiff_univ_report.pdf'
    , 'image_forensics'
    , '__pycache__'
    , 'StyleContentModel.py'
]

shorten_folders = [
      'fingerprints'
    , 'workloads'

]

project_contents = sorted(os.listdir())
project_contents = [content for content in project_contents if content not in excluded_contents]

with open(output_file_path, 'w') as output_file:
    for folder in project_contents:
        if '.' in folder:
            continue
        print(folder, file=output_file)
        folder_contents = os.listdir(folder)
        folder_contents = [_file for _file in folder_contents if _file not in excluded_files]
        folder_contents = folder_contents[:3] if folder in shorten_folders else folder_contents
        for content in folder_contents:
            print(f'   |________ {content}', file=output_file)
        if folder in shorten_folders:
            print(f'   |________ [ ... ]', file=output_file)


    for folder in project_contents:
        if '.' not in folder:
            continue
        print(folder, file=output_file)
