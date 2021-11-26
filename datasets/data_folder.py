"""
Data folder class
"""
import os
import os.path

# For now only support preprocessed .tsv file with omics matrix
OMICS_EXTENSIONS = [
    '.tsv',
    '.csv',
    '.h5'
]


def is_omics_file(filename):
    return any(filename.endswith(extension) for extension in OMICS_EXTENSIONS)


def make_dataset(directory):
    """
    Return paths of omics files in the dataset
    """
    data_paths = []
    assert os.path.isdir(directory), '%s is not a valid directory!' % directory

    for root, _, file_names in sorted(os.walk(directory)):
        for file_name in file_names:
            if is_omics_file(file_name):
                file_path = os.path.join(root, file_name)
                data_paths.append(file_path)
    return data_paths
