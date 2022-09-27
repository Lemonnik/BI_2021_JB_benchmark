from datasets.davisDataset import Davis
from datasets.dtiMinorDataset import DtiMinor
# import your dataset here


def select_dataset(base_path, dataset_params):
    """
       Dataset selection must be stated here.
    """

    dataset_name = dataset_params.name

    if dataset_name == 'Davis':
        dataset = Davis(base_path,
                        force_download=dataset_params.force_download,
                        load_from_raw=dataset_params.load_from_raw)
    elif dataset_name == 'dtiMinor':
        dataset = DtiMinor(base_path,
                           force_download=dataset_params.force_download,
                           load_from_raw=dataset_params.load_from_raw)
    ###########################################################################
    # ==================== INITIALIZE YOUR DATASET HERE ===================== #
    # elif dataset_name == 'Your dataset name':
    #   dataset = ...
    ###########################################################################
    else:
        raise ValueError(f'No dataset is stated for {dataset_name} in select_dataset.py file.')

    return dataset
