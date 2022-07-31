import numpy as np

from preprocess.CPI_2018_preprocess import cpi_preprocess


def mhsadti_preprocess(dataset, radius, ngram, graph_max, seq_max):
    dataset = cpi_preprocess(dataset,
                             radius=radius,
                             ngram=ngram)

    compounds = dataset.features['compounds']
    adjacencies = dataset.features['adjacencies']
    proteins = dataset.features['proteins']

    print("max protein len in dataset:", max([i[0] for i in proteins]))
    print("max adj len in dataset:", max([len(i) for i in adjacencies]))
    print("graph_max is set to:", graph_max, "\tseq_max is set to:", seq_max)

    all_compounds = []
    all_compounds_mask = []
    all_adjs = []
    all_proteins = []
    all_proteins_mask = []
    for cmp in compounds:
        cmp_len = len(cmp)
        if cmp_len > graph_max:
            res = cmp[:graph_max]
            mask = [1] * graph_max
        else:
            res = cmp.tolist() + [0] * (graph_max - cmp_len)
            mask = [1] * cmp_len + [0] * (graph_max - cmp_len)
        all_compounds.append(res)
        all_compounds_mask.append(mask)

    for adj in adjacencies:
        if len(adj) > graph_max:
            adj = adj[:graph_max, :graph_max]
        pad = np.zeros([graph_max, graph_max], dtype=np.int32)
        pad[np.where(adj > 0)[0], np.where(adj > 0)[1]] = 1
        all_adjs.append(pad)

    for pro in proteins:
        pro_len = len(pro)
        if pro_len > seq_max:
            pro = pro[:seq_max]
            mask = [1] * seq_max
        else:
            pro = pro.tolist() + [0] * (seq_max - pro_len)
            mask = [1] * pro_len + [0] * (seq_max - pro_len)
        all_proteins.append(pro)
        all_proteins_mask.append(mask)

    """Add features to dataset."""
    dataset.add_feature(feat_name="compounds_mhsadti", feat_values=all_compounds)
    dataset.add_feature(feat_name="compound_masks_mhsadti", feat_values=all_compounds_mask)
    dataset.add_feature(feat_name="adjacencies_mhsadti", feat_values=all_adjs)
    dataset.add_feature(feat_name="proteins_mhsadti", feat_values=proteins)
    dataset.add_feature(feat_name="protein_masks_mhsadti", feat_values=all_proteins_mask)

    """Return result dataset"""
    return dataset
