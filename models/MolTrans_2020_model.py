from models.BaseModel import DTI_model


class MolTrans_(DTI_model):
    """
    Implementation of CPI_prediction model detailed in 2018 paper by Tsubaki M, et al.

    References
    ----------
    * Tsubaki M., Tomii K., Sese J.
      Compound-protein interaction prediction with
      end-to-end learning of neural networks for graphs and sequences.
      Bioinformatics. 2019; 35(2): 309-318.
      doi:10.1093/bioinformatics/bty535

    Notes
    -----
    All code is taken from authors' orignal github with small changes
    https://github.com/masashitsubaki/CPI_prediction

    Parameters
    ----------
    dim: int
        Dimension of embedding space.
    n_filters: int
        Number of filters used for convolution.
    n_entities: int
        Number of entities in the current data set.
    n_relations: int
        Number of relations in the current data set.

    """