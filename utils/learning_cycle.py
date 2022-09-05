import timeit
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


class Trainer(object):
    """
    Function to train model.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    lr : float
        Learning rate for optimizer.
    weight_decay : float
        Weight decay for optimizer.
    device : str
        Whether to use GPU acceleration. Possibles are 'cuda' and 'cpu'.
    """
    def __init__(self, model, lr: float, weight_decay: float, device: str) -> None:
        self.model = model
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)

    def train(self, dataset, batch_size: int) -> float:
        """
        Train cycle for one epoch.

        Parameters
        ----------
        dataset
            Any dataset compatible with our DTI benchmark (inherited from DtiDataset Class).
        batch_size : int
        """
        dataset.return_type = self.model.return_type
        dataset.mode = 'train'
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        self.model.train()

        loss_total = 0

        for batch in loader:
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()
        return loss_total


class Tester(object):
    """
    Function to test model.

    Parameters
    ----------
    model
        Pretrained model compatible with our DTI benchmark (inherited from BaseModel Class).
    """
    def __init__(self, model) -> None:
        self.model = model

    def test(self, dataset, batch_size: int) -> (float, float, float):
        """
        Test cycle for one epoch.

        Parameters
        ----------
        dataset
            Any dataset compatible with our DTI benchmark (inherited from DtiDataset Class).
        batch_size : int

        Returns
        -------
        auc_test : float
        precision : float
        recall : float
        """
        dataset.return_type = self.model.return_type
        dataset.mode = 'test'
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        self.model.eval()

        T, Y, S = [], [], []
        for params in loader:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(params, train=False)
            T.extend(correct_labels)
            Y.extend(predicted_labels)
            S.extend(predicted_scores)
        auc_test = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return auc_test, precision, recall


def run_model(model,
              dataset,
              device: str,
              batch_size: int = 64,
              n_epochs: int = 100,
              print_every: int = 10,
              lr_decay: float = 0.5,
              decay_interval: int = 10,
              lr: float = 0.7e-3,
              weight_decay: float = 1e-6,
              **kwargs) -> None:
    """
    Main function for running training and testing cycles.

    Parameters
    ----------
    model
        Any model compatible with our DTI benchmark (inherited from BaseModel Class).
    dataset
        Any dataset compatible with our DTI benchmark (inherited from DtiDataset Class).
    device : str
        Whether to use GPU acceleration. Possibles are 'cuda' and 'cpu'.
    batch_size : int
        Batch size.
    n_epochs : int
        Number of epochs.
    print_every : int
        How often program will print all the main information
        (epoch number; time spent; train loss; auc, precision and recall metrics in test).
        print_every=1 means 'print info about each epoch'
    lr_decay : float
        Learning rate will be multiplied (*) by lr_decay every decay_interval epoch.
    decay_interval : int
        Learning rate will be multiplied (*) by lr_decay every decay_interval epoch.
    lr : float
        Optimizers' learning rate.
    weight_decay : float

    """

    trainer = Trainer(model, lr, weight_decay, device)
    tester = Tester(model)

    AUCs = ('Epoch    '
            'Time(sec)     '
            'Loss_train     '
            'AUC_test     '
            'Precision_test    '
            'Recall_test')

    """ Start training. """
    print('--- Training ---')
    print(AUCs)
    start = timeit.default_timer()

    for epoch in range(1, n_epochs+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset, batch_size)
        # AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset, batch_size)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train,
                AUC_test, precision_test, recall_test]

        if epoch % print_every == 0:
            print(f'{epoch:>5d} '
                  f'{time:>11.3f} '
                  f'{loss_train:>15.5f} '
                  f'{AUC_test:>12.5f} '
                  f'{precision_test:>18.5f} '
                  f'{recall_test:>14.5f} ')
