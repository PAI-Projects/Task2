import os

import numpy as np
import torch
import torch.optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn import functional as F
from tqdm import trange

from models import DenseNet, BayesNet
from util import ece

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_solution(dataset_train: torch.utils.data.Dataset, data_dir: str = os.curdir, output_dir: str = '/results/') -> 'Model':
    """
    Run your task 2 solution.
    This method should train your model, evaluate it, and return the trained model at the end.
    Make sure to preserve the method signature and to return your trained model,
    else the checker will fail!

    :param dataset_train: Training dataset
    :param data_dir: Directory containing the datasets
    :return: Your trained model
    """

    # Create model
    model = Model()

    # Train the model
    print('Training model')
    model.train(dataset_train)

    # Predict using the trained model
    print('Evaluating model on training data')
    eval_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=64, shuffle=False, drop_last=False
    )
    evaluate(model, eval_loader, data_dir, output_dir)

    # IMPORTANT: return your model here!
    return model


class Model(object):
    """
    Task 2 model that can be used to train a BNN using Bayes by backprop and create predictions.
    You need to implement all methods of this class without changing their signature,
    else the checker will fail!
    """

    def __init__(self):
        # Hyperparameters and general parameters
        # You might want to play around with those
        self.num_epochs = 50  # number of training epochs
        self.batch_size = 128  # training batch size
        learning_rate = 1e-4  # training learning rates
        hidden_layers = (100, 100)  # for each entry, creates a hidden layer with the corresponding number of units
        use_densenet = False  # set this to True in order to run a DenseNet for comparison
        self.print_interval = 100  # number of batches until updated metrics are displayed during training

        # Determine network type
        if use_densenet:
            # DenseNet
            print('Using a DenseNet model for comparison')
            self.network = DenseNet(in_features=28 * 28, hidden_features=hidden_layers, out_features=10)
        else:
            # BayesNet
            print('Using a BayesNet model')
            self.network = BayesNet(in_features=28 * 28, hidden_features=hidden_layers, out_features=10)

        self.network.to(DEVICE)

        # Optimizer for training
        # Feel free to try out different optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def train(self, dataset: torch.utils.data.Dataset):
        """
        Train your neural network.
        If the network is a DenseNet, this performs normal stochastic gradient descent training.
        If the network is a BayesNet, this should perform Bayes by backprop.

        :param dataset: Dataset you should use for training
        """

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        self.network.train()

        # print(list(self.network.parameters()))

        progress_bar = trange(self.num_epochs)
        for _ in progress_bar:
            num_batches = len(train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)
                batch_x = batch_x.to(DEVICE).float()
                batch_y = batch_y.to(DEVICE).long()

                self.network.zero_grad()

                if isinstance(self.network, DenseNet):
                    # DenseNet training step

                    # Perform forward pass
                    current_logits = self.network(batch_x)

                    # Calculate the loss
                    # We use the negative log likelihood as the loss
                    # Combining nll_loss with a log_softmax is better for numeric stability
                    loss = F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y, reduction='sum')

                    # Backpropagate to get the gradients
                    loss.backward()
                else:
                    # BayesNet training step via Bayes by backprop
                    assert isinstance(self.network, BayesNet)

                    NR_SAMPLES = 3

                    output_features = torch.zeros((NR_SAMPLES, batch_x.shape[0], 10))
                    log_prior = torch.zeros(NR_SAMPLES)
                    log_variational_posterior = torch.zeros(NR_SAMPLES)
                    for i in range(NR_SAMPLES):
                        output_features[i], log_prior[i], log_variational_posterior[i] = self.network(batch_x)

                    output_features = output_features.mean(dim=0).to(DEVICE)
                    log_prior = log_prior.mean().to(DEVICE)
                    log_variational_posterior = log_variational_posterior.mean().to(DEVICE)

                    # # q(w | theta) - log P(w) - log P(D | w)
                    loss = (log_variational_posterior - log_prior) / batch_x.shape[0] \
                           + F.nll_loss(F.log_softmax(output_features, dim=1), batch_y, reduction='sum')

                    loss.backward()

                self.optimizer.step()

                # Update progress bar with accuracy occasionally
                if batch_idx % self.print_interval == 0:
                    if isinstance(self.network, DenseNet):
                        current_logits = self.network(batch_x)
                    else:
                        assert isinstance(self.network, BayesNet)
                        current_logits, _, _ = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item(),
                                             log_variational_posterior=log_variational_posterior.detach().cpu().numpy(),
                                             log_prior=log_prior.detach().cpu().numpy())

    def predict(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Predict the class probabilities using your trained model.
        This method should return an (num_samples, 10) NumPy float array
        such that the second dimension sums up to 1 for each row.

        :param data_loader: Data loader yielding the samples to predict on
        :return: (num_samples, 10) NumPy float array where the second dimension sums up to 1 for each row
        """

        self.network.eval()

        probability_batches = []
        for batch_x, batch_y in data_loader:
            current_probabilities = self.network.predict_probabilities(batch_x).detach().numpy()
            probability_batches.append(current_probabilities)

        output = np.concatenate(probability_batches, axis=0)
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 10
        assert np.allclose(np.sum(output, axis=1), 1.0)
        return output


def evaluate(model: Model, eval_loader: torch.utils.data.DataLoader, data_dir: str, output_dir: str):
    """
    Evaluate your model.
    :param model: Trained model to evaluate
    :param eval_loader: Data loader containing the training set for evaluation
    :param data_dir: Data directory from which additional datasets are loaded
    :param output_dir: Directory into which plots are saved
    """

    # Predict class probabilities on test data
    predicted_probabilities = model.predict(eval_loader)

    # Calculate evaluation metrics
    predicted_classes = np.argmax(predicted_probabilities, axis=1)
    actual_classes = eval_loader.dataset.tensors[1].detach().numpy()
    accuracy = np.mean((predicted_classes == actual_classes))
    ece_score = ece(predicted_probabilities, actual_classes)
    print(f'Accuracy: {accuracy.item():.3f}, ECE score: {ece_score:.3f}')

    if EXTENDED_EVALUATION:
        eval_samples = eval_loader.dataset.tensors[0].detach().numpy()

        # Determine confidence per sample and sort accordingly
        confidences = np.max(predicted_probabilities, axis=1)
        sorted_confidence_indices = np.argsort(confidences)

        # Plot samples your model is most confident about
        print('Plotting most confident MNIST predictions')
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(eval_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {predicted_classes[sample_idx]}, actual {actual_classes[sample_idx]}')
                bar_colors = ['C0'] * 10
                bar_colors[actual_classes[sample_idx]] = 'C1'
                ax[row + 1, col].bar(
                    np.arange(10), predicted_probabilities[sample_idx], tick_label=np.arange(10), color=bar_colors
                )
        fig.suptitle('Most confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'mnist_most_confident.pdf'))

        # Plot samples your model is least confident about
        print('Plotting least confident MNIST predictions')
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(eval_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {predicted_classes[sample_idx]}, actual {actual_classes[sample_idx]}')
                bar_colors = ['C0'] * 10
                bar_colors[actual_classes[sample_idx]] = 'C1'
                ax[row + 1, col].bar(
                    np.arange(10), predicted_probabilities[sample_idx], tick_label=np.arange(10), color=bar_colors
                )
        fig.suptitle('Least confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'mnist_least_confident.pdf'))

        print('Plotting ambiguous and rotated MNIST confidences')
        ambiguous_samples = torch.from_numpy(np.load(os.path.join(data_dir, 'test_x.npz'))['test_x']).reshape([-1, 784])[:10]
        ambiguous_dataset = torch.utils.data.TensorDataset(ambiguous_samples, torch.zeros(10))
        ambiguous_loader = torch.utils.data.DataLoader(
            ambiguous_dataset, batch_size=10, shuffle=False, drop_last=False
        )
        ambiguous_predicted_probabilities = model.predict(ambiguous_loader)
        ambiguous_predicted_classes = np.argmax(ambiguous_predicted_probabilities, axis=1)
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = 5 * row // 2 + col
                ax[row, col].imshow(np.reshape(ambiguous_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {ambiguous_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), ambiguous_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Predictions on ambiguous and rotated MNIST', size=20)
        fig.savefig(os.path.join(output_dir, 'ambiguous_rotated_mnist.pdf'))

        # Do the same evaluation as on MNIST also on FashionMNIST
        print('Predicting on FashionMNIST data')
        fmnist_samples = torch.from_numpy(np.load(os.path.join(data_dir, 'fmnist.npz'))['x_test']).reshape([-1, 784])
        fmnist_dataset = torch.utils.data.TensorDataset(fmnist_samples, torch.zeros(fmnist_samples.shape[0]))
        fmnist_loader = torch.utils.data.DataLoader(
            fmnist_dataset, batch_size=64, shuffle=False, drop_last=False
        )
        fmnist_predicted_probabilities = model.predict(fmnist_loader)
        fmnist_predicted_classes = np.argmax(fmnist_predicted_probabilities, axis=1)
        fmnist_confidences = np.max(fmnist_predicted_probabilities, axis=1)
        fmnist_sorted_confidence_indices = np.argsort(fmnist_confidences)

        # Plot FashionMNIST samples your model is most confident about
        print('Plotting most confident FashionMNIST predictions')
        most_confident_indices = fmnist_sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(fmnist_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {fmnist_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), fmnist_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Most confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'fashionmnist_most_confident.pdf'))

        # Plot FashionMNIST samples your model is least confident about
        print('Plotting least confident FashionMNIST predictions')
        least_confident_indices = fmnist_sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_idx = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(np.reshape(fmnist_samples[sample_idx], (28, 28)), cmap='gray')
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f'predicted {fmnist_predicted_classes[sample_idx]}')
                ax[row + 1, col].bar(
                    np.arange(10), fmnist_predicted_probabilities[sample_idx], tick_label=np.arange(10)
                )
        fig.suptitle('Least confident predictions', size=20)
        fig.savefig(os.path.join(output_dir, 'fashionmnist_least_confident.pdf'))

        print('Determining suitability of your model for OOD detection')
        all_confidences = np.concatenate([confidences, fmnist_confidences])
        dataset_labels = np.concatenate([np.ones_like(confidences), np.zeros_like(fmnist_confidences)])
        print(
            'AUROC for MNIST vs. FashionMNIST OOD detection based on confidence: '
            f'{roc_auc_score(dataset_labels, all_confidences):.3f}'
        )
        print(
            'AUPRC for MNIST vs. FashionMNIST OOD detection based on confidence: '
            f'{average_precision_score(dataset_labels, all_confidences):.3f}'
        )


def main():
    # raise RuntimeError(
    #     'This main method is for illustrative purposes only and will NEVER be called by the checker!\n'
    #     'The checker always calls run_solution directly.\n'
    #     'Please implement your solution exclusively in the methods and classes mentioned in the task description.'
    # )

    # Load training data
    data_dir = os.curdir
    output_dir = os.curdir
    raw_train_data = np.load(os.path.join(data_dir, 'train_data.npz'))
    x_train = torch.from_numpy(raw_train_data['train_x']).reshape([-1, 784])
    y_train = torch.from_numpy(raw_train_data['train_y']).long()
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    # Run actual solution
    run_solution(dataset_train, data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
