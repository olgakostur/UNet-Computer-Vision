from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_dataloader: Dataset,
        validation_dataloader: Optional[Dataset] = None,
        lr_scheduler: torch.optim.lr_scheduler = None,
        epochs: int = 100,
        epoch: int = 0,
        notebook: bool = False,
    ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []

    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_dataloader is not None:
                self._validate()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if (
                    self.validation_dataloader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    self.lr_scheduler.batch(
                        self.validation_loss[i]
                    )  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.training_dataloader),
            "Training",
            total=len(self.training_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input_x, target_y = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input_x)  # one forward pass
            loss = self.criterion(out, target_y)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(
                f"Training: (loss {loss_value:.4f})"
            )  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_dataloader),
            "Validation",
            total=len(self.validation_dataloader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(
                self.device
            )  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()

def plot_training(
    training_losses,
    validation_losses,
    learning_rate,
    gaussian=True,
    sigma=2,
    figsize=(8, 6),
):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines["top"].set_visible(False)
        subfig.spines["right"].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = "."
        color_original_train = "lightcoral"
        color_original_valid = "lightgreen"
        color_smooth_train = "red"
        color_smooth_valid = "green"
        alpha = 0.25
    else:
        linestyle_original = "-"
        color_original_train = "red"
        color_original_valid = "green"
        alpha = 1.0

    # Subfig 1
    subfig1.plot(
        x_range,
        training_losses,
        linestyle_original,
        color=color_original_train,
        label="Training",
        alpha=alpha,
    )
    subfig1.plot(
        x_range,
        validation_losses,
        linestyle_original,
        color=color_original_valid,
        label="Validation",
        alpha=alpha,
    )
    if gaussian:
        subfig1.plot(
            x_range,
            training_losses_gauss,
            "-",
            color=color_smooth_train,
            label="Training",
            alpha=0.75,
        )
        subfig1.plot(
            x_range,
            validation_losses_gauss,
            "-",
            color=color_smooth_valid,
            label="Validation",
            alpha=0.75,
        )
    subfig1.title.set_text("Training & validation loss")
    subfig1.set_xlabel("Epoch")
    subfig1.set_ylabel("Loss")

    subfig1.legend(loc="upper right")

    # Subfig 2
    subfig2.plot(x_range, learning_rate, color="black")
    subfig2.title.set_text("Learning rate")
    subfig2.set_xlabel("Epoch")
    subfig2.set_ylabel("LR")

    return fig


