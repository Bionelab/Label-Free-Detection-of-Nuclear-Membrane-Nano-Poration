import numpy as np
import torch
import tqdm
import wandb

def eval_batch(x, model, agg_metrics, train=1):
    """
    Evaluate a single dataloader batch.
    Agg_metrics is a dict keeping a running sum of eval statistics over the epoch.
    Args:
        x (tensor): batch of input data to be passed to model.
        model: the model that can forward pass `x` and has function attribute
            `loss` that can be called.
    Returns:
        loss (tensor) shape 0. Loss that can be backpropped on.
        agg_metrics (tensor): same object that was passed as Arg that is updated
            with the new running total after this batch.
    """
    # forward pass & loss
    n_samples = len(x)
    x, y, mu, logvar = model(x)
    loss_dict = model.loss(x, y, mu, logvar)
    loss = loss_dict["loss"] * n_samples

    # aggregate statistics
    agg_metrics["n_samples"] += n_samples
    agg_metrics["loss"] += loss.item()
    agg_metrics["loss_recon"] += loss_dict["loss_recon"] * n_samples
    agg_metrics["loss_kl"] += loss_dict["loss_kl"] * n_samples
    agg_metrics["beta"] = loss_dict["beta"]

    return loss, agg_metrics


def log_metrics(
    epoch, agg_metrics, batch_idx=0, train=1, do_wandb=0, do_progress_bar=0, tq=None
):
    """
    Log metrics. If training, then log metrics at batch intervals defined (defined
    by the config parameter
    Optionally log to wandb, or update a tqdm progress bar.
    """
    # if using wandb in training, only update on certain increments, otherwise
    # it may slow things down https://docs.wandb.ai/guides/technical-faq#will-wandb-slow-down-my-training
    if (
        train
        and do_wandb
        and batch_idx % wandb.config["logging"]["train_batch_freq"] != 0
    ):
        return

    # agg the metrics
    n_samples = agg_metrics["n_samples"]
    loss = agg_metrics["loss"] / n_samples
    loss_recon = agg_metrics["loss_recon"] / n_samples
    loss_kl = agg_metrics["loss_kl"] / n_samples
    beta = agg_metrics["beta"]

    # tqdm progress bar
    if do_progress_bar:
        train_test = "Train" if train else "**** Test"
        msg = (
            f"{train_test} epoch {epoch} | loss: {loss:.5g}; recon: {loss_recon:.5g}; "
            f"kl : {loss_kl:.5g}; beta {beta}"
        )
        tq.set_description(msg)

    # wandb logging
    if do_wandb:
        loss_recon_per_pixel = loss_recon / wandb.config["data"]["n_pixels"]
        if train:
            step = epoch * wandb.config["data"]["n_loader_train"] + batch_idx
            wandb.log(
                step=step,
                data=dict(
                    epoch=epoch,
                    train_loss=loss,
                    train_loss_recon=loss_recon,
                    train_loss_kl=loss_kl,
                    loss_recon_per_pixel=loss_recon_per_pixel,
                    beta=beta,
                ),
            )
        else:
            # Validation: called once per validation epoch.
            # called on the same step as the last train run
            step = wandb.run.step
            wandb.log(
                step=step,
                data=dict(
                    epoch=epoch,
                    valid_loss=loss,
                    valid_loss_recon=loss_recon,
                    valid_loss_kl=loss_kl,
                    valid_recon_per_pixel=loss_recon_per_pixel,
                    beta=beta,
                ),
            )

    return

def eval_slow(epoch, model, loader_train, loader_test, do_wandb=1, device="cuda"):
    """
    Validation methods that may take a while to run, so in a separate function to be
    called less frequently.
    Do eval methods that are time consuming like plot generating, model fitting
    and so on.

    Currently nothing is implemented, but you can use it to save reconstruction grids for example
    """
    if not do_wandb:
        return
    model.eval()
    with torch.no_grad():
        step = wandb.run.step
        # f_recons_train=utils.
        # f_recons_test=utils.
        # wandb.log(step=step, data=dict(epoch=epoch, f_recons_train=f_recons_train, f_recons_test=f_recons_test))

def train(
    epoch,
    model,
    loader_train,
    optimizer,
    do_wandb=0,
    do_progress_bar=1,
    device="cuda",
    batch_lim=None,
):
    """Training loop for one epoch."""
    model.train()
    model.to(device)
    agg_metrics = dict(n_samples=0, loss=0, loss_recon=0, loss_kl=0, beta=0)

    if do_wandb:
        wandb.config["data"]["n_loader_train"] = len(loader_train)
        wandb.config["data"]["n_pixels"] = np.product(loader_train.dataset[0][0].shape)

    if do_progress_bar:
        t = tqdm.tqdm(loader_train, desc=f"Train Epoch {epoch}")
    else:
        t = loader_train

    for batch_idx, (x, _) in enumerate(t):
        x = x.to(device)
        if batch_lim is not None and batch_idx >= batch_lim:
            break
        # Evaluate batch and backpropagate
        optimizer.zero_grad()
        loss, agg_metrics = eval_batch(x, model, agg_metrics)
        loss.backward()
        optimizer.step()

        # Log metrics
        log_metrics(
            epoch,
            agg_metrics,
            batch_idx=batch_idx,
            train=True,
            do_wandb=do_wandb,
            do_progress_bar=do_progress_bar,
            tq=t,
        )
    
    # After epoch, calculate average metrics
    avg_metrics = {
        'train_loss': agg_metrics['loss'] / agg_metrics['n_samples'],
        'train_loss_recon': agg_metrics['loss_recon'] / agg_metrics['n_samples'],
        'train_loss_kl': agg_metrics['loss_kl'] / agg_metrics['n_samples'],
        'beta': agg_metrics['beta'],
    }

    return avg_metrics


def valid(epoch, model, loader_test, do_wandb=0, do_progress_bar=1, device="cuda"):
    """Validation loop for one epoch."""
    agg_metrics = dict(
        n_samples=0,
        loss=0,
        loss_recon=0,
        loss_kl=0,
        beta=0,
    )

    if do_progress_bar:
        t = tqdm.tqdm(loader_test, desc=f"Valid Epoch {epoch}")
    else:
        t = loader_test

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(t):
            x = x.to(device)
            loss, agg_metrics = eval_batch(x, model, agg_metrics, train=False)
            log_metrics(
                epoch,
                agg_metrics,
                train=False,
                do_wandb=do_wandb,
                do_progress_bar=do_progress_bar,
                tq=t,
            )
    
    # After epoch, calculate average metrics
    avg_metrics = {
        'valid_loss': agg_metrics['loss'] / agg_metrics['n_samples'],
        'valid_loss_recon': agg_metrics['loss_recon'] / agg_metrics['n_samples'],
        'valid_loss_kl': agg_metrics['loss_kl'] / agg_metrics['n_samples'],
        'beta': agg_metrics['beta'],
    }

    return avg_metrics
