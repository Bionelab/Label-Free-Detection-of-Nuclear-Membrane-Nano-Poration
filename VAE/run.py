import logging
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from data.datasets import TensorDatasetWithTransform
from models.decoders.cnn_decoder import CnnDecoder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/encoders_o2/')))
from e2scnn import E2SFCNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/encoders_vanilla/')))
from models.encoders_vanilla.cnn_encoder import CnnEncoder
from models.vae import VAE
import random
import numpy as np
import sys
sys.path.append("VAE/")
import importlib
import train_loops
import wandb
from pathlib import Path
from utils import utils,eval_utils
import run
from configs.config_cytomodel import config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
importlib.reload(utils)
from models.vae import VAE
import re
import logging
# Function to save checkpoints


warnings.filterwarnings(
    "ignore",
    message=".*aten/src/ATen/native*",
)  # filter 2 specific warnings from e2cnn library

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_datasets_from_config(config, file_name= "new_images_2time_128.npy", train_size_x = 0.8):
    print(config.data.data_dir)
    assert os.path.isdir(config.data.data_dir), f"config.data.data_dir does not exist"
    
    # Load the .npy file
    data_x = np.load(os.path.join(config.data.data_dir, file_name))
    
    # Convert the data to a PyTorch tensor
    data_x = torch.Tensor(data_x)
    
    # Reshape to [42125, 1, 128, 128]
    data_x = data_x.permute(0, 3, 1, 2)
    
    # Create a dummy y data
    data_y = torch.arange(len(data_x))

    assert len(data_x) == len(data_y)

    # Split the data into training and testing sets
    dataset = TensorDataset(data_x, data_y)
    train_size = int(train_size_x * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Apply the transforms to the datasets
    train_dataset = TensorDatasetWithTransform(
        train_dataset[:][0], train_dataset[:][1], transform=config.data.transform_train
    )
    test_dataset = TensorDatasetWithTransform(
        test_dataset[:][0], test_dataset[:][1], transform=config.data.transform_test
    )

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=config.data.shuffle_data_loader,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=False,
    )

    return train_dataset, train_loader, test_dataset, test_loader




def get_datasets_from_config_all(config, file_name= "new_images_2time_128.npy"):
    assert os.path.isdir(config.data.data_dir), f"config.data.data_dir does not exist"
    
    # Load the .npy file
    data_x = np.load(os.path.join(config.data.data_dir, file_name))
    
    # Convert the data to a PyTorch tensor
    data_x = torch.Tensor(data_x)
    
    # Reshape to [42125, 1, 128, 128]
    data_x = data_x.permute(0, 3, 1, 2)
    
    # Create a dummy y data
    data_y = torch.arange(len(data_x))

    assert len(data_x) == len(data_y)

    # Split the data into training and testing sets
    dataset = TensorDataset(data_x, data_y)
    
    # Apply the transforms to the datasets
    train_dataset = TensorDatasetWithTransform(
        dataset[:][0], dataset[:][1], transform=config.data.transform_train)

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        shuffle=config.data.shuffle_data_loader,
    )

    return train_dataset, train_loader

    
def build_model_from_config(config):
    if config.model.vanilla:
        logging.warning("Using vanila (not O2-invariant) VAE")
        config.model.encoder.name = "cnn_encoder"
        config.loss.align_loss = False

    # class lookups for encoder, decoder, and vae
    lookup_model = dict(
        vae=VAE, o2_cnn_encoder=E2SFCNN, cnn_encoder=CnnEncoder, cnn_decoder=CnnDecoder
    )

    # encoder
    config.model.encoder.n_classes = (
        config.model.zdim * 2
    )  # bc vae saves mean and stdDev vecors
    q_net_class = lookup_model[config.model.encoder.name]

    q_net = q_net_class(**config.model.encoder)

    # decoder
    p_net_class = lookup_model[config.model.decoder.name]
    config.model.decoder.zdim = config.model.zdim
    config.model.decoder.out_channels = config.model.encoder.n_channels
    p_net = p_net_class(**config.model.decoder)

    # vae
    model_class = lookup_model[config.model.name]
    model_kwargs = config.model
    model_kwargs.p_net = p_net
    model_kwargs.q_net = q_net
    model_kwargs.loss_kwargs = config.loss
    model_class = lookup_model[config.model.name]
    model = model_class(**model_kwargs)

    return model


def save_checkpoint(state, filename):
    temp_filename = filename + temp_suffix
    try:
        torch.save(state, temp_filename)
        os.rename(temp_filename, filename)
        logger.info(f"Checkpoint saved atomically to {filename}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint {filename}: {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            logger.info(f"Temporary checkpoint {temp_filename} removed.")


def find_latest_checkpoint(checkpoint_dir, base_fname_model, interrupted_suffix):
    """
    Finds the latest valid checkpoint file in the checkpoint directory.
    Prioritizes complete checkpoints over interrupted ones.

    Args:
        checkpoint_dir (str): Directory where checkpoints are saved.
        base_fname_model (str): Base filename pattern for checkpoints, e.g., '128_lr0.005_epoch_{}.pth'.
        interrupted_suffix (str): Suffix for interrupted checkpoints, e.g., '_interrupted.pth'.

    Returns:
        tuple:
            - str or None: Path to the latest valid checkpoint file or None if no valid checkpoint exists.
            - int or None: The epoch number of the latest checkpoint or None.
    """
    checkpoint_files = os.listdir(checkpoint_dir)

    # Build regex patterns for complete and interrupted checkpoints
    # Escape special characters in base_fname_model
    base_fname_pattern = re.escape(base_fname_model).replace('\\{\\}', '(\\d+)')
    regex_complete = re.compile('^' + base_fname_pattern + '$')

    # Build pattern for interrupted checkpoints
    interrupted_fname_model = base_fname_model[:-4] + '_interrupted.pth'  # Remove '.pth' and add '_interrupted.pth'
    interrupted_fname_pattern = re.escape(interrupted_fname_model).replace('\\{\\}', '(\\d+)')
    regex_interrupted = re.compile('^' + interrupted_fname_pattern + '$')

    latest_epoch = -1
    latest_checkpoint = None

    # First, search for complete checkpoints
    for fname in checkpoint_files:
        match = regex_complete.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = fname

    # If no complete checkpoints found, search for interrupted ones
    if latest_checkpoint is None:
        for fname in checkpoint_files:
            match = regex_interrupted.match(fname)
            if match:
                epoch = int(match.group(1))
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_checkpoint = fname

    if latest_checkpoint:
        return os.path.join(checkpoint_dir, latest_checkpoint), latest_epoch
    else:
        return None, None



def start_training( loader, loader_test ,
                   do_train = False,
                   num_epochs = 100,
                   check_points = "VAE/model_training_checkpoints"
                   
                  ):

    config.model.zdim = 32
    config.model.encoder.n_channels=1
    model = run.build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    decoder = model.p_net
    print(config.model.zdim)
    print(config.model.encoder.cnn_dims)
    print(config.model.decoder.cnn_dims)
    print(model.model_details())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    checkpoint_dir = os.path.join(os.getcwd(), "VAE/model_training_checkpoints")
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_fname_model = 'lr0.005_epoch_{}.pth'
    interrupted_suffix = '_interrupted.pth'
    temp_suffix = '.tmp'  # Suffix for temporary files during saving
    
    
    # Now use this function to find the latest checkpoint
    latest_checkpoint_path, latest_epoch = find_latest_checkpoint(
        checkpoint_dir, base_fname_model, interrupted_suffix
    )
    print(latest_checkpoint_path)
    # Load the checkpoint if it exists
    if latest_checkpoint_path:
        logger.info(f"Found checkpoint at {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only = True)
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Set the start epoch
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {checkpoint['epoch']})")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint_path}: {e}")
            logger.info("Starting training from epoch 1.")
            start_epoch = 1
    else:
        # No checkpoint found, start from scratch
        start_epoch = 1
        logger.info("No checkpoint found, starting training from scratch.")
    
    
            # Now use this function to find the latest checkpoint
    latest_checkpoint_path, latest_epoch = find_latest_checkpoint(
        checkpoint_dir, base_fname_model, interrupted_suffix
    )
    print(latest_checkpoint_path)
    # Load the checkpoint if it exists
    if latest_checkpoint_path:
        logger.info(f"Found checkpoint at {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only = True)
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Set the start epoch
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {checkpoint['epoch']})")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint_path}: {e}")
            logger.info("Starting training from epoch 1.")
            start_epoch = 1
    else:
        # No checkpoint found, start from scratch
        start_epoch = 1
        logger.info("No checkpoint found, starting training from scratch.")
    
    # # # Now you can proceed with your training loop
    num_epochs = num_epochs  # Number of epochs you want to train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('done')
    loss_trains = []
    loss_vals = []
    if do_train:
        try:   
            for epoch in range(start_epoch, start_epoch + num_epochs):
                # Training phase
                loss_t = train_loops.train(
                    epoch,
                    model,
                    loader,
                    optimizer,
                    do_progress_bar=config.logging.do_progress_bar,
                    do_wandb=0,
                    device=device
                )
                loss_trains.append(loss_t)
                # Validation phase
                if config.run.do_validation and epoch % config.run.valid_freq == 0:
                    loss_v =train_loops.valid(
                        epoch,
                        model,
                        loader_test,
                        do_progress_bar=config.logging.do_progress_bar,
                        do_wandb=0,
                        device=device
                    )
                    loss_vals.append(loss_v)
                # Define the filename for the current epoch
                fname_model = base_fname_model.format(epoch)
                full_path = os.path.join(checkpoint_dir, fname_model)
        
                # Save the model checkpoint atomically
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Include if using a scheduler
                    # 'loss': loss,  # Include if tracking loss
                }
                save_checkpoint(checkpoint_state, full_path)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        
            # Optionally, save the model at the point of interruption
            if 'epoch' in locals():
                fname_model_interrupt = f'128_lr0.005_epoch_{epoch}_interrupted.pth'
                full_path_interrupt = os.path.join(checkpoint_dir, fname_model_interrupt)
                checkpoint_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),  # Include if using a scheduler
                    # 'loss': loss,  # Include if tracking loss
                }
                # Save interrupted checkpoint atomically
                save_checkpoint(checkpoint_state, full_path_interrupt)
                logger.info(f"Interrupted checkpoint saved to {full_path_interrupt}.")
    else:
        # Now use this function to find the latest checkpoint
        latest_checkpoint_path, latest_epoch = find_latest_checkpoint(
            checkpoint_dir, base_fname_model, interrupted_suffix
        )
        print(latest_checkpoint_path)
        # Load the checkpoint if it exists
        if latest_checkpoint_path:
            logger.info(f"Found checkpoint at {latest_checkpoint_path}")
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only = True)
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Set the start epoch
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Loaded checkpoint '{latest_checkpoint_path}' (epoch {checkpoint['epoch']})")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {latest_checkpoint_path}: {e}")
                logger.info("Starting training from epoch 1.")
                start_epoch = 1
        else:
            # No checkpoint found, start from scratch
            start_epoch = 1
            logger.info("No checkpoint found, starting training from scratch.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print('done')
    
    return model
                    
                
            
            
