import logging
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as T_f
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from models import align_reconstructions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a custom colormap
colors_list = ['#2F0C75', '#0063CA', '#0AA174', '#FFDD00']
custom_cmap = mcolors.LinearSegmentedColormap.from_list('shap_custom_cmap', colors_list)


def reconstruction_aligned(model, x, y, align=True, device=device):
    """
    Align the reconstructed images.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    x : torch.Tensor
        Batch of input data with shape (N, C, H, W).
    y : torch.Tensor
        The reconstructed images.
    align : bool, optional
        Whether to apply alignment (default True).
    device : torch.device, optional
        Device on which to run the computations.

    Returns
    -------
    torch.Tensor
        The aligned reconstructed images.
    """
    assert len(x) <= 256, "Batch size should be at most 256."
    model.eval().to(device)
    if align:
        # Compute optimal rotation and flip transformations
        _, align_transforms = align_reconstructions.loss_reconstruction_fourier_batch(
            x, y, recon_loss_type=model.loss_kwargs["recon_loss_type"]
        )
        # Apply vertical flip where needed
        idxs_flip = np.where(align_transforms[:, 1])[0]
        y[idxs_flip] = T_f.vflip(y[idxs_flip])
        # Rotate images based on optimal angles
        y = align_reconstructions.rotate_batch(y, angles=align_transforms[:, 0])
    return y


def decode_img(model, z, recon_loss_type='ce'):
    """
    Decode an image from a latent representation.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    z : torch.Tensor
        Latent representations with shape (batch_size, embedding_dim).
    recon_loss_type : str, optional
        Reconstruction loss type (default 'ce').

    Returns
    -------
    torch.Tensor
        The decoded image.
    """
    if not isinstance(z, torch.Tensor):
        raise TypeError("z must be a PyTorch tensor")
    if z.dim() != 2:
        raise ValueError(f"z must have shape (batch_size, embedding_dim), but got {z.shape}")
    y = model.p_net(z)
    return y


def decode_embedding(model, embedding):
    """
    Decode an image from an embedding.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    embedding : np.ndarray or torch.Tensor
        The latent embedding vector.

    Returns
    -------
    np.ndarray
        The decoded image moved to the CPU.
    """
    if isinstance(embedding, np.ndarray):
        if embedding.dtype == np.object_:
            embedding = np.asarray(embedding, dtype=np.float32)
        embedding = torch.from_numpy(embedding).float().to(device)
    elif not isinstance(embedding, torch.Tensor):
        raise TypeError("embedding must be a NumPy array or PyTorch tensor")

    model.eval()
    with torch.no_grad():
        decoded = decode_img(model, embedding)
        if getattr(model, 'do_sigmoid', False):
            decoded = torch.sigmoid(decoded)
    return decoded.cpu().numpy()


def align_embed(embed, org_sample, model):
    """
    Align a decoded image from an embedding with an original sample image.

    Parameters
    ----------
    embed : np.ndarray
        The embedding vector.
    org_sample : torch.Tensor or np.ndarray
        The original sample image, assumed to be convertible to shape (1, 1, 128, 128).
    model : torch.nn.Module
        The trained model.

    Returns
    -------
    torch.Tensor
        The aligned decoded image.
    """
    # Prepare the original sample and embedding
    org_sample = org_sample.reshape(1, 1, 128, 128).to(device).cpu()
    embed = embed.reshape(1, -1)
    decoded = decode_embedding(model, embed)
    decoded = torch.tensor(decoded).to(device).cpu()

    _, align_transforms = align_reconstructions.loss_reconstruction_fourier_batch(
        org_sample, decoded, recon_loss_type=model.loss_kwargs["recon_loss_type"]
    )
    idxs_flip = np.where(align_transforms[:, 1])[0]
    decoded[idxs_flip] = T_f.vflip(decoded[idxs_flip])
    decoded = align_reconstructions.rotate_batch(decoded, angles=align_transforms[:, 0])
    return decoded


def colorize_white_pixels(img_tensor, pred_value, cmap=custom_cmap, white_threshold=0.9):
    """
    Replace near-white pixels in an image with a color selected from a colormap.

    Parameters
    ----------
    img_tensor : torch.Tensor or np.ndarray
        A 2D image with pixel values in [0, 1].
    pred_value : float
        A value in [0, 1] used to select the color from the colormap.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use (default is custom_cmap).
    white_threshold : float, optional
        Threshold above which pixels are considered white (default 0.9).

    Returns
    -------
    np.ndarray
        A colorized image with shape (H, W, 3).
    """
    if isinstance(img_tensor, torch.Tensor):
        img_np = img_tensor.detach().cpu().numpy()
    else:
        img_np = img_tensor

    rgba_color = cmap(pred_value)
    rgb_color = rgba_color[:3]
    colorized_img = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=img_np.dtype)
    white_mask = (img_np >= white_threshold)
    colorized_img[white_mask] = rgb_color
    return colorized_img


def process_samples_and_generate_images(new_samples_df, feature_names, cell_cols, nuc_cols,
                                          model2, cells_test, nucs_test):
    """
    Process sample data to generate aligned images based on embeddings and predictions.

    Parameters
    ----------
    new_samples_df : pandas.DataFrame
        DataFrame containing sample data with at least 'sample_name', 'feature_name', and 'preds' columns.
    feature_names : list
        List of feature names.
    cell_cols : list
        List of cell feature column names.
    nuc_cols : list
        List of nucleus feature column names.
    model2 : torch.nn.Module
        The trained model.
    cells_test : array-like
        Array of cell test images.
    nucs_test : array-like
        Array of nucleus test images.

    Returns
    -------
    dict
        Dictionary with keys (sample_name, feature_name) and values as lists of tuples 
        (feature_value, pred_value, image).
    """
    model2.to(device)
    logger.info("Model moved to GPU.")
    images_dict = defaultdict(list)
    grouped = new_samples_df.groupby(['sample_name', 'feature_name'])
    logger.info("Grouped data by sample_name and feature_name.")

    for (sample_name, feature_name), group in grouped:
        logger.info(f"Processing sample {sample_name} with feature {feature_name}.")

        # Determine feature type and select the appropriate original image.
        if feature_name in cell_cols:
            feature_type = 'cell'
            org_img = cells_test[int(sample_name)]
        elif feature_name in nuc_cols:
            feature_type = 'nuc'
            org_img = nucs_test[int(sample_name)]
        else:
            logger.warning(f"Feature {feature_name} not in cell_cols or nuc_cols. Skipping.")
            continue

        embeddings = group[cell_cols].values if feature_type == 'cell' else group[nuc_cols].values

        for pos, (_, row) in enumerate(group.iterrows()):
            feature_value = row[feature_name]
            pred_value = row['preds']
            embedding = embeddings[pos]

            img = align_embed(embedding, org_img, model2)
            img = img.squeeze()  # Convert to (H, W)

            # Optionally, colorize based on pred_value:
            # colorized_img = colorize_white_pixels(img, pred_value, cmap=custom_cmap)
            colorized_img = img

            images_dict[(sample_name, feature_name)].append((feature_value, pred_value, colorized_img))
            logger.info(f"Generated image for sample {sample_name} and feature {feature_name}.")

    return images_dict


def generate_new_samples(X_test, feature_names, top_features, other_features, cell_cols, nuc_cols,pipeline, n=8, k=10):
    """
    Generate new samples by varying specified top features in the test set.

    For each sample in X_test and for each top feature (excluding those in other_features),
    the function creates k new samples by varying the feature value across an extended range.

    Parameters
    ----------
    X_test : np.ndarray
        Test data with shape (num_samples, num_features).
    feature_names : list
        List of feature names corresponding to the columns of X_test.
    top_features : list
        List of top feature names to vary.
    other_features : list
        List of feature names to skip.
    cell_cols : list
        List of cell feature names.
    nuc_cols : list
        List of nucleus feature names.
    n : int, optional
        Number of top features to vary (default 8).
    k : int, optional
        Number of steps to vary each feature (default 10).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the new samples with additional columns:
        'sample_name', 'feature_name', and 'feature_type'.
    """
    new_samples = []
    num_samples = X_test.shape[0]

    # Iterate over each sample
    for i in range(num_samples):
        sample = X_test[i, :]
        # Iterate over the top n features
        for feature in top_features[:n]:
            if feature in other_features:
                continue  # Skip features in other_features

            # Determine feature type
            if feature in cell_cols:
                feature_type = 'cell'
            elif feature in nuc_cols:
                feature_type = 'nuc'
            else:
                continue  # Skip if not found

            feature_idx = feature_names.index(feature)
            feature_min = X_test[:, feature_idx].min()
            feature_max = X_test[:, feature_idx].max()
            diff = feature_max - feature_min

            # Vary the feature value across an extended range
            for value in np.linspace(feature_min - 0.5 * diff, feature_max + 0.5 * diff, k):
                new_sample = sample.copy()
                new_sample[feature_idx] = value
                # Append additional info: sample index, feature name, and feature type
                new_samples.append(np.append(new_sample, [i, feature, feature_type]))

    # Create a DataFrame from the generated samples
    columns = feature_names + ['sample_name', 'feature_name', 'feature_type']
    new_samples_df = pd.DataFrame(new_samples, columns=columns)
    new_x = np.array(new_samples_df[feature_names])
    preds = pipeline.predict_proba(new_x)[:, 1]
    new_samples_df['preds'] = preds
    return new_samples_df




def show_figs(images_dict, cells_test, nucs_test):
    """
    Display figures for each key in the images dictionary.

    For each (sample_name, feature_name) key in the dictionary (filtered here to only include
    feature_name == 'embed_nuc_11'), the function retrieves the original image from either
    cells_test or nucs_test and plots it alongside the step images with their prediction scores.

    Parameters
    ----------
    images_dict : dict
        Dictionary with keys as (sample_name, feature_name) and values as lists of tuples
        (feature_value, pred_value, image).
    cells_test : np.ndarray
        Array of cell test images.
    nucs_test : np.ndarray
        Array of nucleus test images.
    """
    for key_ in images_dict.keys():
        sample_name, feature_name = key_
        # Filter to display only the desired feature, adjust condition as needed
        if feature_name == 'embed_nuc_11':
            print(key_)
            print('sample_name, feature_name:', sample_name, feature_name)
            # Select the original image based on feature type
            if 'cel' in feature_name:
                org_img = cells_test[:, 0, :, :][int(sample_name)]
            else:
                org_img = nucs_test[:, 0, :, :][int(sample_name)]

            step_images = []
            preds = []
            for step in images_dict[key_]:
                _, pred, img = step
                step_images.append(img)
                preds.append(pred)

            # Create a figure with 6 subplots (1 original + 5 step images)
            fig, axes = plt.subplots(1, 6, figsize=(20, 4))
            # Display the original image
            axes[0].imshow(org_img)
            axes[0].set_title('Original')
            axes[0].axis('off')

            # Display each step image
            for idx, img in enumerate(step_images):
                axes[idx + 1].imshow(img)
                axes[idx + 1].set_title(str(round(preds[idx], 2)))
                axes[idx + 1].axis('off')

            plt.tight_layout()
            plt.show()



