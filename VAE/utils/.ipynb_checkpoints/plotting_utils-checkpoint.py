import torch
import numpy as np
import matplotlib.pyplot as plt
import glasbey
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
import torchvision.transforms.functional as T_f
from models import align_reconstructions
import umap
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgb

def plot_embedding_space_w_labels(X, y, figsize=(9,9),
            scatter_kwargs=dict(s=0.1,legend_fontsize=10, legend_marker_size=100, hide_labels=True),
            colormap="glasbey",
):
    """
    Make a 2d scatter plot of an embedding space (e.g. umap) colored by labels.
    """
    assert X.shape[1]==2
    f,axs = plt.subplots(figsize=(9,9))

    if colormap=="glasbey":
        colors = glasbey.create_palette(palette_size=10)
    else:
        colors = sns.color_palette("tab10")
    y_uniq = np.unique(y)
    for i, label in enumerate(y_uniq):
        idxs = np.where(y==label)[0]
        axs.scatter(X[idxs,0], X[idxs,1],
                    color=colors[i], s=scatter_kwargs['s'], label=i)

    legend=plt.legend(fontsize=scatter_kwargs['legend_fontsize'])
    [legend.legendHandles[i].set_sizes([scatter_kwargs['legend_marker_size']], dpi=300)
             for i in range(len(legend.legendHandles))]

    if scatter_kwargs['hide_labels']:
        axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.close()

    return f, axs

def get_embedding_space_embedded_images(embedding, data, n_yimgs=70, n_ximgs=70,
            xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Given an embedding (e.g. umap or tsne embedding) and their original images, 
    generate an image (that can be passed to plt.imshow) that samples images from
    the space. This is similar to the tensorflow embedding projector.

    How it works: break space into grid. Each image is assigned to a rectangles if it's
    enclosed by that rectangle. The image nearest the rectangle centroid (L2
    distance) is assigned to it, and plotted.

    Args
      embedding: Array shape (n_imgs, 2,) holding the tsne embedding coordinates
        of the imgs stored in data
      data: original data set of images. len(data)==len(embedding). To be plotted
        on the TSNE grid.
    Returns
        img_plot (Tensor): the tensor to pass to `plt.imshow()`. 
        object_indices: the indices of the imgs in `data` corresponding to the 
            grid points.
    """
    assert len(data)==len(embedding)
    img_shape = data.shape[-2:]
    ylen, xlen = data.shape[-2:]
    if xmin is None: xmin=embedding[:,0].min()
    if xmax is None: xmax=embedding[:,0].max()
    if ymin is None: ymin=embedding[:,1].min()
    if ymax is None: ymax=embedding[:,1].max()

    # Define grid corners
    ycorners, ysep = np.linspace(ymin, ymax, n_yimgs, endpoint=False, retstep=True)
    xcorners, xsep = np.linspace(xmin, xmax, n_ximgs, endpoint=False, retstep=True)
    # centroids of the grid
    ycentroids=ycorners+ysep/2
    xcentroids=xcorners+xsep/2

    # determine which point in the grid each embedded point belongs
    img_grid_indxs = (embedding - np.array([xmin, ymin])) // np.array([xsep,ysep])
    img_grid_indxs = img_grid_indxs.astype(dtype=int)

    #  Array that will hold each points distance to the centroid
    img_dist_to_centroids = np.zeros(len(embedding))

    # array to hold the final set of images
    img_plot=torch.zeros(n_yimgs*img_shape[0], n_ximgs*img_shape[1])

    # array that will give us the returnedindices
    object_indices=torch.zeros((n_ximgs, n_yimgs), dtype=torch.int)

    # Iterate over the grid
    for i in range(n_ximgs):
        for j in range(n_yimgs):
            ## Get indices of points that are in this box
            indxs=indxs = np.where(
                    np.all(img_grid_indxs==np.array([i,j])
                    ,axis=1)
                )[0]

            ## calculate distance to centroid for each point
            centroid=np.array([xcentroids[i],ycentroids[j]])
            img_dist_to_centroids[indxs] = np.linalg.norm(embedding[indxs] - centroid, ord=2, axis=1)

            ## Find the nearest image to the centroid
            # if there are no imgs in this box, then skip
            if len(img_dist_to_centroids[indxs])==0:
                indx_nearest=-1
            # else find nearest
            else:
                # argmin over the distances to centroid (is over a restricted subset)
                indx_subset = np.argmin(img_dist_to_centroids[indxs])
                indx_nearest = indxs[indx_subset]
                # Put image in the right spot in the larger image
                xslc = slice(i*xlen, i*xlen+xlen)
                yslc = slice(j*ylen, j*ylen+ylen)
                img_plot[xslc, yslc] = torch.Tensor(data[int(indx_nearest)])

            # save the index
            object_indices[i,j] = indx_nearest

    # turns out the x and y coordiates got mixed up so I have to transpose it here
    # and also I need to flip the image
    img_plot = torch.transpose(img_plot, 1,0)
    img_plot = torch.flip(img_plot,dims=[0])

    return img_plot, object_indices


def plot_embedding_space_w_labels_with_nan(X, y, figsize=(9, 9),
                                       scatter_kwargs=dict(s=4, legend_fontsize=10, legend_marker_size=100, hide_labels=True),
                                       colormap="glasbey"):
    """
    Make a 2d scatter plot of an embedding space (e.g. umap) colored by labels.
    """
    assert X.shape[1] == 2
    f, axs = plt.subplots(figsize=figsize)

    if colormap == "glasbey":
        colors = glasbey.create_palette(palette_size=10)
    else:
        colors = sns.color_palette("tab10")

    # Convert list to np.array if needed and handle NaN values
    y = np.array(y, dtype=object)
    y[pd.isna(y)] = 'nan'
    y_uniq = np.unique(y)

    for i, label in enumerate(y_uniq):
        idxs = np.where(y == label)[0]
        color = 'black' if label == 'nan' else colors[i % len(colors)]  # Ensure NaN values are always black
        scatter_kwargs['s'] =  1 if label == 'nan' else 8  # Ensure NaN values are always black
        axs.scatter(X[idxs, 0], X[idxs, 1], color=color, s=scatter_kwargs['s'], label=label)

    legend = axs.legend(fontsize=scatter_kwargs['legend_fontsize'])
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i].set_sizes([scatter_kwargs['legend_marker_size']])

    if scatter_kwargs['hide_labels']:
        axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.close()

    return f, axs



def compute_and_plot_metrics(model, cells_np, nucs_np, device, batch_size=100):

    images_cell3 = cells_np[:, 0, :, :]
    images_nuc3 = nucs_np[:, 0, :, :]

    model.eval().to(device)

    def reconstruct_images(images):
        images_rec = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                print(f"Processing batch starting at index {i}")
                batch_end = min(i + batch_size, len(images))
                x = images[i:batch_end].reshape(-1, 1, 128, 128)
                y = model.reconstruct(x.to(device)).cpu()
                _, align_transforms = align_reconstructions.loss_reconstruction_fourier_batch(
                    x, y, recon_loss_type=model.loss_kwargs["recon_loss_type"]
                )
                idxs_flip = np.where(align_transforms[:, 1])[0]
                y[idxs_flip] = T_f.vflip(y[idxs_flip])
                y = align_reconstructions.rotate_batch(y, angles=align_transforms[:, 0])
                images_rec.append(np.array(y))

        images_rec = np.concatenate(images_rec, axis=0)[:, 0, :, :]
        return images_rec

    images_rec_cell3 = reconstruct_images(images_cell3)
    images_rec_nuc3 = reconstruct_images(images_nuc3)

    def compute_reconstruction_metrics(images_rec, images_orig):
        images_rec_tensor = torch.from_numpy(images_rec).float().cpu()
        images_orig_tensor = (
            images_orig.float().cpu() if torch.is_tensor(images_orig)
            else torch.from_numpy(images_orig).float().cpu()
        )

        mse = ((images_rec_tensor - images_orig_tensor)**2).view(images_rec_tensor.size(0), -1).mean(dim=1)
        bce = torch.nn.BCELoss(reduction='none')(images_rec_tensor, images_orig_tensor).view(images_rec_tensor.size(0), -1).mean(dim=1)

        return mse, bce

    mse_values_cell, bce_values_cell = compute_reconstruction_metrics(images_rec_cell3, images_cell3)
    mse_values_nuc, bce_values_nuc = compute_reconstruction_metrics(images_rec_nuc3, images_nuc3)

    df_plot = pd.DataFrame({
        "value":  np.concatenate([mse_values_cell, mse_values_nuc, bce_values_cell, bce_values_nuc]),
        "metric": (["MSE"] * len(mse_values_cell) + ["MSE"] * len(mse_values_nuc) +
                   ["BCE"] * len(bce_values_cell) + ["BCE"] * len(bce_values_nuc)),
        "type":   (["Cell"] * len(mse_values_cell) + ["Nucleus"] * len(mse_values_nuc) +
                   ["Cell"] * len(bce_values_cell) + ["Nucleus"] * len(bce_values_nuc))
    })

    # Compute and print mean ± std for each group
    group_stats = df_plot.groupby(['metric', 'type'])['value'].agg(['mean', 'std']).reset_index()
    print("Reconstruction metrics (mean ± std):")
    for _, row in group_stats.iterrows():
        print(f"{row['metric']} - {row['type']}: {row['mean']:.4f} ± {row['std']:.4f}")

    sns.set(style="white")

    palette = {"Cell": "#00ADDC", "Nucleus": "#F15A22"}

    fig, ax_mse = plt.subplots(figsize=(6, 6))
    ax_bce = ax_mse.twinx()

    hue_order = ["Cell", "Nucleus"]

    sns.boxplot(x="metric", y="value", hue="type", data=df_plot[df_plot["metric"] == "MSE"],
                palette=palette, showfliers=False, ax=ax_mse, width=0.6, linewidth=1.5,
                boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
                whiskerprops=dict(color="black"), capprops=dict(color="black"),
                order=["MSE", "BCE"], hue_order=hue_order)

    sns.stripplot(x="metric", y="value", hue="type", data=df_plot[df_plot["metric"] == "MSE"],
                  color="black", alpha=0.6, size=2, jitter=True, dodge=True, ax=ax_mse,
                  order=["MSE", "BCE"], hue_order=hue_order)

    if ax_mse.legend_:
        ax_mse.legend_.remove()

    sns.boxplot(x="metric", y="value", hue="type", data=df_plot[df_plot["metric"] == "BCE"],
                palette=palette, showfliers=False, ax=ax_bce, width=0.6, linewidth=1.5,
                boxprops=dict(edgecolor="black"), medianprops=dict(color="black"),
                whiskerprops=dict(color="black"), capprops=dict(color="black"),
                order=["MSE", "BCE"], hue_order=hue_order)

    sns.stripplot(x="metric", y="value", hue="type", data=df_plot[df_plot["metric"] == "BCE"],
                  color="black", alpha=0.6, size=2, jitter=True, dodge=True, ax=ax_bce,
                  order=["MSE", "BCE"], hue_order=hue_order)

    if ax_bce.legend_:
        ax_bce.legend_.remove()

    ax_mse.set_ylabel("MSE")
    ax_bce.set_ylabel("BCE")

    handles = [patches.Patch(color=palette[key], label=key) for key in hue_order]
    ax_mse.legend(handles=handles, loc='upper right', title='Type')

    plt.tight_layout()
    plt.savefig("mse_bce_separate_y_axes.svg", format="svg", dpi=300)
    plt.show()


def plot_umap_embed(images_,embeds_,y):
    cmap = {False: '#00ADDC', True: '#F15A22'}
    def hex_to_rgb(hex_color):
        """Convert hex color to normalized RGB."""
        return np.array(to_rgb(hex_color))
    color_map_rgb = {label: hex_to_rgb(color) for label, color in cmap.items()}
    if images_.max() > 1.0:
        images_ = images_ / 255.0
    data_x_rgb = torch.zeros((images_.shape[0], 3, images_.shape[1], images_.shape[2]))
    for i in range(len(images_)):
        img = images_[i,  :, :]  # Grayscale image
        label = y[i]
        color_rgb = color_map_rgb[bool(label)]  # Convert label to boolean
    
        # Convert color_rgb to a PyTorch tensor
        color_rgb_tensor = torch.tensor(color_rgb, dtype=torch.float32)
    
        # Apply the color overlay
        data_x_rgb[i, 0, :, :] = torch.tensor(img) * color_rgb_tensor[0]  # Red channel
        data_x_rgb[i, 1, :, :] = torch.tensor(img) * color_rgb_tensor[1]  # Green channel
        data_x_rgb[i, 2, :, :] = torch.tensor(img) * color_rgb_tensor[2]  # Blue channel
    
    data_x_rgb = torch.zeros((images_.shape[0], 3, images_.shape[1], images_.shape[2]))
    # Convert grayscale images to RGB with color overlays
    # Convert grayscale images to RGB with color overlays
    for i in range(len(images_)):
        img = images_[i, :, :]  # Grayscale image
        label = y[i]
        color_rgb = color_map_rgb[bool(label)]  # Convert label to boolean
    
        # Convert color_rgb to a PyTorch tensor
        color_rgb_tensor = torch.tensor(color_rgb, dtype=torch.float32)
    
        # Apply the color overlay
        data_x_rgb[i, 0, :, :] = torch.tensor(img) * color_rgb_tensor[0]  # Red channel
        data_x_rgb[i, 1, :, :] = torch.tensor(img) * color_rgb_tensor[1]  # Green channel
        data_x_rgb[i, 2, :, :] = torch.tensor(img) * color_rgb_tensor[2]  # Blue channel
    
    mask = (data_x_rgb.sum(dim=1) == 0)  # Shape: (N, H, W)
    mask_expanded = mask.unsqueeze(1).expand_as(data_x_rgb)
    data_x_rgb[mask_expanded] = 1.0  # Set black pixels to white
    def get_embedding_space_embedded_images(embedding, data, n_yimgs=30, n_ximgs=30,
                                            xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Generate an image grid from embeddings with images colored based on labels.
        """
        assert len(data) == len(embedding), "Data and embedding must have the same length."
        img_shape = data.shape[-2:]  # (H, W)
        channels = data.shape[1]     # Number of channels (should be 3 now)
        ylen, xlen = img_shape
    
        # Define the embedding space boundaries
        if xmin is None:
            xmin = embedding[:, 0].min()
        if xmax is None:
            xmax = embedding[:, 0].max()
        if ymin is None:
            ymin = embedding[:, 1].min()
        if ymax is None:
            ymax = embedding[:, 1].max()
    
        # Define grid corners
        ycorners, ysep = np.linspace(ymin, ymax, n_yimgs, endpoint=False, retstep=True)
        xcorners, xsep = np.linspace(xmin, xmax, n_ximgs, endpoint=False, retstep=True)
        ycentroids = ycorners + ysep / 2
        xcentroids = xcorners + xsep / 2
    
        # Determine grid indices for each embedding point
        img_grid_indxs = (embedding - np.array([xmin, ymin])) // np.array([xsep, ysep])
        img_grid_indxs = img_grid_indxs.astype(dtype=int)
    
        # Initialize the grid with ones for a white background
        img_plot = np.ones((n_yimgs * ylen, n_ximgs * xlen, channels))
    
        # Track indices of images placed in the grid
        object_indices = np.ones((n_ximgs, n_yimgs), dtype=int) * -1
    
        # Iterate over the grid
        for i in range(n_ximgs):
            for j in range(n_yimgs):
                indxs = np.where(np.all(img_grid_indxs == np.array([i, j]), axis=1))[0]
                if len(indxs) > 0:
                    centroid = np.array([xcentroids[i], ycentroids[j]])
                    distances = np.linalg.norm(embedding[indxs] - centroid, axis=1)
                    indx_nearest = indxs[np.argmin(distances)]
    
                    yslc = slice(j * ylen, (j + 1) * ylen)
                    xslc = slice(i * xlen, (i + 1) * xlen)
                    img = data[int(indx_nearest)].permute(1, 2, 0).numpy()
                    img_plot[yslc, xslc, :] = img
                    object_indices[i, j] = indx_nearest
                else:
                    object_indices[i, j] = -1
    
        img_plot = np.flipud(img_plot)
        return img_plot, object_indices
    
    # Step 4: Generate and Plot the Grid
    grid, idxs = get_embedding_space_embedded_images(
        embeds_, data_x_rgb, n_ximgs=30, n_yimgs=30
    )
    
    # Plot the grid
    f, axs = plt.subplots(figsize=(20, 20))
    axs.set_axis_off()
    axs.imshow(grid)
    plt.show()

def plot_umap_scatter(embeddings_cells, embeddings_nucs, y, 
                      n_components=2, random_state=42, 
                      min_dist=0.5, spread=20.2, 
                      figsize=(8, 4), dpi=300):
    """
    Generates UMAP embeddings and side-by-side scatter plots for cell and nuc embeddings.
    
    Parameters:
    - embeddings_cells: array-like, cell embeddings of shape (n_cells, n_features)
    - embeddings_nucs: array-like, nuc embeddings of shape (n_nucs, n_features)
    - y: array-like, color labels for the scatter plots (e.g., binary or continuous data)
    - n_components: int, number of UMAP dimensions (default 2)
    - random_state: int, random seed for UMAP (default 42)
    - min_dist: float, UMAP min_dist parameter (default 0.5)
    - spread: float, UMAP spread parameter (default 20.2)
    - figsize: tuple, size of the figure (default (8, 4))
    - dpi: int, dots per inch for the figure (default 300)
    
    Returns:
    - fig: matplotlib Figure object
    - axes: array of Axes objects for further customization if needed
    """
    # Convert embeddings to numpy arrays.
    all_cells_embeds = np.array(embeddings_cells)
    all_nucs_embeds = np.array(embeddings_nucs)
    all_cells_nucs_embeds = np.concatenate([all_cells_embeds, all_nucs_embeds])
    print("Shapes:", all_cells_embeds.shape, all_cells_nucs_embeds.shape)
    
    # Standard scaling on the combined embeddings.
    scaler = StandardScaler()
    embeddings_combined_scaled = scaler.fit_transform(all_cells_nucs_embeds)
    
    # Initialize and fit UMAP on the combined (unscaled) embeddings.
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, 
                        min_dist=min_dist, spread=spread)
    reducer.fit(all_cells_nucs_embeds)
    
    # Define a helper function to apply scaling and UMAP reduction.
    def umap_apply(data, scaler=scaler, reducer=reducer):
        return reducer.transform(scaler.transform(data))
    
    # Create a vivid red & blue colormap for binary data.
    binary_cmap = ListedColormap(["#00ADDC", "#F15A22"])
    
    # Set up plot style.
    plt.style.use('default')
    plt.rcParams['axes.grid'] = False
    
    # Get UMAP embeddings for cells and nucs.
    cell_embeds_umap = umap_apply(all_cells_embeds)
    nuc_embeds_umap  = umap_apply(all_nucs_embeds)
    
    # Determine axis limits based on the data.
    x_min = min(cell_embeds_umap [:, 0].min(), nuc_embeds_umap [:, 0].min())
    x_max = max(cell_embeds_umap [:, 0].max(), nuc_embeds_umap [:, 0].max())
    y_min = min(cell_embeds_umap [:, 1].min(), nuc_embeds_umap [:, 1].min())
    y_max = max(cell_embeds_umap [:, 1].max(), nuc_embeds_umap [:, 1].max())
    x_lim = [x_min - 0.1 * abs(x_min), x_max + 0.1 * abs(x_max)]
    y_lim = [y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max)]
    
    # Create subplots.
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Helper function to remove extra spines and create a minimalist axis.
    def minimalist_ax(ax):
        ax.tick_params(axis='both', which='both', length=3)
    
    # Plot for 'Cell'.
    axes[0].scatter(
        cell_embeds_umap [:, 0],
        cell_embeds_umap [:, 1],
        c=y,
        cmap=binary_cmap,
        alpha=0.8,
        s=10,
        edgecolor=None
    )
    axes[0].set_title('Cell', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Embedding Dimension 1', fontsize=10)
    axes[0].set_ylabel('Embedding Dimension 2', fontsize=10)
    axes[0].set_xlim(x_lim)
    axes[0].set_ylim(y_lim)
    minimalist_ax(axes[0])
    
    # Plot for 'Nuc'.
    axes[1].scatter(
        nuc_embeds_umap [:, 0],
        nuc_embeds_umap [:, 1],
        c=y,
        cmap=binary_cmap,
        alpha=0.8,
        s=10,
        edgecolor=None
    )
    axes[1].set_title('Nuc', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Embedding Dimension 1', fontsize=10)
    axes[1].set_ylabel('Embedding Dimension 2', fontsize=10)
    axes[1].set_xlim(x_lim)
    axes[1].set_ylim(y_lim)
    minimalist_ax(axes[1])
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes, cell_embeds_umap ,nuc_embeds_umap 
