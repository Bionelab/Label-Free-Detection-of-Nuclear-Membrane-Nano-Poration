# plot_utils.py
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import sys
sys.path.append("PREP/")

###
# Paths to your data
polygons_path = 'PREP/polys' # naming : sample_40X_NP_C2_002_Merge.json
images_path_rgb_only = 'PREP/all_images_rgb' #sample_40X_NP_C2_002_RGB.tif
images_path_org='PREP/all_images'
###
PIXEL_TO_MICROMETER = 0.091377  # µm/pixel
AREA_CONVERSION = PIXEL_TO_MICROMETER ** 2  # µm²/pixel²

features= ['cell_area', 'cell_perimeter', 'cell_circularity', 'cell_longest_axis',
       'cell_shortest_axis', 'cell_axis_ratio', 'cell_ellipse_orientation',
       'cell_eccentricity', 'nuc_area', 'nuc_perimeter', 'nuc_circularity',
       'nuc_longest_axis', 'nuc_shortest_axis', 'nuc_axis_ratio',
       'nuc_ellipse_orientation', 'nuc_eccentricity',
       'mutual_centroid_distance', 'mutual_extent']

length_columns = [
    'cell_perimeter',
    'cell_longest_axis',
    'cell_shortest_axis',
    'nuc_perimeter',
    'nuc_longest_axis',
    'nuc_shortest_axis',
    'mutual_extent',
    'mutual_centroid_distance',
]

area_columns = [
    'cell_area',
    'nuc_area',
]

# Handle potential division by zero or infinite values
ratio_columns = [
    'nuc_area_per_cell_area',
    'nuc_circularity_per_cell_circularity',
    'nuc_longest_axis_per_cell_longest_axis',
    'nuc_longest_axis_per_cell_shortest_axis'
]


features_toplot = [
        ('cell_area', 'Cell Area (µm²)'),
        ('cell_perimeter', 'Cell Perimeter (µm)'),
        ('cell_circularity', 'Cell Circularity'),
        ('cell_longest_axis', 'Cell Longest Axis (µm)'),
        ('cell_shortest_axis', 'Cell Shortest Axis (µm)'),
        ('nuc_area', 'Nucleus Area (µm²)'),
        ('nuc_perimeter', 'Nucleus Perimeter (µm)'),
        ('nuc_circularity', 'Nucleus Circularity'),
        ('nuc_longest_axis', 'Nucleus Longest Axis (µm)'),
        ('nuc_shortest_axis', 'Nucleus Shortest Axis (µm)'),
        ('nuc_area_per_cell_area', 'Nucleus Area / Cell Area'),
        ('cell_eccentricity', 'Cell Eccentricity'),
        ('cell_axis_ratio', 'Cell Axis Ratio'),
        ('nuc_eccentricity', 'Nucleus Eccentricity'),
        ('nuc_axis_ratio', 'Nucleus Axis Ratio'),
        ('nuc_circularity_per_cell_circularity', 'Nucleus Circ. / Cell Circ.'),
        ('nuc_longest_axis_per_cell_longest_axis', 'Nuc Long. Axis / Cell Long. Axis'),
        ('nuc_longest_axis_per_cell_shortest_axis', 'Nuc Long. Axis / Cell Short. Axis'),
        ('mutual_extent','Extent (µm)'),
        ('mutual_centroid_distance', 'Cell–Nucleus COM Distance (µm)'),
        ('mse_diff','mse_diff')
    ]
###
def plot_cell_segs(full_image_raw,full_image,cell,nuc,cyto):
    vmin=0
    vmax=0.1
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 7))  # Adjust figsize to fit your needs


    axs[0].imshow(full_image_raw)  # Assuming cyto_part_in_image is one of your images
    axs[0].axis('off')  # Remove axis ticks and labels
    axs[0].set_title('full_image_raw')  # Optional: Set a title for the first image

    
    axs[1].imshow(full_image,vmin=vmin, vmax=vmax)  # Assuming cyto_part_in_image is one of your images
    axs[1].axis('off')  # Remove axis ticks and labels
    axs[1].set_title('full_image')  # Optional: Set a title for the first image

    
    axs[2].imshow(cell,vmin=vmin, vmax=vmax)  # Replace 'image2' with your second image variable
    axs[2].axis('off')
    axs[2].set_title('cell')

    axs[3].imshow(nuc,vmin=vmin, vmax=vmax)  # Replace 'image3' with your third image variable
    axs[3].axis('off')
    axs[3].set_title('nuc')

    axs[4].imshow(cyto,vmin=vmin, vmax=vmax)  # Replace 'image4' with your fourth image variable
    axs[4].axis('off')
    axs[4].set_title('cyto')

    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.show()  # Display the images
    
# images2.keys()

    
def plot_seg_on_image2(im, segs):
    im2 = im.copy()
    # Ensure the segments are iterable, in case it's a single Polygon object
    if isinstance(segs, MultiPolygon):
        polygons = [np.array(poly.exterior.coords, np.int32) for poly in segs.geoms]
    else:  # Single Polygon object
        polygons = [np.array(segs.exterior.coords, np.int32)]
    
    for polygon in polygons:
        # OpenCV expects points as a shape of (1, -1, 2)
        polygon = polygon.reshape(1, -1, 2)
        cv2.polylines(im2, [polygon], isClosed=True, color=(255, 0, 0), thickness=5)
    
    plt.imshow(im2)
    plt.axis('off')  # Optional: Removes the axis for a cleaner image display
    plt.show()
    
    

def plot_embedded(nuc_embed):
    # Create a scatter plot of the 2D embeddings
    fig = go.Figure(data=[go.Scatter(
        x=nuc_embed[:, 0],
        y=nuc_embed[:, 1],
        mode='markers',
        text=[str(i) for i in range(len(nuc_embed))],
        textposition="top center",
        marker=dict(
            size=5,
            color='LightSkyBlue',
        )
    )])

    fig.update_layout(
        title='2D Embedding of Images with ISOMAP',
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        hovermode='closest',
            width=1000,  # Width in pixels
        height=700,  # Height in pixels
    )

    # Show the figure
    fig.show()
    

def plot_with_sample(n_sample,nuc_embed,d1,d2,recreated_nuc,image_zoom= 0.2,figsize=(30, 20)):
    np.random.seed(0)  # For reproducible output
    kmeans = KMeans(n_clusters=n_samples, random_state=0).fit(nuc_embed)
    selected_indices_dummy = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = np.linalg.norm(nuc_embed - kmeans.cluster_centers_[i], axis=1)
        selected_indices_dummy[i] = dists.argmin()
    fig, ax = plt.subplots(figsize=figsize, dpi = 250)
    ax.scatter(nuc_embed[:, d1], nuc_embed[:, d2], alpha=0.5)
    for idx in selected_indices_dummy:
        img = recreated_nuc[idx]  # Get the dummy image
        imagebox = OffsetImage(img, zoom=image_zoom, cmap = 'gray')
        ab = AnnotationBbox(imagebox, (nuc_embed[idx, d1], nuc_embed[idx, d2]), frameon=False)
        ax.add_artist(ab)
    plt.show()

    

    
def plot_seg_and_line(im, polygon,small_polygon, longest_line):
    im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    
    # Draw the polygon
    polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(im2, [polygon_coords], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Draw the longest line
    if longest_line:
        x, y = longest_line.xy
        pt1 = (int(x[0]), int(y[0]))
        pt2 = (int(x[1]), int(y[1]))
        cv2.line(im2, pt1, pt2, color=(255, 255, 255), thickness=2)
    
    # Draw the smaller polygon
    small_polygon_coords = np.array(small_polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(im2, [small_polygon_coords], isClosed=True, color=(0, 255, 0), thickness=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.axis('off')
    plt.show()


    
def plot_seg_on_image2(im, segs):
    im2 = im.copy()
    # Ensure the segments are iterable, in case it's a single Polygon object
    if isinstance(segs, MultiPolygon):
        polygons = [np.array(poly.exterior.coords, np.int32) for poly in segs.geoms]
    else:  # Single Polygon object
        polygons = [np.array(segs.exterior.coords, np.int32)]
    
    for polygon in polygons:
        # OpenCV expects points as a shape of (1, -1, 2)
        polygon = polygon.reshape(1, -1, 2)
        cv2.polylines(im2, [polygon], isClosed=True, color=(255, 0, 0), thickness=5)
    
    return im2
    # plt.imshow(im2)
    # plt.axis('off')  # Optional: Removes the axis for a cleaner image display
    # plt.show()
    
    
def plot_seg_and_line_axis(ax,im, polygon,small_polygon, longest_line):
    # im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im2 = im.copy()
    # Draw the polygon
    polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(im2, [polygon_coords], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Draw the longest line
    if longest_line:
        x, y = longest_line.xy
        pt1 = (int(x[0]), int(y[0]))
        pt2 = (int(x[1]), int(y[1]))
        cv2.line(im2, pt1, pt2, color=(255, 255, 255), thickness=2)
    
    # # Draw the smaller polygon
    # small_polygon_coords = np.array(small_polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    # cv2.polylines(im2, [small_polygon_coords], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # large_polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    # cv2.polylines(im2, [large_polygon_coords], isClosed=True, color=(255, 0, 255), thickness=2)
    
    ax.imshow(im2)



    
    
    
def plot_seg(im, polygon, small_polygon):
    # Create a copy of the image
    im2 = im.copy()
    
    # Draw the main polygon
    if polygon:
        polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(im2, [polygon_coords], isClosed=True, color=(0, 0, 255), thickness=5)  # Blue color for the main polygon
    
    # Draw the smaller polygon
    if small_polygon:
        small_polygon_coords = np.array(small_polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(im2, [small_polygon_coords], isClosed=True, color=(255, 0, 0), thickness=5)  # Green color for the smaller polygon

    # Show the image with the drawn shapes
    # plt.imshow(im2)
    return im2
    # plt.set_xlabel('')
    # plt.set_xticks([])
    # plt.set_yticks([])

def plot_seg_and_line( im, polygon, small_polygon, longest_line):
    # Create a copy of the image
    im2 = im.copy()
    
    # Draw the main polygon
    if polygon:
        polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(im2, [polygon_coords], isClosed=True, color=(255, 0, 0), thickness=5)  # Blue color for the main polygon
    
    # Draw the smaller polygon
    if small_polygon:
        small_polygon_coords = np.array(small_polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(im2, [small_polygon_coords], isClosed=True, color=(255, 0, 0), thickness=5)  # Green color for the smaller polygon
    
    # Draw the longest line
    if longest_line:
        x, y = longest_line.xy
        pt1 = (int(x[0]), int(y[0]))
        pt2 = (int(x[1]), int(y[1]))
        cv2.line(im2, pt1, pt2, color=(255, 255, 255), thickness=5)  # White color for the longest line
    
    return im2
    
def plot_seg_and_line_axis(ax,im, polygon,small_polygon, longest_line):
    # im2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im2 = im.copy()
    # Draw the polygon
    polygon_coords = np.array(polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(im2, [polygon_coords], isClosed=True, color=(255, 0, 0), thickness=2)
    
    # Draw the longest line
    if longest_line:
        x, y = longest_line.xy
        pt1 = (int(x[0]), int(y[0]))
        pt2 = (int(x[1]), int(y[1]))
        cv2.line(im2, pt1, pt2, color=(255, 255, 255), thickness=2)
    
    # Draw the smaller polygon
    small_polygon_coords = np.array(small_polygon.exterior.coords, np.int32).reshape((-1, 1, 2))
    cv2.polylines(im2, [small_polygon_coords], isClosed=True, color=(0, 255, 0), thickness=2)
    ax.imshow(im2)

    
def plot_polygons_and_their_lines(df2,lines,polygons_dic,images_path_rgb_only,images_path_org):
    ind = df2.index
    for counter, n in  enumerate (ind):
        print(counter, '/', len(ind))
        name = df2.loc[n,'image_name']
        kl_val = df2.loc[n,'mse_diff']
        print(name,kl_val)
        longest_line,x, zi, x_inside, zi_inside = lines[(n,'ku80')]
        longest_line,x_nuc, zi_nuc, x_inside_nuc, zi_inside_nuc= lines[(n,'dapi')]
        nuc =  polygons_dic[(n,'nuc')]
        window_length = int(len(zi)/3)
        poly_order = 3
        zi_filtered = savgol_filter(zi, window_length, poly_order)
        zi_nuc_filtered = savgol_filter(zi_nuc, window_length, poly_order)
        a1 = zi_filtered/np.max(zi_filtered)
        a2 = zi_nuc_filtered/np.max(zi_nuc_filtered)
        image1 = load_image(name+'_RGB', images_path_rgb_only)
        print(image1.shape)
        image2 = load_image(name+'_Merge', images_path_org)    
        nuc_poly = Polygon(polygons_dic[(n,'nuc_not_scaled')])
        cell_poly = Polygon(polygons_dic[(n,'cell')])
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 4))  # Adjust figsize to fit your needs
        plot_seg_and_line_axis(axs[0],image2, cell_poly,nuc_poly, longest_line)
        plot_seg_and_line_axis(axs[1],image1[:,:,1], cell_poly,nuc_poly, longest_line)
        plot_seg_and_line_axis(axs[2],image1[:,:,2], cell_poly,nuc_poly, longest_line)
        axs[3].plot(zi/np.max(zi), color = 'red', label ='KU80') 
        axs[3].plot(zi_nuc/np.max(zi_nuc), color = 'blue', label ='Dapi')

        axs[4].plot(a1, color = 'red', label ='KU80') 
        axs[4].plot(a2, color = 'blue', label ='Dapi')
        plt.legend()
        plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
import numpy as np
from PIL import Image

def display_image_with_polygons(image, polygon1, polygon2):

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # --- Left Panel: Original Image ---
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')  # Hide axis

    # --- Right Panel: Image with Polygons ---
    axes[1].imshow(image)
    axes[1].set_title('Image with Polygons')
    axes[1].axis('off')  # Hide axis

    # Function to add a polygon to an axis
    def add_polygon(ax, polygon, edge_color, face_color, label):
        """
        Adds a Shapely polygon to a Matplotlib axis.

        Parameters:
        - ax (matplotlib.axes.Axes): The axis to draw on.
        - polygon (shapely.geometry.Polygon): The polygon to draw.
        - edge_color (str): Color of the polygon edge.
        - face_color (str): Fill color of the polygon.
        - label (str): Label for the polygon.

        Returns:
        - None
        """
        if not polygon.is_valid:
            print(f"Invalid polygon: {label}")
            return

        x, y = polygon.exterior.xy
        mpl_poly = MplPolygon(np.array([x, y]).T, closed=True,
                              edgecolor=edge_color, facecolor=face_color, alpha=0.5, label=label)
        ax.add_patch(mpl_poly)

    # Add first polygon
    add_polygon(axes[1], polygon1, edge_color='red', face_color='red', label='Polygon 1')

    # Add second polygon
    add_polygon(axes[1], polygon2, edge_color='blue', face_color='blue', label='Polygon 2')

    # Optional: Add legend
    axes[1].legend(loc='upper right')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    
def get_significance_marker(p):
    """
    Return a star annotation based on p-value thresholds.
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'  # Not significant

# -------------------------------
# Plot 1: Overall Rupture vs. Non-Rupture Across Features
# -------------------------------
def plot_overall_rupture_significance(df,features):
        # -------------------------------
    # Global Matplotlib Settings
    # -------------------------------
    mpl.rcParams['axes.unicode_minus'] = True
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.formatter.use_mathtext'] = False
    mpl.rcParams['pdf.fonttype']    = 42
    mpl.rcParams['ps.fonttype']     = 42
    mpl.rcParams['font.size']       = 14
    mpl.rcParams['axes.labelsize']  = 16
    mpl.rcParams['axes.titlesize']  = 18
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    """
    Create a grid of subplots comparing 'ruptured_man' groups for each feature.
    
    This plot uses boxplots and overlaid stripplots to show the distributions of 
    each feature for ruptured vs. non-ruptured cases. Statistical significance 
    is annotated based on the Mann–Whitney U test.
    """

    # Keep only the necessary columns and drop rows with missing data
    required_cols = ['ruptured_man'] + [f[0] for f in features]
    df_plot = df[required_cols].copy().dropna()
    
    sns.set(style="white")
    palette = {'False': "#00ADDC", 'True': "#F15A22"}
    
    n_features = len(features)
    cols = 6
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flat
    
    for i, (feature_col, feature_label) in enumerate(features):
        ax = axes_flat[i]
        # Create boxplot
        sns.boxplot(
            x='ruptured_man',
            y=feature_col,
            data=df_plot,
            palette=palette,
            showfliers=False,
            linewidth=1.2,
            width=0.6,
            ax=ax
        )
        # Overlay stripplot for individual data points
        sns.stripplot(
            x='ruptured_man',
            y=feature_col,
            data=df_plot,
            color='black',
            alpha=0.6,
            size=2,
            jitter=True,
            ax=ax
        )
        
        ax.set_title(feature_label, fontsize=12, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=10)
        ax.set_xticklabels(["Non-Ruptured", "Ruptured"], rotation=0, ha='center')
        
        # Limit number of y-ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which='both', direction='out')
        
        # Apply logarithmic scale for mse_diff
        if feature_col == 'mse_diff':
            ax.set_yscale('log')
            ax.set_ylabel("Value (log scale)", fontsize=10)
        
        # Calculate statistical significance using the Mann–Whitney U test
        group_non_rup = df_plot[df_plot['ruptured_man'] == False][feature_col]
        group_rup = df_plot[df_plot['ruptured_man'] == True][feature_col]
        
        if len(group_non_rup) > 0 and len(group_rup) > 0:
            stat, pval = mannwhitneyu(group_non_rup, group_rup, alternative='two-sided')
            marker = get_significance_marker(pval)
            
            if marker != 'ns':
                max_group_val = max(group_non_rup.max(), group_rup.max())
                y_buffer = 0.03 * (df_plot[feature_col].max() - df_plot[feature_col].min())
                line_y = max_group_val + y_buffer
                ax.text(
                    0.5,
                    line_y,
                    marker,
                    ha='center',
                    va='bottom',
                    color='black',
                    fontsize=12,
                    fontweight='bold'
                )
    
    # Hide any unused subplots
    for j in range(n_features, rows * cols):
        axes_flat[j].set_visible(False)
    
    # Add a single legend for both groups
    handles = [mpl.patches.Patch(color=palette['False'], label="Non-Ruptured"),
               mpl.patches.Patch(color=palette['True'], label="Ruptured")]
    fig.legend(handles, ["Non-Ruptured", "Ruptured"], loc="upper center", ncol=2, fontsize=10, frameon=False)
    
    plt.tight_layout()
    plt.savefig("ruptured_vs_nonruptured.svg", dpi=300, bbox_inches='tight', format='svg')
    plt.show()


# -------------------------------
# Plot 2: Per-Experiment Comparison with Significance
# -------------------------------
def plot_per_experiment_significance(df,features):
        # -------------------------------
    # Global Matplotlib Settings
    # -------------------------------
    mpl.rcParams['axes.unicode_minus'] = True
    mpl.rcParams['font.sans-serif'] = "DejaVu Sans"
    mpl.rcParams['font.family'] = "sans-serif"
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['axes.formatter.use_mathtext'] = False
    mpl.rcParams['pdf.fonttype']    = 42
    mpl.rcParams['ps.fonttype']     = 42
    mpl.rcParams['font.size']       = 14
    mpl.rcParams['axes.labelsize']  = 16
    mpl.rcParams['axes.titlesize']  = 18
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    """
    Create a grid of subplots comparing 'ruptured_man' groups for each feature per experiment.
    
    The x-axis represents different experiments (sample names) with both groups shown as hues.
    Statistical significance for each experiment is computed with the Mann–Whitney U test
    and annotated on the plots.
    """
    # Define features: tuple of (column name, display label)

    
    # Keep only the necessary columns and drop rows with missing data
    required_cols = ['ruptured_man', 'sample_name'] + [f[0] for f in features]
    df_plot = df[required_cols].copy().dropna()
    experiment_order = sorted(df_plot['sample_name'].unique())
    
    sns.set_context("talk")
    sns.set(style="white")
    palette = {False: "#00ADDC", True: "#F15A22"}
    # palette = {'False': "#00ADDC", 'True': "#F15A22"}

    
    n_features = len(features)
    cols = 6
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axes_flat = axes.flat
    
    for i, (feature_col, feature_label) in enumerate(features):
        ax = axes_flat[i]
        # Create boxplot with hue for rupture status
        sns.boxplot(
            x='sample_name',
            y=feature_col,
            hue='ruptured_man',
            data=df_plot,
            order=experiment_order,
            palette=palette,
            showfliers=False,
            linewidth=1.5,
            width=0.6,
            ax=ax,
            dodge=True
        )
        # Overlay stripplot for individual data points
        sns.stripplot(
            x='sample_name',
            y=feature_col,
            hue='ruptured_man',
            data=df_plot,
            order=experiment_order,
            color='black',
            alpha=0.6,
            size=3,
            jitter=True,
            dodge=True,
            ax=ax
        )
        
        # Remove duplicate legends from individual subplots
        if ax.get_legend():
            ax.legend_.remove()
        
        ax.set_title(feature_label, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which='both', direction='out')
        
        # Use logarithmic scale for mse_diff feature
        if feature_col == 'mse_diff':
            ax.set_yscale('log')
            ax.set_ylabel("Value (log scale)")
        
        # Annotate statistical significance for each experiment
        for x_idx, exp_name in enumerate(experiment_order):
            df_exp = df_plot[df_plot['sample_name'] == exp_name]
            group_non_rup = df_exp[df_exp['ruptured_man'] == False][feature_col]
            group_rup = df_exp[df_exp['ruptured_man'] == True][feature_col]
            
            if len(group_non_rup) > 0 and len(group_rup) > 0:
                stat, pval = mannwhitneyu(group_non_rup, group_rup, alternative='two-sided')
                marker = get_significance_marker(pval)
                
                if marker != 'ns':
                    max_group_val = max(group_non_rup.max(), group_rup.max())
                    y_buffer = 0.03 * (df_plot[feature_col].max() - df_plot[feature_col].min())
                    line_y = max_group_val + y_buffer
                    # Define x positions for the significance line
                    x_non_rup = x_idx - 0.2
                    x_rup = x_idx + 0.2
                    ax.plot(
                        [x_non_rup, x_non_rup, x_rup, x_rup],
                        [line_y, line_y + y_buffer * 0.2, line_y + y_buffer * 0.2, line_y],
                        lw=1.5, c='black'
                    )
                    ax.text(
                        (x_non_rup + x_rup) * 0.5,
                        line_y + y_buffer * 0.22,
                        marker,
                        ha='center',
                        va='bottom',
                        color='black',
                        fontsize=16,
                        fontweight='bold'
                    )
    
    # Hide any unused subplots
    for j in range(n_features, rows * cols):
        axes_flat[j].set_visible(False)
    
    # Add a single legend (only the first two handles are needed)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    handles, labels = handles[:2], ["Non-Ruptured", "Ruptured"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("per_experiment_significance_plots.png", dpi=300, bbox_inches='tight', format='png')
    plt.show()

