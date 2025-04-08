# utils.py
import json
import os
import cv2
import math
from math import sqrt
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, mapping, Point,MultiPolygon, LineString,   MultiLineString, box,GeometryCollection
from shapely.affinity import rotate,scale,translate
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from skimage.draw import polygon as draw_polygon
from skimage.draw import polygon2mask
from skimage import io, morphology, img_as_float
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from skimage.transform import rescale, resize
from joblib import Parallel, delayed
import scipy.stats as stats
from scipy.signal import savgol_filter,find_peaks,peak_widths,argrelextrema
from scipy.spatial import ConvexHull
from scipy.ndimage import measurements,map_coordinates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter
import seaborn as sns
import re
from skimage.measure import label, regionprops
from scipy.ndimage import shift
import sys
sys.path.append("PREP/")


def get_pairs_of_cell_nuc_percell_90p(mask_for_img):
    polygons = []
    properties = []  # To store properties to identify nucleus and cell
    
    # Parse and store polygons and their properties from the mask data
    if len(mask_for_img['features']) > 0:
        for seg in mask_for_img['features']:
            if seg['geometry']['type'] == 'Polygon':
                poly = seg['geometry']['coordinates'][0]  # Assuming the first set of coordinates represent the polygon
                polygons.append(Polygon(poly))
                properties.append(seg['properties'])
            else:
                polygons.append(None)
                properties.append(None)

    background = []
    final_pairs = set()

    # Find intersecting pairs that meet the overlap criterion
    for i, poly1 in enumerate(polygons):
        if poly1 is None or not 'nuc' in properties[i]['name'].lower():
            continue
        for j, poly2 in enumerate(polygons):
            if i != j and poly2 and 'cel' in properties[j]['name'].lower() and poly1.intersects(poly2):
                intersection_area = poly1.intersection(poly2).area
                if intersection_area / poly1.area > 0.9:  # More than 90% overlap
                    final_pairs.add((i, j))
            
            if properties[j] and 'back' in properties[j]['name'].lower():
                background.append(j)
            if properties[i] and 'back' in properties[i]['name'].lower():
                background.append(i)

    # Convert to list of tuples and remove duplicates from background
    final_pairs_list = list(final_pairs)
    background = list(set(background))

    if background:
        return final_pairs_list, background[0]
    else:
        return final_pairs_list, None




# get image and corresponding mask by its name
def mask_image_pair(masks,images,key):
    # sample1 = list(masks.keys())[numb]
    image1 = images[key]
    image1 = np.array(image1)
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB
    mask_sample1= masks[key]
    return image,mask_sample1


def plot_cell_channels(im1):
    plt.sfig, axs = plt.subplots(1, 4, figsize=(15, 5), dpi = 300)
    for counter,ax in enumerate(axs):
        if counter == 3:
            ax.imshow(im1,  vmin=0, vmax=255)
        else:
            ax.imshow(im1[:,:,counter].reshape(im1.shape[0],im1.shape[1],1),cmap= 'gray', vmin=0, vmax=255)
        ax.axis('off') 
    plt.show()

    
def calculate_ellipse_orientation(polygon):
    # Extract x and y coordinates from the polygon
    x, y = polygon.exterior.xy
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the centroid
    x_centroid = np.mean(x)
    y_centroid = np.mean(y)
    # Central moments
    mu20 = np.mean((x - x_centroid)**2)
    mu02 = np.mean((y - y_centroid)**2)
    mu11 = np.mean((x - x_centroid)*(y - y_centroid))
    # Orientation of the ellipse
    theta_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg



def calculate_polygon_properties(polygon):
    properties = {}
    # Area
    properties['area'] = polygon.area
    # Perimeter
    properties['perimeter'] = polygon.length
    # Circularity: 4*pi*Area / Perimeter^2
    properties['circularity'] = 4 * math.pi * properties['area'] / (properties['perimeter'] ** 2) if properties['perimeter'] else 0
    # Minimum Rotated Rectangle (Bounding Box)
    mrr = polygon.minimum_rotated_rectangle
    x, y = mrr.exterior.coords.xy
    edge_lengths = [math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2) for i in range(1, len(x))]
    longest_axis = max(edge_lengths)
    shortest_axis = min(edge_lengths)
    # Longest Axis
    properties['longest_axis'] = longest_axis
    properties['shortest_axis'] = shortest_axis
    # Ratio of the Longest Axis to the Shortest Axis
    properties['axis_ratio'] = longest_axis / shortest_axis if shortest_axis else float('inf')
    properties['ellipse_orientation'] = calculate_ellipse_orientation(polygon)
    # Eccentricity: sqrt(1 - (shortest_axis^2 / longest_axis^2))
    properties['eccentricity'] = math.sqrt(1 - (shortest_axis ** 2 / longest_axis ** 2)) if longest_axis else 0

    return properties




def calculate_additional_properties(nucleus_polygon, cell_polygon):
    """
    Calculate the distance between the centroids of the nucleus and cell polygons,
    and compute the extent of the protrusion region.

    Parameters:
    - nucleus_polygon (shapely.geometry.Polygon): The polygon representing the nucleus.
    - cell_polygon (shapely.geometry.Polygon): The polygon representing the cell.

    Returns:
    - dict: A dictionary containing:
        - 'centroid_distance': Distance between the centroids of nucleus and cell.
        - 'extent': Difference between max and min distances from nucleus centroid to protrusion.
    """
    properties = {}

    # Calculate centroids
    nucleus_centroid = nucleus_polygon.centroid
    cell_centroid = cell_polygon.centroid

    # Distance between centroids
    centroid_distance = nucleus_centroid.distance(cell_centroid)
    properties['centroid_distance'] = centroid_distance

    # Define protrusion region: cell polygon minus nucleus polygon
    protrusion = cell_polygon.difference(nucleus_polygon)

    # Initialize extent
    extent = 0

    if not protrusion.is_empty:
        # Collect all exterior coordinates from the protrusion
        protrusion_coords = []

        # Handle different geometry types
        if isinstance(protrusion, Polygon):
            protrusion_coords.extend(protrusion.exterior.coords)
        elif isinstance(protrusion, MultiPolygon) or isinstance(protrusion, GeometryCollection):
            for geom in protrusion.geoms:
                if isinstance(geom, Polygon):
                    protrusion_coords.extend(geom.exterior.coords)
        else:
            # Handle other geometry types if necessary
            pass

        # Calculate distances from nucleus centroid to all protrusion points
        distances = [
            nucleus_centroid.distance(Point(x, y)) 
            for x, y in protrusion_coords
        ]

        if distances:
            max_distance = max(distances)
            min_distance = min(distances)
            extent = max_distance - min_distance

    properties['extent'] = extent

    return properties


def scale_polygon_float(polygon, scale_factor=0.5):
    """
    Scale a polygon by a given scale factor.

    Parameters:
    - polygon: A list of tuples representing the polygon's vertices [(x1, y1), (x2, y2), ...].
    - scale_factor: The factor by which to scale the polygon (0.5 to reduce by half).

    Returns:
    - A new list of tuples representing the scaled polygon's vertices.
    """
    # Calculate the centroid of the polygon
    centroid_x = sum(x for x, _ in polygon) / len(polygon)
    centroid_y = sum(y for _, y in polygon) / len(polygon)

    # Scale each vertex
    scaled_polygon = [
        ((x - centroid_x) * scale_factor + centroid_x, (y - centroid_y) * scale_factor + centroid_y)
        for x, y in polygon
    ]

    return scaled_polygon


def scale_polygan_p2(img_name,poly_):
    poly_shape = poly_.copy() # only for size related values
    if '60x' in img_name.lower():
        poly_shape = scale_polygon(poly_shape, scale_factor=1/3)
    elif '40x' in img_name.lower():
        poly_shape = scale_polygon(poly_shape, scale_factor=1/2)
    elif '30x' in img_name.lower():
        poly_shape = scale_polygon(poly_shape, scale_factor=2/3)  
    return poly_shape


# transfer polygon to its (0,0)
def normalize_polygon(polygon):
    shapely_polygon = Polygon(polygon)
    centroid = shapely_polygon.centroid
    # Translate polygon to have its centroid at (0,0)
    normalized_polygon = [(x - centroid.x, y - centroid.y) for x, y in polygon]
    return normalized_polygon

def center_and_draw_polygon(polygon, image_size):
    normalized_polygon = normalize_polygon(polygon)
    img = Image.new('L', (image_size, image_size), 'black')
    draw = ImageDraw.Draw(img)
    
    # After normalization, the polygon's centroid is at (0,0).
    # To center this on the image, we calculate the offset
    # by taking half of the image size.
    # This positions the centroid of the polygon at the center of the image.
    translate_x = image_size / 2
    translate_y = image_size / 2
    
    # Apply translation to each vertex for centering
    centered_polygon = [(x + translate_x, y + translate_y) for x, y in normalized_polygon]
    
    # Draw the centered polygon
    draw.polygon(centered_polygon, outline='white', fill='white')
    return img



def center_and_draw_polygon_border(polygon, image_w,image_h):
    # Normalize the polygon to start at (0,0)
    normalized_polygon = normalize_polygon(polygon)
    
    # Start with a white background and plan to draw the polygon borders in black
    img = Image.new('L', (image_w, image_h), 'black') # background
    draw = ImageDraw.Draw(img)
    
    # Since the polygon is normalized, centering it is straightforward
    # shapely_polygon = Polygon(normalized_polygon) # Uncomment if using Shapely for calculations
    min_x = min(p[0] for p in normalized_polygon)
    min_y = min(p[1] for p in normalized_polygon)
    max_x = max(p[0] for p in normalized_polygon)
    max_y = max(p[1] for p in normalized_polygon)
    width = max_x - min_x
    height = max_y - min_y
    translate_x = (image_w - width) / 2 - min_x
    translate_y = (image_h - height) / 2 - min_y
    
    # Apply translation to each vertex for centering
    centered_polygon = [(x + translate_x, y + translate_y) for x, y in normalized_polygon]
    
    # Draw the centered polygon with a black outline and no fill to get only the borders
    # Specify the 'width' for the outline to make it thicker and visible
    draw.polygon(centered_polygon, outline='white', fill='white', width=1)
    
    return img



def normalize_polygon(polygon):
    min_x = min([p[0] for p in polygon])
    min_y = min([p[1] for p in polygon])
    normalized_polygon = [(p[0] - min_x, p[1] - min_y) for p in polygon]
    return normalized_polygon



def overlay_images_custom(images):
    if not images:
        return None  # Or handle this case as you see fit
    
    # Start with a completely white image
    base_image = Image.new('L', images[0].size, 'white')
    
    for img in images:
        # Iterate over each pixel
        for x in range(img.width):
            for y in range(img.height):
                pixel = img.getpixel((x, y))
                # If the pixel is not white (assuming black or close to black is important),
                # overlay it on the base image
                if pixel != 255:  # Not white, thus important
                    base_image.putpixel((x, y), pixel)
    
    return base_image


def calculate_white_pixel_counts(images):
    """Calculate the count of white pixels."""
    # Assuming all images are the same size, get the size of the first image
    width, height = images[0].size
    
    # Initialize a numpy array to hold the count of white pixels
    white_pixel_count = np.zeros((height, width))
    
    # Iterate over each image, convert it to a numpy array, and count white pixels
    for img in images:
        img_array = np.array(img)
        white_pixel_count += (img_array == 255)  # Assuming white pixels are 255
    
    return white_pixel_count

def plot_white_pixel_counts(counts):
    """Plot the counts of white pixels."""
    plt.imshow(counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Counts of white pixels')
    plt.show()

    


def normalize_image(image): # between 0 and 1
    # Convert the image to a numpy array to ensure compatibility with numpy operations, if it's not already one.
    image_array = np.array(image)
    # Find the maximum pixel value in the image
    max_value = image_array.max()
    # Normalize the image by dividing by the maximum value
    normalized_image = image_array / max_value
    return normalized_image


    
def boolean_seg_for_one_channel_image(image, polygon_coords): # returns the boolean matrix of one speciif channel in polygon exists or not
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    polygon_coords = np.array(polygon_coords)
    if isinstance(polygon_coords, np.ndarray):
        polygon_coords = list(map(tuple, polygon_coords))
    mask_image = Image.new('L', (image_np.shape[1], image_np.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon(polygon_coords, outline=1, fill=1)
    mask = np.array(mask_image)
    mask_bool = mask > 0  # Convert mask to boolean

    return mask_bool



# get the boolean matrixes of cell, nuc and background, combines them, now normalize those area between 0 and 1, return the normlaized image of that
def prep_normalized_image_for_back_nuc_cell(image_ku80,poly_background,cell_polygon,scaled_nuc_polygon):
    
        back_bool =  boolean_seg_for_one_channel_image(image_ku80, poly_background)
        cell_bool =  boolean_seg_for_one_channel_image(image_ku80, cell_polygon)
        nuc_bool =  boolean_seg_for_one_channel_image(image_ku80, scaled_nuc_polygon)
        
        # combined_mask = np.logical_or(cell_bool, back_bool)
        combined_mask = np.logical_or(cell_bool, back_bool) & ~nuc_bool
        # combined_mask = np.logical_or(cell_bool, nuc_bool)
        # combined_mask = np.logical_or(back_bool, np.logical_or(cell_bool, nuc_bool))

        image_channel = image_ku80[:, :, 0] if image_ku80.ndim == 3 else image_ku80
        image_back_cell_nuc_specific = np.where(combined_mask, image_channel, 0)
        # image_back_cell_nuc_specific = np.where(np.logical_or(cell_bool, nuc_bool), image_ku80[:,:,0], 0)
        image_ku80_normalized = normalize_image(image_back_cell_nuc_specific)
        # image_ku80_normalized = image_back_cell_nuc_specific
        image_ku80_normalized = image_ku80_normalized.reshape(image_ku80_normalized.shape[0],image_ku80_normalized.shape[1],1)
        # so in above we have an image containing cell, nuc and background and normalized between them 0,1

        return image_ku80_normalized

def load_image(image_name, images_path):
    image_filename = image_name + '.tif'  # Adjust the extension if necessary
    image_filepath = os.path.join(images_path, image_filename)
    with Image.open(image_filepath) as img:
        img = img.convert('RGB')
        image = np.array(img)
    return image

def load_mask(image_name, preds_path):
    # Adjust the filename pattern based on your naming convention
    mask_filename = image_name + '.json'
    mask_filepath = os.path.join(preds_path, mask_filename)
    with open(mask_filepath, 'r') as file:
        mask = json.load(file)
    return mask

def get_image_names_from_masks(preds_path):
    image_names = []
    for filename in os.listdir(preds_path):
        if filename.endswith('.json'):
            # Extract image name from mask filename
            file_base = filename.split('.')[0]
            mask_name_parts = file_base.split('_')[:-1]
            image_name = '_'.join(mask_name_parts)
            image_names.append(image_name)
    return image_names


def scale_polygon_ref(polygon, scale_factor, ref_point):
    scaled_polygon = []
    for (x, y) in polygon:
        scaled_x = ref_point[0] + scale_factor * (x - ref_point[0])
        scaled_y = ref_point[1] + scale_factor * (y - ref_point[1])
        scaled_polygon.append((scaled_x, scaled_y))
    return scaled_polygon

def scale_polygons_preserve_relative_position(magnification, poly_cell, poly_nuc):
    # Determine the scale factor based on the image name using regex for exact matches
    if int(magnification) == 20:
        scale_factor = 2

    elif int(magnification)  == 40:
        scale_factor = 1 
    elif int(magnification)  == 30:
        scale_factor = 2 / 3
    else:
        scale_factor = 1  # Default scaling
    # Calculate the centroid of the cell polygon as the reference point
    centroid_x =Polygon(poly_cell).centroid.x
    centroid_y = Polygon(poly_cell).centroid.y
    ref_point = (centroid_x, centroid_y)
    poly_cell_scaled = scale_polygon_ref(poly_cell, scale_factor, ref_point)
    poly_nuc_scaled = scale_polygon_ref(poly_nuc, scale_factor, ref_point)
    
    return poly_cell_scaled, poly_nuc_scaled

def close_polygon(polygon):
    if not polygon:
        raise ValueError("Polygon has no coordinates.")
    
    # Compare the first and last coordinates
    first_point = polygon[0]
    last_point = polygon[-1]
    
    if first_point != last_point:
        polygon.append(first_point)
    
    return polygon


def angle_corrector(angle_deg):
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg > 90:
        angle_deg -= 180
    return angle_deg


def angle_of_longest_axis(polygon):
    """
    Returns the angle (in degrees) of the polygon's major axis
    relative to the x-axis, determined via PCA of its exterior coordinates.
    """
    # Extract the exterior coordinates of the polygon
    coords = np.array(polygon.exterior.coords)
    
    # Subtract the mean (centroid) to center the polygon at origin
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid

    # Compute 2x2 covariance matrix
    cov = np.cov(coords_centered.T)

    # Eigen-decomposition
    eigenvals, eigenvecs = np.linalg.eig(cov)

    # Identify the major axis (largest eigenvalue)
    # Sort eigenvalues/eigenvectors from largest to smallest
    sort_idx = np.argsort(eigenvals)[::-1]
    largest_eigenvec = eigenvecs[:, sort_idx[0]]

    # Calculate the angle of this major axis in radians
    angle_rad = np.arctan2(largest_eigenvec[1], largest_eigenvec[0])

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    # Optional: ensure angle is in [-180, 180)
    if angle_deg < 0 :
        angle_deg = angle_deg + 180
    if angle_deg > 180 : 
        angle_deg  = angle_deg  - 180

    return angle_deg

def get_bounding_box(coords):
    """
    Given polygon coordinates coords of shape (N,2),
    return (xmin, ymin, xmax, ymax).
    """
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    return x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()

def get_bbox_scale_factor(coords, target_longest_dim=62.0):
    """
    1. Compute the polygon's axis-aligned bounding box.
    2. Find the bounding box's largest dimension (width or height).
    3. Compute scale_factor = target_longest_dim / largest_dimension.
    
    Returns 1.0 if the polygon is degenerate (largest_dim=0).
    """
    xmin, ymin, xmax, ymax = get_bounding_box(coords)
    width = xmax - xmin
    height = ymax - ymin
    largest_dim = max(width, height)
    
    if largest_dim <= 0:
        return 1.0
    else:
        return target_longest_dim / largest_dim

    

def scale_polygon(coords, scale_factor):
    return coords * scale_factor

def center_polygon(coords, image_size=(128, 128)):
    centroid = coords.mean(axis=0)  # (x_mean, y_mean)
    image_center = np.array([image_size[0]/2.0, image_size[1]/2.0])
    translation = image_center - centroid
    return coords + translation

def rasterize_polygon(coords, image_size=(128, 128)):
    """
    polygon2mask expects (row, col) => we must flip (x, y) -> (y, x)
    """
    coords_rc = np.stack([coords[:,1], coords[:,0]], axis=-1)
    mask = polygon2mask(image_size, coords_rc)
    return mask.astype(np.uint8)

def create_bounding_box_scaled_mask(coords, target_longest_dim=62, final_size=(128, 128)):
    """
    1. Scale polygon so bounding box's largest dimension = target_longest_dim.
    2. Center in final image.
    3. Rasterize into a binary mask.
    """
    # 1) compute scale factor using bounding box
    scale_factor = get_bbox_scale_factor(coords, target_longest_dim)
    
    # 2) scale
    scaled = scale_polygon(coords, scale_factor)
    
    # 3) center
    centered = center_polygon(scaled, image_size=final_size)
    
    # 4) rasterize
    mask = rasterize_polygon(centered, image_size=final_size)
    return mask


def shift_mask(binary_mask, shift_y, shift_x):
    """
    Shift `binary_mask` by (shift_y, shift_x) pixels (integer shifts).
    Positive shift_y => down, shift_x => right.
    """
    shifted = shift(binary_mask, shift=[shift_y, shift_x], order=0, cval=0.0, prefilter=False)
    return (shifted > 0.5).astype(binary_mask.dtype)

def rescale_binary_mask(binary_mask, scale_factor):
    """
    Resample `binary_mask` by `scale_factor` in height and width,
    then threshold back to binary.
    """
    if abs(scale_factor - 1.0) < 1e-9:
        return binary_mask
    
    H, W = binary_mask.shape
    new_h = int(round(H * scale_factor))
    new_w = int(round(W * scale_factor))
    
    # Convert mask to float, resize, then threshold
    resized = resize(binary_mask.astype(float),
                     (new_h, new_w),
                     order=1,  # bilinear
                     mode='constant',
                     cval=0.0,
                     anti_aliasing=False)
    
    resized_bin = (resized > 0.5).astype(binary_mask.dtype)
    return resized_bin


def embed_in_128x128(mask_small):
    """
    Embed a smaller (or larger) 2D mask into a 128×128 canvas, centered.
    """
    h, w = mask_small.shape
    canvas = np.zeros((128, 128), dtype=mask_small.dtype)
    
    # Compute where to place top-left corner
    start_y = (128 - h) // 2
    start_x = (128 - w) // 2
    
    # Range-limited insertion
    end_y = start_y + h
    end_x = start_x + w
    
    # If the mask is bigger than 128, it will clip; 
    # if it's smaller, it will be centered with padding.
    # We can do additional checks if you want to ensure no negative indexes.
    canvas[max(start_y, 0):max(start_y, 0) + h,
           max(start_x, 0):max(start_x, 0) + w] = mask_small[
               0 : min(h, 128),
               0 : min(w, 128)]
    
    return canvas

def adjust_nucleus_area(cell_mask, nuc_mask, cell_area_orig, nuc_area_orig):
    """
    Rescale `nuc_mask` so that the final area ratio (nucleus/cell in pixels)
    matches the original ratio (nuc_area_orig/cell_area_orig).
    Then embed the rescaled nucleus mask into a 128×128 canvas
    so it's shape-compatible with cell_mask.
    """
    A_cell_scaled = cell_mask.sum()
    A_nuc_scaled  = nuc_mask.sum()
    
    # ratio in the final mask
    r_scaled = A_nuc_scaled / (A_cell_scaled + 1e-9)
    
    # ratio in original polygons
    r_orig   = nuc_area_orig / (cell_area_orig + 1e-9)
    
    if r_scaled < 1e-9 or r_orig < 1e-9:
        # degenerate => do nothing
        return embed_in_128x128(nuc_mask)
    
    ratio = r_orig / r_scaled
    scale_factor = np.sqrt(ratio)  # area ~ scale^2
    
    # rescale nucleus
    nuc_mask_rescaled = rescale_binary_mask(nuc_mask, scale_factor)
    
    # embed in 128×128
    nuc_mask_128 = embed_in_128x128(nuc_mask_rescaled)
    return nuc_mask_128
