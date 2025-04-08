#line_method.py
from utils import *
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from skimage.measure import profile_line
from scipy.ndimage import map_coordinates
import sys
sys.path.append("PREP/")

def find_longest_axis_angle(polygon):
    convex_hull = polygon.convex_hull
    max_length = 0
    longest_axis = None
    coords = list(convex_hull.exterior.coords)
    
    for i, point1 in enumerate(coords[:-1]):
        for point2 in coords[i+1:]:
            line = LineString([point1, point2])
            length = line.length
            if length > max_length:
                max_length = length
                longest_axis = line
    
    dx = longest_axis.coords[1][0] - longest_axis.coords[0][0]
    dy = longest_axis.coords[1][1] - longest_axis.coords[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    return angle

def generate_random_lines(cell_polygon, nucleus_polygon, num_lines=100, std_dev=10, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Find the longest axis angle of the cell polygon
    longest_axis_angle = find_longest_axis_angle(cell_polygon)
    
    max_product = 0
    best_line = None
    bad_lines = []
    reasons = []
    
    for _ in range(num_lines):
        # Generate a random angle around the longest axis angle
        angle = np.random.normal(loc=longest_axis_angle, scale=std_dev)
        
        # Choose a random point within the nucleus
        minx, miny, maxx, maxy = nucleus_polygon.bounds
        while True:
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if nucleus_polygon.contains(random_point):
                break
        
        # Create a long line passing through the random point at the chosen angle
        line = LineString([
            translate(random_point, xoff=-1000 * np.cos(np.radians(angle)), yoff=-1000 * np.sin(np.radians(angle))),
            translate(random_point, xoff=1000 * np.cos(np.radians(angle)), yoff=1000 * np.sin(np.radians(angle)))
        ])
        
        # Clip the line to the boundaries of the cell polygon
        clipped_line = cell_polygon.intersection(line)
        
        if clipped_line.is_empty:
            bad_lines.append(clipped_line)
            reasons.append('empty')
            continue
            
        if not clipped_line.intersects(nucleus_polygon):
            bad_lines.append(clipped_line)
            reasons.append('not_intersect')
            continue
        
        # Calculate intersection length with nucleus
        intersecting_line = clipped_line.intersection(nucleus_polygon)
        
        # Skip if the intersection is not a LineString
        if isinstance(intersecting_line, MultiLineString):
            bad_lines.append(clipped_line)
            reasons.append('multi_intersection')
            continue
        
        if isinstance(clipped_line, MultiLineString):
            reasons.append('multi_clipped')
            continue
        
        # Calculate product of the line's length and intersection length
        line_length = clipped_line.length
        intersect_length = intersecting_line.length
        product = line_length * intersect_length
        
        if product > max_product:
            max_product = product
            best_line = clipped_line

    return best_line, bad_lines, reasons


def plot_seg_and_line(im, polygon,small_polygon, longest_line):
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
    
    plt.figure(figsize=(10, 10))
    plt.imshow(im2)
    plt.axis('off')
    plt.show()
    


def pixel_value_profile(image, line, nucleus_polygon):
    # Get the x and y coordinates of the line
    x, y = line.xy
    
    # Use skimage's profile_line to get the pixel values along the line
    zi = profile_line(image, (y[0], x[0]), (y[1], x[1]), order=2)
    
    # Create a LineString from the line points
    line_points = LineString(np.column_stack((x, y)))
    
    # Get the points inside the nucleus polygon
    points_inside = [point for point in line_points.coords if nucleus_polygon.contains(Point(point))]
    
    # Extract the x and y coordinates of the points inside the nucleus polygon
    x_inside = np.array([point[0] for point in points_inside])
    y_inside = np.array([point[1] for point in points_inside])
    
    # Get the pixel values for the points inside the nucleus polygon
    if len(x_inside) > 0 and len(y_inside) > 0:
        zi_inside = map_coordinates(image, np.vstack((y_inside, x_inside)), order=2)
    else:
        zi_inside = np.array([])  # Empty array if no points are inside the nucleus
    
    return x, zi, x_inside, zi_inside

def interpolate_line(line, num_points):
    """
    Interpolates points along a LineString to match the desired number of points.

    Parameters:
    - line: Shapely LineString object.
    - num_points: Integer representing the desired number of points.

    Returns:
    - List of (x, y) tuples representing the interpolated points.
    """
    if num_points < 2:
        raise ValueError("Number of points must be at least 2.")
    
    total_length = line.length
    distances = np.linspace(0, total_length, num_points)
    interpolated_points = [line.interpolate(distance) for distance in distances]
    return [(point.x, point.y) for point in interpolated_points]