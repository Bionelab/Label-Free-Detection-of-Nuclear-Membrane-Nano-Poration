# analyzer.py
from utils import *
from line_method import *
import sys
sys.path.append("PREP/")

# make a new image which just polygon and background
def analyze_polygon(image, polygon_coords, background):
    if isinstance(image, Image.Image):
        # Convert PIL Image to NumPy array
        image_np = np.array(image)
    else:
        image_np = image
    
    polygon_coords = np.array(polygon_coords)
    if isinstance(polygon_coords, np.ndarray):
        polygon_coords = list(map(tuple, polygon_coords))
    
    # Create a mask for the polygon
    mask_image = Image.new('L', (image_np.shape[1], image_np.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon(polygon_coords, outline=1, fill=1)
    mask = np.array(mask_image)
    
    mask_bool = mask > 0  # Convert mask to boolean
    
    # Check if the original image is RGBA (has an alpha channel)
    if image_np.shape[2] == 4:
        # Prepare an empty array with the same shape as the original
        masked_image = np.zeros_like(image_np)
        # Apply the mask to each of the R, G, B, and A channels
        for i in range(4):  # Including the alpha channel
            masked_image[:, :, i] = np.where(mask_bool, image_np[:, :, i], 0)
    else:
        # For RGB or grayscale images without an alpha channel
        masked_image = np.zeros_like(image_np)
        for i in range(image_np.shape[2]):
            masked_image[:, :, i] = np.where(mask_bool, image_np[:, :, i], 0)
    
    results = {}
    # Assuming the image has 3 channels if RGB, or 4 if RGBA
    num_channels = image_np.shape[2]
    # print(num_channels)
    for i in range(num_channels):
        channel = masked_image[:, :, i]
        masked_channel = channel[mask_bool]
        results['sum_'+str(i)] = np.sum(masked_channel)
        if background:
              results['avg_'+str(i)] = np.percentile(masked_channel, 20) if masked_channel.size > 0 else 0
              # results['avg_'+str(i)] = np.mean(masked_channel) if masked_channel.size > 0 else 0
              results['std_'+str(i)] = np.std(masked_channel) if masked_channel.size > 0 else 0
           
          
        else:
            # results['avg_'+str(i)] = np.mean(masked_channel) if masked_channel.size > 0 else 0
            results['avg_'+str(i)] =  np.percentile(masked_channel, 70) if masked_channel.size > 0 else 0
            results['std_'+str(i)] = np.std(masked_channel) if masked_channel.size > 0 else 0
        
    return masked_image,masked_channel,mask_bool, results


def load_and_analyze_polyongs_from_images(polygons_path,images_path_rgb_only,images_path_org, to_plot = False):
    image_names = get_image_names_from_masks(polygons_path)
    df = pd.DataFrame()
    counter = 0
    polygons_dic = {}  # To keep the polygons
    polygons_dic_scaled = {}  
    lines = {}
    for img_name in image_names[:]:
        match = re.search(r'(\d+)x',  img_name.lower())
        magnification = int(match.group(1))
        image = load_image(img_name+'_RGB', images_path_rgb_only)
        image_raw = load_image(img_name+'_Merge', images_path_org)
        mask = load_mask(img_name+'_Merge', polygons_path)
        # we have images and polygons
        maks_pairs, background =   get_pairs_of_cell_nuc_percell_90p(mask)
        image_ku80= image[:,:,1].reshape(image.shape[0],image.shape[1], 1)
        image_dapi= image[:,:,2].reshape(image.shape[0],image.shape[1], 1)
        # poly_background= mask['features'][background]['geometry']['coordinates'][0]
        for nuc_number , cell_number in maks_pairs:
            print(img_name)
            cell_polygon = mask['features'][cell_number]['geometry']['coordinates'][0]
            cell_polygon = close_polygon(cell_polygon.copy())
            nucleus_polygon = mask['features'][nuc_number]['geometry']['coordinates'][0]
            scaled_nuc_polygon = scale_polygon_float(nucleus_polygon, 1.1)
            scaled_nuc_polygon= list(Polygon(scaled_nuc_polygon).exterior.coords)
            image_ku80_normalized = image_ku80
            longest_line = None
            while longest_line is None:
                try:
                    longest_line,bad_lines,reasons = generate_random_lines(Polygon(cell_polygon), Polygon(nucleus_polygon))
                except Exception as e:
                    print(f"An error occurred: {e}. Retrying...")

            x, zi, x_inside, zi_inside = pixel_value_profile(image_ku80[:,:,0], longest_line, Polygon(nucleus_polygon) )
            x_nuc, zi_nuc, x_inside_nuc, zi_inside_nuc = pixel_value_profile(image_dapi[:,:,0], longest_line, Polygon(nucleus_polygon) )
            lines[(counter,'ku80')] = (longest_line,x, zi, x_inside, zi_inside)
            lines[(counter,'dapi')] = (longest_line,x_nuc, zi_nuc, x_inside_nuc, zi_inside_nuc)
            poly_cell_scaled_mic, poly_nuc_scaled_mic = scale_polygons_preserve_relative_position(magnification, cell_polygon, nucleus_polygon)
            cell_props = calculate_polygon_properties(Polygon(poly_cell_scaled_mic))
            cell_masked_image, cell_booled_masked_array, cell_mask_bool, poly_cell_vals = analyze_polygon(image_ku80_normalized, 
                                                                                                      cell_polygon, 
                                                                                                      background = False)
            # nuc part in image, array contains only nuc pixels, boolen for nuc pixels detections, and their values (sum and avg)
            nuc_masked_image, nuc_booled_masked_array, nuc_mask_bool, poly_nuc_vals = analyze_polygon(image_ku80_normalized, nucleus_polygon, background = False)
            nuc_props = calculate_polygon_properties(Polygon(poly_nuc_scaled_mic))
            # scaled nuc part in image, array contains only nuc pixels, boolen for nuc pixels detections, and their values (sum and avg)
            mutual_props = calculate_additional_properties(Polygon(poly_nuc_scaled_mic), Polygon(poly_cell_scaled_mic))

            scaled_nuc_masked_image, scaled_nuc_booled_masked_array, scaled_nuc_mask_bool, scaled_poly_nuc_vals = analyze_polygon(image_ku80_normalized, scaled_nuc_polygon, background = False)
            cyto_part_in_image = np.where(np.logical_and(cell_mask_bool, np.logical_not(scaled_nuc_mask_bool)), image_ku80_normalized[:,:,0], 0)
            channel=image_ku80_normalized[:,:,0] 
            # only contains cytoplasm pixles
            cyto_channel = channel[np.where(np.logical_and(cell_mask_bool, np.logical_not(scaled_nuc_mask_bool)))]

            nuc_poly = Polygon(nucleus_polygon)
            cell_poly = Polygon(cell_polygon)
            if to_plot:
                fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 7))  # Adjust figsize to fit your needs
                plot_seg_and_line_axis(axs[0],image_raw, cell_poly,nuc_poly, longest_line,)
                plot_seg_and_line_axis(axs[1],image_ku80, cell_poly,nuc_poly, longest_line, )
                plot_seg_and_line_axis(axs[2],image_dapi, cell_poly,nuc_poly, longest_line,)
                plt.tight_layout()
                plt.show()
                plt.figure()
                plt.plot(zi/np.max(zi), color = 'red', label = 'KU80')
                plt.plot(zi_nuc/np.max(zi_nuc), color = 'black', label = 'DAPI')
                plt.legend()
                plt.xticks(fontsize=18)  # Change the x-tick size (adjust the fontsize as needed)
                plt.yticks(fontsize=18)  # 
                plt.savefig('/Users/keivanrahmani/Desktop/cells/'+str(cell_number)+'.svg', format='svg', bbox_inches='tight')
                plt.show()        
                plot_cell_segs(image[:,:,:],
                                   image_raw,
                                   cell_masked_image,
                                   scaled_nuc_masked_image,
                                   cyto_part_in_image
                                  )

            df.loc[counter,'magnification']= magnification
            df.loc[counter,'cell_num']= cell_number
            df.loc[counter,'nuc_num']= nuc_number
            df.loc[counter,'image_name']= img_name
            df.loc[counter,['cell_'+i for i in cell_props.keys() ]]= cell_props.values()
            df.loc[counter,['nuc_'+i for i in nuc_props.keys() ]]= nuc_props.values()
            df.loc[counter,['mutual_'+i for i in mutual_props.keys() ]]= mutual_props.values()
            df.loc[counter,'cyto_' +'sum_0']= np.sum(cyto_channel)
            df.loc[counter,'cyto_' +'avg_0']= np.mean(cyto_channel)
            df.loc[counter,'cyto_' +'std_0']= np.std(cyto_channel)
            polygons_dic[(counter,'cell')]=cell_polygon
            polygons_dic[(counter,'nuc')] = scaled_nuc_polygon
            polygons_dic[(counter,'nuc_not_scaled')] = nucleus_polygon
            polygons_dic_scaled[(counter,'cell')]=poly_cell_scaled_mic
            polygons_dic_scaled[(counter,'nuc_not_scaled')] = poly_nuc_scaled_mic
            counter = counter + 1


    df['contains_flat'] = df['image_name'].str.lower().str.contains('flat')
    df['contains_np'] = df['image_name'].str.lower().str.contains('np|d\d+p\d+')
    df[['diameter', 'pitch']] = df['image_name'].str.extract('d(\d+)p(\d+)')
    df = df.rename(columns={'sum_0': 'ku80_pix_sum',
                              'avg_0': 'ku80_pix_avg',

                             })
    return df,polygons_dic , polygons_dic_scaled,lines

# Create a copy of the original DataFrame to store the metrics
def analyze_the_lines(df,lines,polygons_dic):
    df2 = df.copy()
    # Initialize new columns with NaN
    metrics = ['similarity', 'count', 'mse', 'mae',
               'similarity_diff', 'count_diff', 'mse_diff', 'mae_diff']
    for metric in metrics:
        df2[metric] = np.nan

    for n in df2.index:
        # Extract lines for 'ku80' and 'dapi'
        ku80_key = (n, 'ku80')
        dapi_key = (n, 'dapi')

        ku80_data = lines.get(ku80_key, None)
        dapi_data = lines.get(dapi_key, None)

        if ku80_data is None or dapi_data is None:
            # Handle missing data by assigning NaNs
            df2.loc[n, metrics] = np.nan
            print(f"Missing data for index {n}. Metrics set to NaN.")
            continue  # Skip to the next index

        # Unpack the data
        longest_line_ku80, x_ku80, zi, _, _ = ku80_data
        longest_line_dapi, x_dapi, zi_nuc, _, _ = dapi_data

        # Define nucleus polygon for this index
        nuc_coords = polygons_dic.get((n, 'nuc'), None)
        if nuc_coords is None:
            # Handle missing nucleus data
            df2.loc[n, metrics] = np.nan
            print(f"Missing nucleus data for index {n}. Metrics set to NaN.")
            continue  # Skip to the next index

        nuc_polygon = Polygon(nuc_coords)

        # Ensure that the number of points in the line matches the length of zi
        num_zi = len(zi)
        num_coords = len(longest_line_ku80.coords)

        if num_coords != num_zi:
            # Interpolate points along the line to match the length of zi
            try:
                interpolated_coords = interpolate_line(longest_line_ku80, num_zi)
                # Create a new LineString with interpolated points
                longest_line_ku80_interp = LineString(interpolated_coords)
                # Update the coordinates list
                ku80_coords_interp = interpolated_coords
            except ValueError as e:
                print(f"Interpolation error for index {n}: {e}. Metrics set to NaN.")
                df2.loc[n, metrics] = np.nan
                continue
        else:
            # No interpolation needed
            ku80_coords_interp = list(longest_line_ku80.coords)

        # Normalize zi and zi_nuc
        max_zi = np.max(zi)
        a1 = zi / max_zi if max_zi != 0 else zi
        max_zi_nuc = np.max(zi_nuc)
        a2 = zi_nuc / max_zi_nuc if max_zi_nuc != 0 else zi_nuc

        # Apply Savitzky-Golay filter
        poly_order = 3
        window_length = max(3, (len(a1) // 3) | 1)  # Ensure window_length is odd and >= poly_order + 2
        if len(a1) < window_length:
            # Adjust window_length if data is too short
            window_length = len(a1) if len(a1) % 2 != 0 else len(a1) - 1
        try:
            zi_filtered = savgol_filter(a1, window_length, poly_order)
            zi_nuc_filtered = savgol_filter(a2, window_length, poly_order)
        except ValueError:
            # Handle cases where window_length is inappropriate
            zi_filtered = a1
            zi_nuc_filtered = a2
            print(f"Savitzky-Golay filter failed for index {n}. Using unfiltered data.")

        # Compute similarity metrics for the entire line
        try:
            similarity = cosine_similarity(zi_filtered.reshape(-1, 1), zi_nuc_filtered.reshape(-1, 1))[0][0]
        except ValueError:
            similarity = np.nan
            print(f"Cosine similarity calculation failed for index {n}.")

        count = np.sum(a1 > a2)
        mse = mean_squared_error(a1, a2)
        mae = mean_absolute_error(a1, a2)

        # Initialize metrics for the difference part
        similarity_diff = np.nan
        count_diff = np.nan
        mse_diff = np.nan
        mae_diff = np.nan

        # Create a boolean mask for points outside the nucleus
        # Use the interpolated coordinates if interpolation was done
        mask = np.array([not nuc_polygon.contains(Point(coord)) for coord in ku80_coords_interp])

        # Extract zi and zi_nuc for the difference part
        zi_diff = zi_filtered[mask]
        zi_nuc_diff = zi_nuc_filtered[mask]

        # Ensure there are points to compare
        if len(zi_diff) > 0 and len(zi_nuc_diff) > 0:
            # Replace zi_diff with zi_nuc_diff where zi_diff < zi_nuc_diff
            below_mask = zi_diff < zi_nuc_diff        
            try:
                similarity_diff = cosine_similarity(zi_diff.reshape(-1, 1), zi_nuc_diff.reshape(-1, 1))[0][0]
            except ValueError:
                similarity_diff = np.nan
                print(f"Cosine similarity (diff) calculation failed for index {n}.")

            count_diff = np.sum(zi_diff > zi_nuc_diff)
            if np.any(below_mask):
                print(f"Index {n}: zi_diff < zi_nuc_diff found. Replacing zi_diff with zi_nuc_diff where condition is met.")
                zi_diff[below_mask] = zi_nuc_diff[below_mask]
            mse_diff = mean_squared_error(zi_diff, zi_nuc_diff)
            mae_diff = mean_absolute_error(zi_diff, zi_nuc_diff)

            # Calculate the total length of the ku80 line
        total_length = longest_line_ku80_interp.length

        # Calculate the length of the part of the ku80 line outside the nucleus
        outside_length = sum(
            LineString([ku80_coords_interp[i], ku80_coords_interp[i + 1]]).length
            for i in range(len(ku80_coords_interp) - 1)
            if not nuc_polygon.contains(Point(ku80_coords_interp[i])) and not nuc_polygon.contains(Point(ku80_coords_interp[i + 1]))
        )

        # Calculate mse_diff per unit length outside the nucleus
        if outside_length > 0:
            mse_diff_normalized = mse_diff / outside_length
        else:
            mse_diff_normalized = np.nan
            print(f"Index {n}: No points outside nucleus or zero-length line. Normalized mse_diff set to NaN.")

        # Store the normalized mse_diff in the DataFrame
        df2.loc[n, 'mse_diff_normalized'] = mse_diff_normalized


        # Store the computed metrics in the DataFrame
        df2.loc[n, 'similarity'] = similarity
        df2.loc[n, 'count'] = count
        df2.loc[n, 'mse'] = mse
        df2.loc[n, 'mae'] = mae
        df2.loc[n, 'similarity_diff'] = similarity_diff
        df2.loc[n, 'count_diff'] = count_diff
        df2.loc[n, 'mse_diff'] = mse_diff
        df2.loc[n, 'mae_diff'] = mae_diff

        print(f"Processed index {n}. Metrics updated.")
    df2['sample_name'] = df2['image_name'].str.extract(r'(Exp\d+)')
    return df2



def create_combined_cell_nuc_image(
    cell_mask, nuc_mask,
    distance,          # original centroid distance
    angle_deg,         # original angle (deg)
    cell_scale_factor, # stored from df5 (if needed)
    nuc_scale_factor,  # stored from df5 (if needed)
    cell_area_orig=None, 
    nuc_area_orig=None,
    preserve_area_ratio=False,
    cell_val=1.0,
    nuc_val=0.5
):
    """
    Combine cell_mask (128×128) and nuc_mask (128×128) into one image.
    
    Steps:
    1) Optionally adjust nucleus to preserve area ratio => get nucleus in 128×128.
    2) Place cell_mask in center (already 128×128).
    3) Compute offset (distance * cell_scale_factor) in *image* coordinates 
       => invert the sign of offset_y because image coords go down as y increases.
    4) Shift nucleus by offset => shift_mask => still 128×128.
    5) Overlay => cell=1.0, nucleus=0.5.
    """
    # Make a float copy for final
    combined = np.zeros((128, 128), dtype=np.float32)
    
    # (1) Put the cell in
    combined[cell_mask > 0] = cell_val
    
    # (2) Possibly preserve area ratio
    if preserve_area_ratio and (cell_area_orig is not None) and (nuc_area_orig is not None):
        nuc_mask_128 = adjust_nucleus_area(cell_mask, nuc_mask, cell_area_orig, nuc_area_orig)
    else:
        # Just embed the bounding-box–scaled nucleus in 128×128
        nuc_mask_128 = embed_in_128x128(nuc_mask)
    
    # (3) Compute offset in *image* pixel space
    #     We invert offset_y because in image coords, +y is downward
    distance_scaled = distance * cell_scale_factor
    
    angle_rad = math.radians(angle_deg)
    offset_x =  distance_scaled * math.cos(angle_rad)
    offset_y = distance_scaled * math.sin(angle_rad)  # <-- negative sign
    
    shift_x = int(round(offset_x))
    shift_y = int(round(offset_y))
    
    # (4) Shift the nucleus in the 128×128 space
    nuc_mask_shifted = shift_mask(nuc_mask_128, shift_y, shift_x)
    
    # (5) Overlay nucleus => 0.5
    combined[nuc_mask_shifted > 0] = nuc_val
    
    return combined


def prep_binary_images(df2,polygons_dic_scaled, rupt_min = 0.003, non_rupt_max = 0.002,  desired_dim = 62):

        ### TEST SHOWING full, KU80, DAPI, profile
    df3 = df2.copy()
    ind = df3.index.tolist()
    # Rupture detection
    df3.loc[df3['mse_diff'] > rupt_min, 'ruptured_man'] = True
    df3.loc[df3['mse_diff'] < non_rupt_max, 'ruptured_man'] = False
    nuc_polys = [polygons_dic_scaled.get((i, 'nuc_not_scaled'), []) for i in ind]
    cell_polys = [polygons_dic_scaled.get((i, 'cell'), []) for i in ind]

    df5 = df3.copy().reset_index()
    for counter, (cell_poly, nuc_poly) in enumerate(zip(cell_polys, nuc_polys)):
    # for counter, (cell_poly, nuc_poly) in enumerate(zip(rotated_cells, rotated_nucs)):
        if cell_poly and nuc_poly:
            # Create shapely polygons
            cell_polygon = Polygon(cell_poly)
            nuc_polygon  = Polygon(nuc_poly)
            # Compute centroids
            cell_centroid = cell_polygon.centroid
            nuc_centroid  = nuc_polygon.centroid

            # Calculate Euclidean distance
            distance = cell_centroid.distance(nuc_centroid)
            df5.loc[counter, 'distance'] = distance

            # Calculate angle between centroids
            delta_x = nuc_centroid.x - cell_centroid.x
            delta_y = nuc_centroid.y - cell_centroid.y
            angle_rad = np.arctan2(delta_y, delta_x)
            angle_deg = np.degrees(angle_rad)
            df5.loc[counter, 'angle_dist_rec'] = angle_deg
            df5.loc[counter, 'angle_dist']     = abs(angle_corrector(angle_deg))

            # Calculate longest axis angles
            df5.loc[counter, 'cell_longest_angle'] = angle_of_longest_axis(cell_polygon)
            df5.loc[counter, 'nuc_longest_angle']  = angle_of_longest_axis(nuc_polygon)
            df5.loc[counter, 'cell-nuc_longest_angle']  = angle_corrector(angle_of_longest_axis(cell_polygon)- angle_of_longest_axis(nuc_polygon))

            # ----------------------------------------------------------
            # ADD: Compute bounding-box–based scale factors for each poly
            # ----------------------------------------------------------
            # Convert to (x,y) coords
            cell_coords = np.array(cell_polygon.exterior.coords, dtype=np.float32)
            nuc_coords  = np.array(nuc_polygon.exterior.coords,  dtype=np.float32)

            # bounding-box scale factors
            cell_scale_factor = get_bbox_scale_factor(cell_coords, target_longest_dim=desired_dim)
            nuc_scale_factor  = get_bbox_scale_factor(nuc_coords,  target_longest_dim=desired_dim)

            # store them
            df5.loc[counter, 'cell_scale_factor'] = cell_scale_factor
            df5.loc[counter, 'nuc_scale_factor']  = nuc_scale_factor
            
    df5['rel_area' ] = df5['nuc_area']/df5['cell_area']


    cell_binary_images = []
    nuc_binary_images  = []

    for coords in cell_polys:
    # for coords in rotated_cells:
        coords_np = np.array(Polygon(coords).exterior.coords, dtype=np.float32)
        mask_128 = create_bounding_box_scaled_mask(coords_np,
                                                   target_longest_dim=desired_dim,
                                                   final_size=(128, 128))
        cell_binary_images.append(mask_128)

    # for coords in rotated_nucs:
    for coords in nuc_polys:
        coords_np = np.array(Polygon(coords).exterior.coords, dtype=np.float32)
        mask_128 = create_bounding_box_scaled_mask(coords_np,
                                                   target_longest_dim=desired_dim,
                                                   final_size=(128, 128))
        nuc_binary_images.append(mask_128)

    cell_binary_images = np.array(cell_binary_images)
    nuc_binary_images  = np.array(nuc_binary_images)
    nuc_binary_images_reshaped = np.expand_dims(nuc_binary_images, axis=-1)
    cell_binary_images_reshaped = np.expand_dims(cell_binary_images, axis=-1)

    
    return df5, cell_binary_images_reshaped,nuc_binary_images_reshaped





