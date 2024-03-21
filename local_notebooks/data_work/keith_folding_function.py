import numpy as np
import xarray as xr

def fold_image(data_array, fold_axis):
    """
    Method to fold image along a specified axis.
    
    Parameters:
    - data_array (xarray DataArray): The DataArray to fold
    - fold_axis (str): The axis along which to fold the image
    
    Returns:
    - xarray DataArray: The folded image
    """
    # Filter data for fold_axis >= 0 and fold_axis <= 0
    positive_data = data_array.where(data_array[fold_axis] >= 0, drop=True)
    negative_data = data_array.where(data_array[fold_axis] <= 0, drop=True)
    
    # Reverse negative_data for easier comparison
    negative_data = negative_data.reindex({fold_axis: negative_data[fold_axis][::-1]})
    
    # Find the maximum coordinate of the shorter quadrant (positive_data)
    max_positive_coord = float(positive_data[fold_axis].max())
    
    # Find the equivalent coordinate in the negative_data
    abs_diff = np.abs(negative_data[fold_axis].values + max_positive_coord)
    
    # Minimize the difference
    min_diff_idx = np.argmin(abs_diff)
    
    # Check if the lengths are equivalent
    len_pos = len(positive_data[fold_axis])
    len_neg = len(negative_data[fold_axis][:min_diff_idx+1])
    
    if len_pos != len_neg:
        # Adjust the coordinate range for negative_data
        for i in range(1, 4):  # Check 3 neighbors
            new_idx = min_diff_idx + i
            len_neg = len(negative_data[fold_axis][:new_idx+1])
            if len_pos == len_neg:
                min_diff_idx = new_idx
                break
                
    # Crop the negative_data to match positive_data length
    negative_data_cropped = negative_data.isel({fold_axis: slice(0, min_diff_idx+1)})
    
    # Prepare the new data array
    new_data = xr.zeros_like(positive_data)
    
    # Fold the image
    for i in range(len(positive_data[fold_axis])):
        pos_val = positive_data.isel({fold_axis: i}).values
        neg_val = negative_data_cropped.isel({fold_axis: i}).values
        
        # Pixel comparison and averaging or summing
        new_data.values[i] = np.where(
            (pos_val > 0) & (neg_val > 0), (pos_val + neg_val) / 2,
            np.where((pos_val == 0) & (neg_val > 0), neg_val, pos_val)
        )
        
    # Append residual data from the longer quadrant if exists
    if len(negative_data[fold_axis]) > min_diff_idx+1:
        residual_data = negative_data.isel({fold_axis: slice(min_diff_idx+1, None)})
        residual_data[fold_axis] = np.abs(residual_data[fold_axis])
        new_data = xr.concat([new_data, residual_data], dim=fold_axis)
        
    # Update data_array with the folded image
    data_array = new_data.sortby(fold_axis)
    
    # Inherit coordinates and metadata attributes from the original data_array
    data_array.attrs = data_array.attrs.copy()
    data_array.attrs['fold_axis'] = fold_axis  # Add 'fold_axis' attribute

    # Ensure all original coordinates are retained in the new data_array
    for coord in data_array.coords:
        if coord not in data_array.coords:
            data_array = data_array.assign_coords({coord: data_array[coord]})

    return data_array