from solid import *
from solid.utils import *

import numpy as np
import os

# Parameters
GRID_SIZE = 80  # How wide the ocean plate is (50mm to 80mm)
HEIGHT_LIMIT = 40  # Max height in mm
MATRIX_SIZE = 50  # Height map resolution (30-100)


def generate_initials(initials):
    txt = linear_extrude(height=0.5)(text(initials, size=5))
    mirrored_txt = mirror([0,1,0])(txt)  # Mirror across Y axis to correct
    return translate([3, 6, 0])(mirrored_txt)

###### Gaussian Filter ###############################

import numpy as np

def convolve2d(image, kernel):
    """
    Perform a 2D convolution manually using numpy.
    
    Parameters:
    - image: 2D numpy array (input)
    - kernel: 2D numpy array (filter)
    
    Returns:
    - convolved 2D numpy array
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # Flip the kernel (convolution flips the filter)
    kernel = np.flipud(np.fliplr(kernel))

    # Pad the image with zeros so output size matches input
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')  # 'reflect' is nice for boundaries

    # Output array
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)

    return output


def gaussian_kernel(size, sigma):
    """
    Generates a (size x size) Gaussian kernel with standard deviation sigma.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

def gaussian_filter(image, sigma, truncate=4.0):
    """
    Apply a gaussian filter manually using numpy.

    Parameters:
    - image: 2D numpy array
    - sigma: standard deviation of Gaussian
    - truncate: how many sigmas to include (default 4.0)
    """

    # Kernel size is 2*truncate*sigma rounded up to nearest odd integer
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1

    kernel = gaussian_kernel(size, sigma)

    # Apply convolution
    smoothed = convolve2d(image, kernel)

    return smoothed


# Perlin noise #######################################
def fade(t):
    # Fade function as defined by Ken Perlin
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp(a, b, t):
    # Linear interpolation
    return a + t * (b - a)

def perlin(x, y, gradients):
    # x, y in grid cell coordinates
    # gradients: (gx, gy) array at each grid point

    # Grid coordinates
    x0 = x.astype(int)
    x1 = x0 + 1
    y0 = y.astype(int)
    y1 = y0 + 1

    # Relative x, y inside the grid cell
    sx = x - x0
    sy = y - y0

    # Dot products between gradient and distance vectors
    n00 = (gradients[x0, y0, 0] * (x - x0) +
        gradients[x0, y0, 1] * (y - y0))
    n10 = (gradients[x1, y0, 0] * (x - x1) +
        gradients[x1, y0, 1] * (y - y0))
    n01 = (gradients[x0, y1, 0] * (x - x0) +
        gradients[x0, y1, 1] * (y - y1))
    n11 = (gradients[x1, y1, 0] * (x - x1) +
        gradients[x1, y1, 1] * (y - y1))

    # Fade curves for x and y
    u = fade(sx)
    v = fade(sy)

    # Interpolate
    nx0 = lerp(n00, n10, u)
    nx1 = lerp(n01, n11, u)
    nxy = lerp(nx0, nx1, v)

    return nxy

def generate_perlin_noise_2d(shape, res, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Ensure res is integer
    res_x = int(res[0])
    res_y = int(res[1])

    # Generate random gradient vectors
    angles = 2 * np.pi * np.random.rand(res_x+1, res_y+1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=2)

    # Generate coordinate grid
    xs = np.linspace(0, res_x, shape[0], endpoint=False)
    ys = np.linspace(0, res_y, shape[1], endpoint=False)
    x, y = np.meshgrid(xs, ys, indexing='ij')

    noise = perlin(x, y, gradients)

    # Normalize to [0,1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise

#####################################################################

# Biharmonic Spline interpolation

def mergesimpts(data, tols, mode='average'):
    data_ = data.copy()[np.argsort(data[:,0])]
    newdata = []
    tols_ = np.array(tols)
    idxs_ready = []
    point = 0
    for point in range(data_.shape[0]):
        if point in idxs_ready:
            continue
        else:
            similar_pts = np.where(np.prod(np.abs(data_ - data_[point]) < tols_, axis=-1))
            similar_pts = np.array(list(set(similar_pts[0].tolist())- set(idxs_ready)))
            idxs_ready += similar_pts.tolist()
            if mode == 'average':
                exemplar = np.mean(data_[similar_pts], axis=0)
            else:
                exemplar = data_[similar_pts].copy()[0]  # first
            newdata.append(exemplar)
    return np.array(newdata)

def mergepoints2D(x, y, v):
    # Sort x and y so duplicate points can be averaged

    # Need x,y and z to be column vectors
    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x, (sz), order='F')
    y = np.reshape(y, (sz), order='F')
    v = np.reshape(v, (sz), order='F')

    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x)))**(1/3)
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y)))**(1/3)
    
    # Look for x, y points that are identical (within a tolerance)
    # Average out the values for these points
    if np.all(np.isreal(v)):
        data = np.stack((y, x, v), axis=-1)
        yxv = mergesimpts(data, [myepsy, myepsx, np.inf], 'average')
        x = yxv[:, 1]
        y = yxv[:, 0]
        v = yxv[:, 2]
    else:
        # If z is imaginary split out the real and imaginary parts
        data = np.stack((y, x, np.real(v), np.imag(v)), axis=-1)
        yxv = mergesimpts(data, [myepsy, myepsx, np.inf, np.inf], 'average')
        x = yxv[:, 1]
        y = yxv[:, 0]
        # Re-combine the real and imaginary parts
        v = yxv[:, 2] + 1j * yxv[:, 3]
    
    # Give a warning if some of the points were duplicates (and averaged out)
    if sz > x.shape[0]:
        print('Warning: Duplicate data points detected and averaged')
    
    return x, y, v

def gdatav4(x, y, v, xq, yq):
    """
    GDATAV4 MATLAB 4 GRIDDATA interpolation

    Reference: David T. Sandwell, Biharmonic spline
    interpolation of GEOS-3 and SEASAT altimeter
    data, Geophysical Research Letters, 2, 139-142,
    1987. Describes interpolation using value or
    gradient of value in any dimension.
    """
    x, y, v = mergepoints2D(x, y, v)

    xy = x + 1j * y
    xy = np.squeeze(xy)
    
    # Determine distances between points
    d = np.abs(np.subtract.outer(xy, xy))
    
    # Determine weights for interpolation
    # Fix log(0) issue by adding small epsilon to distances
    epsilon = 1e-10
    d_safe = np.maximum(d, epsilon)
    g = np.square(d_safe) * (np.log(d_safe) - 1)  # Green's function
    
    # Fix value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    
    # Use a more stable solver with explicit regularization
    weights = np.linalg.lstsq(g, v, rcond=1e-10)[0]

    (m, n) = xq.shape
    vq = np.zeros(xq.shape)

    # Evaluate at requested points (xq, yq)
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i, j] + 1j * yq[i, j] - xy)
            # Fix log(0) issue
            d_safe = np.maximum(d, epsilon)
            g = np.square(d_safe) * (np.log(d_safe) - 1)  # Green's function
            # Value of Green's function at zero
            g[np.where(np.isclose(d, 0))] = 0
            vq[i, j] = (np.expand_dims(g, axis=0) @ np.expand_dims(weights, axis=1))[0][0]
    
    return xq, yq, vq

#####################################################################

def create_paysage_from_heightmap(heightmap, waterfall_mask, river_mask,grid_size, height_limit, text_mask=None):
    """Convert height map to OpenSCAD polyhedron."""
    terrain_parts = []
    
    # Get dimensions of the heightmap
    rows, cols = heightmap.shape
    
    # Calculate cell size
    cell_size = grid_size / (cols - 1)
    
    waterfall_depth = 0.5  # How deep to carve the waterfall
    #make everything not empty at least 0.1mm high
    ocean_level = 0.02  # Ocean level (height of the ocean base)
    min_level = 1.2*ocean_level  # Minimum height for the terrain (to avoid floating parts)
    not_empty_indexes = heightmap > 0
    heightmap[not_empty_indexes] = np.maximum(heightmap[not_empty_indexes], min_level)  # Set minimum height to ocean level


    # Create polygons for each cell in the height map
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Calculate the coordinates for this cell
            x1 = j * cell_size
            y1 = i * cell_size
            
            z1 = heightmap[i, j] * height_limit

            # Check if this is part of the waterfall
            is_waterfall = waterfall_mask[i, j] > 0
            is_river = river_mask[i, j] > 0

            # Alternative. Check if this is part of the waterfall
            #is_waterfall = waterfall_mask[i, j] > 0.5 and z1 > 0  # Only consider as waterfall if above water level

            # Check if this cell is masked by text
            is_text = text_mask is not None and text_mask[i, j] > 0.5

            if z1 <= 0:
                # Fill underwater regions with ocean cubes
                terrain_parts.append(
                    translate([x1, y1, 0])(
                        color([0, 0.5, 0.8])(  # Same blue as ocean
                            cube([cell_size, cell_size, ocean_level*height_limit])
                        )
                    )
                )
            elif is_text:
                # Cells where text mask is active become yellow
                terrain_parts.append(
                    translate([x1, y1, 0])(
                        color([1.0, 1.0, 0.0])(  # Yellow
                            cube([cell_size, cell_size, max(ocean_level*height_limit, z1)])
                        )
                    )
                )
            elif is_waterfall:
                # For waterfall parts, create a blue cube slightly above the terrain
                waterfall_height = max(max(0.1, z1 - waterfall_depth),ocean_level)  # waterfall is slightly below terrain but above ocean level
                
                terrain_parts.append(
                    translate([x1, y1, 0])(
                        color([0, 0.4, 0.8])(  # Blue for waterfall
                            cube([cell_size, cell_size, waterfall_height])
                        )
                    )
                )
            elif is_river:
                # For river parts, create a blue cube slightly above the terrain
                river_height = max(max(0.1, z1 - waterfall_depth),ocean_level)
                terrain_parts.append(
                    translate([x1, y1, 0])(
                        color([0, 0.4, 0.8])(  # Blue for river
                            cube([cell_size, cell_size, river_height])
                        )
                    )
                )
            else:
                # Regular terrain (non-waterfall)
                # Color based on height (green to brown to white)
                h_ratio = z1 / height_limit
                if h_ratio < 0.3:  # Low elevation: greenish
                    color_val = [0.2, 0.6, 0.2]
                elif h_ratio < 0.7:  # Medium elevation: brownish
                    color_val = [0.6, 0.4, 0.2]
                else:  # High elevation: white/gray (snow)
                    color_val = [0.8, 0.8, 0.8]
                
                terrain_parts.append(
                    translate([x1, y1, 0])(
                        color(color_val)(
                            cube([cell_size, cell_size, max(ocean_level*height_limit, z1)])
                        )
                    )
                )
    
    # Combine terrain and waterfall parts
    all_parts = terrain_parts
    return union()(*all_parts)


def generate_control_points(num_points, grid_size, rng,kind='central_island'):
    """Generate random control points for the terrain."""
    # Generate points more likely to be in the center
    x = rng.normal(loc=grid_size/2, scale=grid_size/4, size=num_points)
    y = rng.normal(loc=grid_size/2, scale=grid_size/4, size=num_points)
    
    # Clip to ensure within bounds
    x = np.clip(x, 0, grid_size)
    y = np.clip(y, 0, grid_size)
    
    # This is the most important part of the generation, as it defines the height of the terrain
    # Generate height values (higher in center)
    v = np.zeros_like(x)
    if kind == 'central_island':
        for i in range(num_points):
            # Distance from center
            dist = np.sqrt((x[i] - grid_size/2)**2 + (y[i] - grid_size/2)**2)
            dist_ratio = dist / (grid_size/2)
            
            # Points closer to center have higher probability of being high
            if dist_ratio < 0.5 and rng.random() < 0.8:
                # Central island points
                v[i] = rng.uniform(0.5, 1.0)
            elif dist_ratio < 0.7 and rng.random() < 0.4:
                # Medium distance points
                v[i] = rng.uniform(0.2, 0.6)
            else:
                # Outer points are mostly underwater
                v[i] = rng.uniform(-0.2, 0.1)
    if kind == 'decentralized':
        num_islands = 10 # More flexible number of islands

        centers = []
        attempts = 0
        max_attempts = 1000  # Safety to avoid infinite loops

        min_dist = grid_size / (num_islands)  # Minimum distance between islands

        while len(centers) < num_islands and attempts < max_attempts:
            candidate_x = rng.uniform(0.1 * grid_size, 0.9 * grid_size)
            candidate_y = rng.uniform(0.1 * grid_size, 0.9 * grid_size)
            candidate = (candidate_x, candidate_y)

            # Check if candidate is far enough from all previous centers
            if all(np.linalg.norm(np.array(candidate) - np.array(c)) >= min_dist for c in centers):
                centers.append(candidate)

            attempts += 1
        
        # just randomize the centers

        # for i in range(num_islands):
        #     center_x = rng.uniform(0.1 * grid_size, 0.9 * grid_size)
        #     center_y = rng.uniform(0.1 * grid_size, 0.9 * grid_size)
        #     centers.append((center_x, center_y))

        size_islands = 0.8 #rng.uniform(0.5, 0.8)  # Size of islands relative to grid size
        for i in range(num_islands):
            center_x, center_y = centers[i]
            # # Randomize the center slightly
            # center_x += rng.uniform(-size_islands*grid_size/4, size_islands*grid_size/4)
            # center_y += rng.uniform(-size_islands*grid_size/4, size_islands*grid_size/4)
            
            # Random height for the island
            center_height = rng.uniform(0.6, 0.8)
            # Apply same logic as above but centered around the random point
            for j in range(num_points):
                dist = np.sqrt((x[j] - center_x)**2 + (y[j] - center_y)**2)
                dist_ratio = dist / (size_islands*grid_size)
                if dist_ratio < 0.5 and rng.random() < 0.8:
                    # Central island points
                    v[j] = center_height * rng.uniform(0.5, 0.7)
                elif dist_ratio < 0.7 and rng.random() < 0.4:
                    v[j] = center_height * rng.uniform(0.2, 0.4)
                else:
                    # Outer points are mostly underwater
                    v[j] = rng.uniform(-0.2, 0)
    return x, y, v

def add_perlin_detail(height_map, octaves=3, persistence=0.5, scale=2, seed=None):
    """Add Perlin noise detail to the height map."""
    shape = height_map.shape
    
    # Generate base noise - ensure scale is integer
    scale_int = int(scale)
    noise = generate_perlin_noise_2d(shape, (scale_int, scale_int), seed)
    
    # Add octaves of noise
    for i in range(1, octaves):
        weight = persistence ** i
        scale_octave = int(scale_int * (2**i))
        noise_octave = generate_perlin_noise_2d(shape, (scale_octave, scale_octave), seed)
        noise = noise + weight * noise_octave
    
    # Normalize noise
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Mix with the original height map (more effect on higher elevations)
    detail_factor = 0.2  # Amount of detail to add
    
    # Apply detail with more effect on land than water
    mask = height_map > 0  # Land mask
    result = height_map.copy()
    
    # Add detail to land
    result[mask] = height_map[mask] * (1.0 + detail_factor * (noise[mask] - 0.5))
    
    # Add minimal detail to underwater areas
    result[~mask] = height_map[~mask] + 0.02 * (noise[~mask] - 0.5)
    
    # Ensure values stay in reasonable range
    result = np.clip(result, -0.2, 1.0)
    
    return result

def generate_waterfall_path(height_map):
    """Generate a waterfall path from the highest point to sea level using gradient descent."""
    start_i, start_j = np.unravel_index(np.argmax(height_map), height_map.shape)
    
    rows, cols = height_map.shape
    waterfall_mask = np.zeros_like(height_map)
    waterfall_width = 1
    current_i, current_j = start_i, start_j
    waterfall_mask[current_i, current_j] = 1
    max_steps = rows * cols
    step_count = 0
    
    while height_map[current_i, current_j] > 0 and step_count < max_steps:
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = current_i + di, current_j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbors.append((ni, nj, height_map[ni, nj]))
        
        valid_neighbors = [(ni, nj, h) for ni, nj, h in neighbors if h < height_map[current_i, current_j]]
        
        if not valid_neighbors:
            if neighbors:
                next_i, next_j, _ = min(neighbors, key=lambda x: x[2])
                height_map[next_i, next_j] = height_map[current_i, current_j] - 0.01
            else:
                break
        else:
            next_i, next_j, _ = min(valid_neighbors, key=lambda x: x[2])
        
        current_i, current_j = next_i, next_j
        
        for di in range(-waterfall_width, waterfall_width + 1):
            for dj in range(-waterfall_width, waterfall_width + 1):
                ri, rj = current_i + di, current_j + dj
                if 0 <= ri < rows and 0 <= rj < cols:
                    waterfall_mask[ri, rj] = 1
        
        step_count += 1
    
    waterfall_mask = gaussian_filter(waterfall_mask, sigma=0.3)
    waterfall_mask = (waterfall_mask > 0.3).astype(float)
    
    return waterfall_mask

def generate_river_path(height_map):
    # use just a random spline between two edges


    # Similar to generate_waterfall_path but for rivers
    # we start from one edge to any other edge
    # using a random walk
    # and then we smooth the path with a gaussian filter
    # and then we create a mask for the river

    # Start from a random edge
    rows, cols = height_map.shape
    start_edge = np.random.choice(['top', 'bottom', 'left', 'right'])
    final_options = ['top', 'bottom', 'left', 'right']
    final_options.remove(start_edge)
    final_edge = np.random.choice(final_options)

    # Randomly choose a starting point on the start edge and an end point on the final edge
    start_point = np.random.uniform(0, 1)
    end_point = np.random.uniform(0, 1)

    # Calculate the coordinates of the starting point based on the edge
    coordinates_start = start_point * (rows if start_edge in ['top', 'bottom'] else cols)
    coordinates_end = end_point * (rows if final_edge in ['top', 'bottom'] else cols)

    if start_edge == 'top' or start_edge == 'bottom':
        start_i = int(coordinates_start)
        start_j = rows - 1 if start_edge == 'bottom' else 0
        #np.random.randint(0, cols)
    elif start_edge == 'left' or start_edge == 'right':
        start_i = int(coordinates_start)
        start_j = cols - 1 if start_edge == 'right' else 0
        #np.random.randint(0, rows)
    
    #make a random spline between the start and end points
    # and then we smooth it with a gaussian filter
    # and then we create a mask for the river

    random_spline = np.zeros((rows, cols))



    if start_edge == 'top':
        start_i = 0
        start_j = np.random.randint(0, cols)
    elif start_edge == 'bottom':
        start_i = rows - 1
        start_j = np.random.randint(0, cols)
    elif start_edge == 'left':
        start_i = np.random.randint(0, rows)
        start_j = 0
    else:  # 'right'
        start_i = np.random.randint(0, rows)
        start_j = cols - 1

    # Initialize the river path
    river_mask = np.zeros_like(height_map)
    river_width = 2
    current_i, current_j = start_i, start_j
    river_mask[current_i, current_j] = 1
    
    reach_edge = False

    max_steps = rows * cols

    step_count = 0

    
    j_biased_choices = [-1, 0, 1]
    i_biased_choices = [-1, 0, 1]
    if start_edge == 'top':
        j_biased_choices.remove(-1)
    elif start_edge == 'bottom':
        j_biased_choices.remove(1)
    elif start_edge == 'left':
        i_biased_choices.remove(-1)
    else:  # 'right'
        i_biased_choices.remove(1)
    while not reach_edge and step_count < max_steps:
        # Random walk biased towards the final edge
        current_i = current_i + np.random.choice(i_biased_choices)
        current_j = current_j + np.random.choice(j_biased_choices)
        # check bounds
        if current_i < 0 or current_i >= rows or current_j < 0 or current_j >= cols:
            # If out of bounds, revert to previous position
            current_i = np.clip(current_i, 0, rows - 1)
            current_j = np.clip(current_j, 0, cols - 1)
        river_mask[current_i, current_j] = 1
        # Check if we reached the final
        if final_edge == 'top' and current_i == 0:
            reach_edge = True
        elif final_edge == 'bottom' and current_i == rows - 1:
            reach_edge = True
        elif final_edge == 'left' and current_j == 0:
            reach_edge = True
        elif final_edge == 'right' and current_j == cols - 1:
            reach_edge = True
    return river_mask
def generate_model(seed=42,num_control_points=60,kind='central_island'):
    rng = np.random.default_rng(seed)
    
    # Grid for interpolation and final height map
    xs = np.linspace(0, GRID_SIZE, MATRIX_SIZE)
    ys = np.linspace(0, GRID_SIZE, MATRIX_SIZE)
    xq, yq = np.meshgrid(xs, ys)
    
    # Generate control points for the terrain
    #num_control_points = 15  # Adjust for more/less detail
    # nice ones with 15 with seed 137
    # 60 with seed 42


    x, y, v = generate_control_points(num_control_points, GRID_SIZE, rng, kind=kind)
    
    # Generate initial height map using biharmonic spline interpolation
    _, _, height_map = gdatav4(x, y, v, xq, yq)
    
    # Add noise detail to the height map
    height_map = add_perlin_detail(height_map, seed=seed)
    
    # Normalize height map
    if kind == 'central_island':
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
        river_mask = np.zeros_like(height_map)  # No river for central island
    else:
        river_mask = generate_river_path(height_map)

        
    # Generate waterfall path
    waterfall_mask = generate_waterfall_path(height_map)
    
    # Create the terrain model with waterfall
    paysage = create_paysage_from_heightmap(height_map, waterfall_mask,river_mask, GRID_SIZE, HEIGHT_LIMIT)
    
    
    # differences are really slow... and unions create kind of a shading issue in openscad, i guess because
    # of the many many faces created by the union of all the cubes
    # so we generate a single solid based on heightmap and masks

    # Combine everything
    model = paysage

    
    return model

if __name__ == '__main__':
    if True:
#    try:
        params=[
            (42, 60, 'central_island'),
            (137, 15, 'central_island'),
            (42, 60, 'decentralized'),
            (137, 15, 'decentralized')
        ]
        for seed, num_control_points, kind in params:
            # Generate the model with the specified parameters
            model = generate_model(seed=seed, num_control_points=num_control_points, kind=kind)
            
            # Save the model to a file
            filename = f'paysage_{seed}_{num_control_points}_{kind}.scad'
            scad_render_to_file(model, filepath=filename, file_header='$fn = 100;')
            print(f"Model generated successfully: {filename}")

        # True random
        rng=np.random.default_rng()
        seed = rng.integers(0, 1000000)
        # choose between kinds

        kinds = ['decentralized']
        kind = rng.choice(kinds)
        num_control_points = rng.integers(10, 100)

        # save params for reproducibility
        with open('paysage_random.txt', 'w') as f:
            f.write(f"seed: {seed}\n")
            f.write(f"num_control_points: {num_control_points}\n")
            f.write(f"kind: {kind}\n")
        scad_render_to_file(generate_model(seed=seed), filepath='paysage_random.scad', file_header='$fn = 100;')
        print("Model generated successfully!")
#    except Exception as e:
    else:
        print(f"Error generating model: {str(e)}")