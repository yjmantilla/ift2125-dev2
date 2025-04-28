from solid import *
from solid.utils import *

import numpy as np
import os

# Parameters
GRID_SIZE = 50  # How wide the ocean plate is (50mm to 80mm)
HEIGHT_LIMIT = 40  # Max height in mm
MATRIX_SIZE = 50  # Height map resolution (30-100)

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

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    # Generate random gradient vectors
    angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=2)

    # Generate coordinate grid
    xs = np.linspace(0, res[0], shape[0], endpoint=False)
    ys = np.linspace(0, res[1], shape[1], endpoint=False)
    x, y = np.meshgrid(xs, ys, indexing='ij')

    noise = perlin(x, y, gradients)

    # Normalize to [0,1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise

#####################################################################

# Biharmonic Spline interpolation

def mergesimpts(data,tols,mode='average'):
    data_ = data.copy()[np.argsort(data[:,0])]
    newdata = []
    tols_ = np.array(tols)
    idxs_ready =[]
    point = 0
    for point in range(data_.shape[0]):
        if point in idxs_ready:
            continue
        else:
            similar_pts = np.where(np.prod(np.abs(data_ - data_[point]) < tols_, axis=-1))
            similar_pts = np.array(list(set(similar_pts[0].tolist())- set(idxs_ready)))
            idxs_ready += similar_pts.tolist()
            if mode == 'average':
                exemplar = np.mean(data_[similar_pts],axis=0)
            else:
                exemplar = data_[similar_pts].copy()[0] # first
            newdata.append(exemplar)
    return np.array(newdata)

def mergepoints2D(x,y,v):
    # Sort x and y so duplicate points can be averaged

    # Need x,y and z to be column vectors

    sz = x.size
    x = x.copy()
    y = y.copy()
    v = v.copy()
    x = np.reshape(x,(sz),order='F');
    y = np.reshape(y,(sz),order='F');
    v = np.reshape(v,(sz),order='F');

    myepsx = np.spacing(0.5 * (np.max(x) - np.min(x)))**(1/3);
    myepsy = np.spacing(0.5 * (np.max(y) - np.min(y)))**(1/3);
    # % look for x, y points that are indentical (within a tolerance)
    # % average out the values for these points
    if np.all(np.isreal(v)):
        data = np.stack((y,x,v), axis=-1)
        yxv = mergesimpts(data,[myepsy,myepsx,np.inf],'average')
        x = yxv[:,1]
        y = yxv[:,0]
        v = yxv[:,2]
    else:
        #% if z is imaginary split out the real and imaginary parts
        data = np.stack((y,x,np.real(v),np.imag(v)), axis=-1)
        yxv = mergesimpts(data,[myepsy,myepsx,np.inf,np.inf],'average')
        x = yxv[:,1]
        y = yxv[:,0]
        #% re-combine the real and imaginary parts
        v = yxv[:,2]+1j*yxv[:,3]
    #% give a warning if some of the points were duplicates (and averaged out)
    if sz > x.shape[0]:
        print('MATLAB:griddata:DuplicateDataPoints')
    return x,y,v

def gdatav4(x,y,v,xq,yq):
    """
    %GDATAV4 MATLAB 4 GRIDDATA interpolation

    %   Reference:  David T. Sandwell, Biharmonic spline
    %   interpolation of GEOS-3 and SEASAT altimeter
    %   data, Geophysical Research Letters, 2, 139-142,
    %   1987.  Describes interpolation using value or
    %   gradient of value in any dimension.
    """
    x, y, v = mergepoints2D(x,y,v);

    xy = x + 1j*y
    xy = np.squeeze(xy)
    #% Determine distances between points
    
    # d = np.zeros((xy.shape[0],xy.shape[0]))
    # for i in range(xy.shape[0]):
    #     for j in range(xy.shape[0]):
    #         d[i,j]=np.abs(xy[i]-xy[j])

    d = np.abs(np.subtract.outer(xy, xy))
    # % Determine weights for interpolation
    g = np.square(d) * (np.log(d)-1) #% Green's function.
    # % Fixup value of Green's function along diagonal
    np.fill_diagonal(g, 0)
    weights = np.linalg.lstsq(g, v)[0]

    (m,n) = xq.shape
    vq = np.zeros(xq.shape);
    #xy = np.tranpose(xy);

    # % Evaluate at requested points (xq,yq).  Loop to save memory.
    for i in range(m):
        for j in range(n):
            d = np.abs(xq[i,j] + 1j*yq[i,j] - xy);
            g = np.square(d) * (np.log(d)-1);#   % Green's function.
            #% Value of Green's function at zero
            g[np.where(np.isclose(d,0))] = 0;
            vq[i,j] = (np.expand_dims(g,axis=0) @ np.expand_dims(weights,axis=1))[0][0]
    return xq,yq,vq


#####################################################################

def create_ocean_base(size_mm):
    """Creates the ocean base plate."""
    thickness = 2  # thickness of the ocean plate
    ocean = cube([size_mm, size_mm, thickness])
    return color([0, 0, 1])(ocean)  # RGB: blue

def generate_model(seed=42):

    rng = np.random.default_rng(seed)

    # Start with the ocean
    ocean = create_ocean_base(GRID_SIZE)

    # Placeholder for island (we'll add next)
    island = up(2)(sphere(r=10))  # just for testing, a simple island

    # Combine everything
    model = union()(
        ocean,
    )
    return model

if __name__ == '__main__':
    scad_render_to_file(generate_model(), filepath='model.scad', file_header='$fn = 100;')
