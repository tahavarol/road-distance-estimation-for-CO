import pickle
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import CRS, Transformer

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import random

from geopy.distance import geodesic   

from tqdm import tqdm

import requests

import math

from utils import visualize_grid_and_polygon as viz

import osmnx as ox

from shapely.geometry import Polygon, Point, LineString                

from turtle import st
from pathlib import Path, PurePath
import re

import time

#load GMApoly.pkl
with open('dataset/GMApoly.pkl', 'rb') as f:
    GMApoly = pickle.load(f)

def GetExtremePoints(coords):
    
    
    lat_min = np.array(coords)[:,0].min()
    lat_max = np.array(coords)[:,0].max()

    lon_min = np.array(coords)[:,1].min()
    lon_max = np.array(coords)[:,1].max()

    # Get the extreme points
    extreme_points = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max
    }

    return lat_min, lon_min, lat_max, lon_max



def grid_corners(extremes, divisions):
    """
    Split a 2-D rectangle into a regular grid and return each cell’s corner points.

    extremes   : (xmin, ymin, xmax, ymax)
    divisions  : (nx, ny)  – number of columns (x) and rows (y)

    Returns
    -------
    dict { (ix, iy): ((x0, y0), (x1, y0), (x1, y1), (x0, y1)) }
    """
    ymin, xmin, ymax, xmax = extremes
    
    nx, ny = divisions
    if nx <= 0 or ny <= 0:
        raise ValueError("divisions must be positive integers")

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    cells = {}
    for iy in range(ny):          # rows (south → north)
        for ix in range(nx):      # columns (west → east)
            x0 = xmin + ix * dx
            y0 = ymin + iy * dy
            x1 = x0 + dx
            y1 = y0 + dy
            cells[(ix, iy)] = ((y0, x0), (y0, x1), (y1, x1), (y1, x0))

    return cells

def compute_cell_centers(grid):
    """
    Compute the center point for each cell in the grid.

    Parameters:
    ----------
    grid : dict
        Dictionary with keys as cell indices and values as a tuple of corners:
        ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    
    Returns:
    -------
    dict
        Dictionary with the same keys as grid, with the center point (x, y).
    """
    centers = {}
    for key, corners in grid.items():
        x_center = (corners[0][1] + corners[2][1]) / 2
        y_center = (corners[0][0] + corners[2][0]) / 2
        centers[key] = (y_center,x_center)
    return centers




def locate_point(point, extremes, divisions):
    """
    Given a point (x, y), return the grid index (ix, iy) it falls into,
    or None if the point is outside the overall bounding box.

    Parameters
    ----------
    point     : tuple[float, float]   (x, y)
    extremes  : same as in grid_corners
    divisions : same as in grid_corners

    Returns
    -------
    (ix, iy)  – zero-based column / row indices, or None.
    """
    y, x = point
    ymin, xmin, ymax, xmax = extremes
    nx, ny = divisions

    # quick reject if outside bounds
    if not (xmin <= x < xmax and ymin <= y < ymax):
        return None

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    ix = int((x - xmin) // dx)
    iy = int((y - ymin) // dy)

    # Guard against rounding artefacts when point == xmax or ymax
    ix = min(ix, nx - 1)
    iy = min(iy, ny - 1)

    return (ix, iy)


def get_padding_neighbors(grid):
    """
    For each cell in the grid (given as the output of grid_corners),
    compute a 3x3 neighborhood (with padding at the edges) using the same keys.
    The neighborhood is a tuple of 9 indices in the following order:
      West, Northwest, North, Northeast, East, Southeast, South, Southwest, center.
      
    For example, for cell (1,1) the result will be:
      ((0,1), (0,2), (1,2), (2,2), (2,1), (2,0), (1,0), (0,0), (1,1))
    
    Parameters
    ----------
    grid : dict
        A dictionary with keys as (ix, iy) cell indices.
    
    Returns
    -------
    dict
        A dictionary with the same keys as grid but with values set to the 3x3 neighborhood indices.
        Out-of-bound indices are padded (clipped) to the grid edge.
    """
    # Determine grid dimensions from keys
    xs = [key[0] for key in grid.keys()]
    ys = [key[1] for key in grid.keys()]
    nx = max(xs) + 1  # number of columns
    ny = max(ys) + 1  # number of rows

    def clip(val, low, high):
        return max(low, min(val, high))
    
    neighbors_dict = {}
    for (i, j) in grid.keys():
        # Compute neighbor indices with padding
        neighbors = []
        neighbors.append((clip(i - 1, 0, nx - 1), j))                # West
        neighbors.append((clip(i - 1, 0, nx - 1), clip(j + 1, 0, ny - 1))) # Northwest
        neighbors.append((i, clip(j + 1, 0, ny - 1)))                  # North
        neighbors.append((clip(i + 1, 0, nx - 1), clip(j + 1, 0, ny - 1))) # Northeast
        neighbors.append((clip(i + 1, 0, nx - 1), j))                  # East
        neighbors.append((clip(i + 1, 0, nx - 1), clip(j - 1, 0, ny - 1))) # Southeast
        neighbors.append((i, clip(j - 1, 0, ny - 1)))                  # South
        neighbors.append((clip(i - 1, 0, nx - 1), clip(j - 1, 0, ny - 1))) # Southwest
        neighbors.append((i, j))                                      # Center
        
        neighbors_dict[(i, j)] = tuple(neighbors)
        
    return neighbors_dict



import numpy as np
R = 6_371_008.8  # mean Earth radius in metres

def fast_haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat  = lat2 - lat1
    dlon  = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))      


def compute_features_for_new_data_2(c1, 
                                  c2,
                                  grid_size,
                                  padded_neighbors, 
                                  centers_dist_mat,
                                  cell_centers,
                                  bounding_box,
                                  center_keys,
                                  grid_building_matchings,
                                  building_centroids,
                                  building_distance_matrix,
                                  ref_streets,
                                  large_features = True):
    """
    Compute features for a pair of coordinates (c1, c2).
    
    Parameters:
    - c1: tuple (lat1, lon1)
    - c2: tuple (lat2, lon2)
    - padded_neighbors: dictionary of padded neighbors
    - centers_dist_mat_100: precomputed distance matrix
    
    Returns:
    - features: list of computed features
    """
    cell1 = locate_point(c1, bounding_box, grid_size)
    cell2 = locate_point(c2, bounding_box, grid_size)


    geodesic_distance = fast_haversine(c1[0], c1[1], c2[0], c2[1])

    c1_pads = padded_neighbors[cell1]
    c2_pads = padded_neighbors[cell2]

    cell_road_distance_driving = centers_dist_mat[center_keys[cell1]][center_keys[cell2]]

    full_padded_road_distances_driving = [centers_dist_mat[center_keys[c1_pads[k]]][center_keys[c2_pads[l]]]
                                  for k in range(len(c1_pads))
                                  for l in range(len(c2_pads))]
    
    
    


    padded_geodesic_c1 = [fast_haversine(c1[0], c1[1], cell_centers[cell][0],cell_centers[cell][1])
                                  for cell in padded_neighbors[cell1]]
    
    padded_geodesic_c2 = [fast_haversine(c2[0], c2[1], cell_centers[cell][0],cell_centers[cell][1])
                                  for cell in padded_neighbors[cell2]]

    c1_pads_coords_lat = [cell_centers[cell][0] for cell in c1_pads]
    c1_pads_coords_lon = [cell_centers[cell][1] for cell in c1_pads]
    c2_pads_coords_lat = [cell_centers[cell][0] for cell in c2_pads]
    c2_pads_coords_lon = [cell_centers[cell][1] for cell in c2_pads]
    
    normalized_c1_pads_coords_lat = [abs(lat - c1[0]) * 100 for lat in c1_pads_coords_lat]
    normalized_c1_pads_coords_lon = [abs(lon - c1[1]) * 100 for lon in c1_pads_coords_lon]
    normalized_c2_pads_coords_lat = [abs(lat - c2[0]) * 100 for lat in c2_pads_coords_lat]
    normalized_c2_pads_coords_lon = [abs(lon - c2[1]) * 100 for lon in c2_pads_coords_lon]

    full_padded_haversine_to_closest_buildings_c1 = [fast_haversine(c1[0], c1[1], 
                                                                    building_centroids[grid_building_matchings[cell][0]][0],
                                                                    building_centroids[grid_building_matchings[cell][0]][1]) 
                                                                    for cell in padded_neighbors[cell1]]
    
    full_padded_haversine_to_closest_buildings_c2 = [fast_haversine(c2[0], c2[1],
                                                                    building_centroids[grid_building_matchings[cell][0]][0],
                                                                    building_centroids[grid_building_matchings[cell][0]][1])
                                                                    for cell in padded_neighbors[cell2]]
    
    full_padded_building_to_building_distances_driving = [building_distance_matrix[grid_building_matchings[c1_pads[k]][0]][grid_building_matchings[c2_pads[l]][0]]
                                  for k in range(len(c1_pads))
                                  for l in range(len(c2_pads))]
    
    
    full_padded_ref_street_direction_c1 = [1 if ref_streets[grid_building_matchings[c1_pads[k]][0]] else 0
                                for k in range(len(c1_pads))]

    full_padded_ref_street_direction_c2 = [1 if ref_streets[grid_building_matchings[c2_pads[l]][0]] else 0
                                for l in range(len(c2_pads))]
    ref_direction_features = full_padded_ref_street_direction_c1 + full_padded_ref_street_direction_c2
        
    features = [
        c1[0], c1[1], c2[0], c2[1],
        abs(c1[0] - c2[0]) * 100,
        abs(c1[1] - c2[1]) * 100,
        cell1[0], cell1[1],
        cell2[0], cell2[1],
        geodesic_distance,
        cell_road_distance_driving,
    ] + full_padded_road_distances_driving \
    + padded_geodesic_c1 + padded_geodesic_c2 \
    + full_padded_haversine_to_closest_buildings_c1 \
    + full_padded_haversine_to_closest_buildings_c2 \
    + full_padded_building_to_building_distances_driving \
    + ref_direction_features\
    

    if not large_features:
        features_ = features[:12]
        return features_
    

    return features

for grid_size in [(50,50),(100,100),(150,150),(200,200),(250, 250)]:
    for _loss_ in ['MAE', 'MSE', 'MAPE']:
        for large_features in [True, False]:
            bbox = GetExtremePoints(GMApoly[2])
            grid = grid_corners(bbox, grid_size)
            padded_neighbors = get_padding_neighbors(grid)
            cell_centers = compute_cell_centers(grid)

            #load dataset/building_centroids.pkl
            with open('dataset/building_centroids.pkl', 'rb') as f:
                building_centroids = pickle.load(f)
                
            cell_center_dist_matrix_driving = np.load(f'dataset/centers_dist_mat_{grid_size[0]}_{grid_size[1]}.npy')

            with open(f'dataset/key_index_dict_{grid_size[0]}_{grid_size[1]}.pkl', 'rb') as f:
                center_key_indexes = pickle.load(f)

            with open('dataset/building_centroid_dist_matrix_driving.pkl', 'rb') as f:
                building_center_dist_matrix = pickle.load(f)

            with open(f'dataset/grid_building_matchings_{grid_size[0]}_{grid_size[1]}.pkl', 'rb') as f:
                grid_building_matchings = pickle.load(f)

            with open('dataset/building_centroids.pkl', 'rb') as f:
                building_centroids = pickle.load(f)

            with open('dataset/ref_building_street.pkl', 'rb') as f:
                ref_streets = pickle.load(f)



            BATCH_SIZE, EPOCHS          = 1024, 150
            HIDDEN_DIM, HIDDEN_LAYERS   = 1024, 6
            EVAL_EVERY, VAL_SPLIT       = 1, 0.20
            LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4
            LOSS_TO_OPTIMISE            = 'MSE'   # 'MSE' | 'MAPE' | 'MAE' | 'MQE'
            CKPT_METRIC                 = 'MSE'   # 'MSE' | 'MAPE' | 'MAE' | 'MQE'
            SEED, LOG_TRANSFORM         = 12, False
            EARLY_STOP_PATIENCE         = 25
            if not large_features:
                input_dim = 12  # Adjusted input dimension based on feature engineering
            else:
                input_dim = 228
            # Define the MLP model
            class MLP(nn.Module):
                def __init__(self, input_dim, hidden_dim=1024, num_layers=6):
                    super().__init__()
                    layers = []
                    layers.append(nn.Linear(input_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    
                    for _ in range(num_layers - 2):
                        layers.append(nn.Linear(hidden_dim, hidden_dim))
                        layers.append(nn.ReLU())
                    
                    layers.append(nn.Linear(hidden_dim, 1))
                    self.net = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.net(x)
                
            device = 'cpu'
            model = MLP(input_dim, HIDDEN_DIM, HIDDEN_LAYERS).to(device)

            #load the best model
            LOSS_TO_OPTIMISE = _loss_
            CKPT_METRIC = _loss_
            
            feat_size = '_large' if large_features else '_small'
            
            best_model_path = 'models/Final/' + f"model_opt_{LOSS_TO_OPTIMISE}_ckpt_{CKPT_METRIC}{feat_size}_{grid_size[0]}_{grid_size[1]}.pt"
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            #load datadict_westmount.pkl
            with open('dataset/data_dict_westmount_2.pkl', 'rb') as f:
                data_dict = pickle.load(f)

            mape_list = []
            squared_errors = []
            data_dict['predictions'] = {}
            IDX = []
            feats = []
            t_start = time.time()
            for i in tqdm(range(2000)):
                for j in range(2000):
                        IDX.append((i, j))
                        c1 = list(data_dict['coords'][i])
                        c2 = list(data_dict['coords'][j])
                        features = compute_features_for_new_data_2(
                        (c1[0], c1[1]),
                        (c2[0], c2[1]),
                        grid_size = grid_size,
                        padded_neighbors = padded_neighbors, 
                        centers_dist_mat = cell_center_dist_matrix_driving,
                        cell_centers = cell_centers,
                        bounding_box = bbox,
                        center_keys = center_key_indexes,
                        grid_building_matchings = grid_building_matchings,
                        building_centroids = building_centroids,
                        building_distance_matrix = building_center_dist_matrix,
                        ref_streets = ref_streets,
                        large_features = large_features
                        )   
                        feats.append(features)

            features_tensor = torch.tensor(feats, dtype=torch.float32).to(device)

            with torch.no_grad():
                pred = model(features_tensor).view(-1, 1).numpy()
            t_end = time.time()
            print(f"Prediction time for 4 million pairs: {t_end - t_start:.2f} seconds")
            
            for i in range(2000):
                data_dict['predictions'][i] = {}
                for j in range(2000):
                    pred_value = pred[i * 2000 + j][0]
                    data_dict['predictions'][i][j] = pred_value
                    # Save the prediction in the data_dict

            with open(f'predictions/new_large_data_dict_westmount_with_predictions_{LOSS_TO_OPTIMISE}_{CKPT_METRIC}{feat_size}_{grid_size[0]}_{grid_size[1]}.pkl', 'wb') as f:
                pickle.dump(data_dict, f)

            for i in range(20):
                for j in range(20):
                    print(data_dict['predictions'][i][j], data_dict['matrix'][i][j])


            del data_dict, building_center_dist_matrix, cell_center_dist_matrix_driving, center_key_indexes, grid_building_matchings, building_centroids, 