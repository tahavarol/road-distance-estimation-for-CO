import pickle
import gurobipy as gp

from gurobipy import GRB
from itertools import combinations
import numpy as np
import random
import time
import math
import pandas as pd

random.seed(42)

import elkai

from tqdm import tqdm

import retrieval 

import distance_matrix

import os


with gp.Env(empty=True) as env:
    env.setParam('OutputFlag', 0)
    env.start()


def random_subset(iterable, k):
    """
    Return a single random subset (as a list) of length k,
    sampled without replacement from iterable.
    """
    pool = list(iterable)        # ensure we have random-access
    return random.sample(pool, k)


from distance_matrix import scaled_euclidean_matrix


def sample_instance(n_stops, data_dict):
    location_idx = random.sample(list(data_dict['coords'].keys()), n_stops)

    location_coords = [list(data_dict['coords'][loc]) for loc in location_idx]
    location_coords = np.array(location_coords)

    road_dist_mat = np.zeros((n_stops, n_stops))
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                road_dist_mat[i, j] = data_dict['matrix'][location_idx[i]][location_idx[j]]

    '''prediction_dist_mat = np.zeros((n_stops, n_stops))
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                prediction_dist_mat[i, j] = data_dict['predictions'][location_idx[i]][location_idx[j]]'''


    manhattan_dist_mat = np.zeros((n_stops, n_stops))
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                manhattan_dist_mat[i, j] = abs(location_coords[i][0] - location_coords[j][0]) + abs(location_coords[i][1] - location_coords[j][1])
    
    euclidean_dist_mat = np.zeros((n_stops, n_stops))
    for i in range(n_stops):
        for j in range(n_stops):
            if i != j:
                euclidean_dist_mat[i, j] = math.sqrt((location_coords[i][0] - location_coords[j][0])**2 + (location_coords[i][1] - location_coords[j][1])**2)


    scaled_euc_matrix = distance_matrix.scaled_euclidean_matrix(location_coords)


    return road_dist_mat, scaled_euc_matrix, location_idx, location_coords


def RouteLength(route, dist_mat):
    """
    Calculate the length of a route given a distance matrix.
    """
    length = 0
    for i in range(len(route)-1):
        length += dist_mat[route[i], route[i+1]]
    return length

def RouteToAdjacency(route):
    """
    Convert a route to an adjacency matrix.
    """
    n = len(route)-1
    adj = np.zeros((n, n))
    for i in range(n-1):
        adj[route[i], route[i+1]] = 1
        adj[route[i+1], route[i]] = 1
    return adj



import numpy as np
import gurobipy as gp
from gurobipy import GRB

def p_median_solver(distance_matrix, n_centers):
    """
    Solve p-median problem using Gurobi with an asymmetric distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        n x n asymmetric distance matrix.
    n_centers : int
        Number of centers to choose.

    Returns
    -------
    adjacency_matrix : np.ndarray
        n x n binary matrix, where adjacency_matrix[i,j] = 1 
        if location i is assigned to center j, else 0.
    centers : list
        Indices of chosen centers.
    """
    n = distance_matrix.shape[0]
    p = n_centers

    # Model
    m = gp.Model("p_median")
    #add time limit
    m.setParam('TimeLimit', 60)
    m.setParam('OutputFlag', 0)
    m.setParam('Threads', 1)
    # Decision variables
    y = m.addVars(n, vtype=GRB.BINARY, name="y")                 # y[j] = 1 if j is a center
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")              # x[i,j] = 1 if i assigned to j

    # Objective: minimize total assignment cost
    m.setObjective(gp.quicksum(distance_matrix[i, j] * x[i, j] for i in range(n) for j in range(n)),
                   GRB.MINIMIZE)

    # Constraints
    # 1. Each node must be assigned to exactly one center
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1)

    # 2. Assignment only allowed if j is a center
    for i in range(n):
        for j in range(n):
            m.addConstr(x[i, j] <= y[j])

    # 3. Exactly p centers must be chosen
    m.addConstr(gp.quicksum(y[j] for j in range(n)) == p)

    # Solve
    m.optimize()


    # Extract solution
    adjacency_matrix = np.zeros((n, n), dtype=int)
    centers = []
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        for i in range(n):
            for j in range(n):
                if x[i, j].x > 0.5:
                    adjacency_matrix[i, j] = 1
        centers = [j for j in range(n) if y[j].x > 0.5]

    return adjacency_matrix, centers , m.objVal, m.MIPGap*100


grid_res = [(50,50),(100,100),(150,150),(200,200),(250,250)]
sizes = [20, 50, 100, 250, 500]
n_centers = {20: [2,4],
             50: [5,10],
             100: [5, 10, 25, 50],
             250: [5, 10, 25, 50],
             500: [5, 10, 25, 50]}
for grid in grid_res:
    with open(f'predictions/new_large_data_dict_westmount_with_predictions_SNAP_{grid[0]}_{grid[1]}.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    print(f'Solving benchmark problems for {grid[0]}_{grid[1]}')
    for size in sizes:
        for n_center in n_centers[size]:
            for instance in tqdm(range(100)):
                with open(f'dataset/optimization_datasets/PMEDIAN/{size}/GUROBI_instance_{instance}_{n_center}.pkl', 'rb') as f:
                    GUROBI_dict = pickle.load(f)
                d_mat = np.zeros((size, size))
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            d_mat[i, j] = data_dict['predictions'][GUROBI_dict['location_idx'][i]][GUROBI_dict['location_idx'][j]]
                adjacency, centers, obj_val, mip_gap = p_median_solver(d_mat, n_center)
                our_dict = {
                    'predicted_centers': centers,
                    'predicted_cost': np.sum(adjacency * GUROBI_dict['road_dist_mat']),
                    'predicted_mip_gap': mip_gap,
                    'gurobi_cost': GUROBI_dict['obj_val']

                }
                with open(f'dataset/optimization_datasets/PMEDIAN/{size}/SNAP_{grid[0]}_{grid[1]}_{instance}_{n_center}.pkl', 'wb') as f:
                    pickle.dump(our_dict, f)
        


