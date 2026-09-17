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

#load data_dict_westmount.pkl
with open('dataset/data_dict_westmount_2.pkl', 'rb') as f:
    data_dict = pickle.load(f)

sizes = [20, 50, 100, 250, 500]
for size in sizes:
    #create directory dataset/RAW/{size} if it does not exist
    os.makedirs(f'dataset/optimization_datasets/RAW/{size}', exist_ok=True)
    for i in tqdm(range(100)):
        road_dist_mat, scaled_euc_matrix, location_idx, location_coords = sample_instance(size, data_dict)
        data_dict_ = {
            'road_dist_mat': road_dist_mat,
            'scaled_euc_matrix': scaled_euc_matrix,
            'location_idx': location_idx,
            'location_coords': location_coords
        }
        with open(f'dataset/optimization_datasets/RAW/{size}/instance_{i}.pkl', 'wb') as f:
            pickle.dump(data_dict_, f)