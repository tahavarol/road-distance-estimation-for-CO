# generate_predictions_for_dfl_models.py
# DFL inference for Westmount — correct for dict/list coords + correct A-dimension (row-wise)

import os
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from typing import Optional


# =========================
# Load polygon
# =========================
with open('dataset/GMApoly.pkl', 'rb') as f:
    GMApoly = pickle.load(f)

def GetExtremePoints(coords):
    lat_min = np.array(coords)[:,0].min()
    lat_max = np.array(coords)[:,0].max()
    lon_min = np.array(coords)[:,1].min()
    lon_max = np.array(coords)[:,1].max()
    return lat_min, lon_min, lat_max, lon_max

def grid_corners(extremes, divisions):
    ymin, xmin, ymax, xmax = extremes
    nx, ny = divisions
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    cells = {}
    for iy in range(ny):
        for ix in range(nx):
            x0 = xmin + ix * dx
            y0 = ymin + iy * dy
            x1 = x0 + dx
            y1 = y0 + dy
            cells[(ix, iy)] = ((y0, x0), (y0, x1), (y1, x1), (y1, x0))
    return cells

def compute_cell_centers(grid):
    centers = {}
    for key, corners in grid.items():
        x_center = (corners[0][1] + corners[2][1]) / 2
        y_center = (corners[0][0] + corners[2][0]) / 2
        centers[key] = (y_center, x_center)
    return centers

def locate_point(point, extremes, divisions):
    y, x = point
    ymin, xmin, ymax, xmax = extremes
    nx, ny = divisions
    if not (xmin <= x < xmax and ymin <= y < ymax):
        return None
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    ix = int((x - xmin) // dx)
    iy = int((y - ymin) // dy)
    ix = min(ix, nx - 1)
    iy = min(iy, ny - 1)
    return (ix, iy)

def get_padding_neighbors(grid):
    xs = [k[0] for k in grid.keys()]
    ys = [k[1] for k in grid.keys()]
    nx = max(xs) + 1
    ny = max(ys) + 1

    def clip(v, lo, hi):
        return max(lo, min(v, hi))

    neighbors_dict = {}
    for (i, j) in grid.keys():
        neighbors = []
        neighbors.append((clip(i - 1, 0, nx - 1), j))
        neighbors.append((clip(i - 1, 0, nx - 1), clip(j + 1, 0, ny - 1)))
        neighbors.append((i, clip(j + 1, 0, ny - 1)))
        neighbors.append((clip(i + 1, 0, nx - 1), clip(j + 1, 0, ny - 1)))
        neighbors.append((clip(i + 1, 0, nx - 1), j))
        neighbors.append((clip(i + 1, 0, nx - 1), clip(j - 1, 0, ny - 1)))
        neighbors.append((i, clip(j - 1, 0, ny - 1)))
        neighbors.append((clip(i - 1, 0, nx - 1), clip(j - 1, 0, ny - 1)))
        neighbors.append((i, j))
        neighbors_dict[(i, j)] = tuple(neighbors)
    return neighbors_dict

R = 6_371_008.8
def fast_haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat  = lat2 - lat1
    dlon  = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_features_for_new_data_2(
    c1,
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
    large_features=True
):
    cell1 = locate_point(c1, bounding_box, grid_size)
    cell2 = locate_point(c2, bounding_box, grid_size)

    geodesic_distance = fast_haversine(c1[0], c1[1], c2[0], c2[1])

    c1_pads = padded_neighbors[cell1]
    c2_pads = padded_neighbors[cell2]

    cell_road_distance_driving = centers_dist_mat[center_keys[cell1]][center_keys[cell2]]

    full_padded_road_distances_driving = [
        centers_dist_mat[center_keys[c1_pads[k]]][center_keys[c2_pads[l]]]
        for k in range(len(c1_pads))
        for l in range(len(c2_pads))
    ]

    padded_geodesic_c1 = [
        fast_haversine(c1[0], c1[1], cell_centers[cell][0], cell_centers[cell][1])
        for cell in padded_neighbors[cell1]
    ]
    padded_geodesic_c2 = [
        fast_haversine(c2[0], c2[1], cell_centers[cell][0], cell_centers[cell][1])
        for cell in padded_neighbors[cell2]
    ]

    full_padded_haversine_to_closest_buildings_c1 = [
        fast_haversine(
            c1[0], c1[1],
            building_centroids[grid_building_matchings[cell][0]][0],
            building_centroids[grid_building_matchings[cell][0]][1],
        )
        for cell in padded_neighbors[cell1]
    ]
    full_padded_haversine_to_closest_buildings_c2 = [
        fast_haversine(
            c2[0], c2[1],
            building_centroids[grid_building_matchings[cell][0]][0],
            building_centroids[grid_building_matchings[cell][0]][1],
        )
        for cell in padded_neighbors[cell2]
    ]

    full_padded_building_to_building_distances_driving = [
        building_distance_matrix[grid_building_matchings[c1_pads[k]][0]][grid_building_matchings[c2_pads[l]][0]]
        for k in range(len(c1_pads))
        for l in range(len(c2_pads))
    ]

    full_padded_ref_street_direction_c1 = [
        1 if ref_streets[grid_building_matchings[c1_pads[k]][0]] else 0
        for k in range(len(c1_pads))
    ]
    full_padded_ref_street_direction_c2 = [
        1 if ref_streets[grid_building_matchings[c2_pads[l]][0]] else 0
        for l in range(len(c2_pads))
    ]
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
      + ref_direction_features

    if not large_features:
        return features[:12]
    return features


# =========================================================
# DFL model (EXACT)
# =========================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_layers=6):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.net[-1].bias, 0.1)

    def forward(self, x):
        return self.net(x)


class ArcCostPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        num_layers=6,
        eps=1e-8,
        feat_mean: Optional[np.ndarray] = None,
        feat_std: Optional[np.ndarray] = None,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.base = MLP(input_dim, hidden_dim, num_layers)
        self.eps = float(eps)
        self.normalize_features = bool(normalize_features)

        if feat_mean is None or feat_std is None:
            feat_mean = np.zeros((input_dim,), dtype=np.float32)
            feat_std = np.ones((input_dim,), dtype=np.float32)

        self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("feat_std",  torch.tensor(feat_std,  dtype=torch.float32).view(1, 1, -1))

    def forward(self, x):
        if self.normalize_features:
            x = (x - self.feat_mean) / (self.feat_std + 1e-8)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, A, Fdim = x.shape
        raw = self.base(x.reshape(B * A, Fdim)).reshape(B, A)
        raw_min = raw.min(dim=1, keepdim=True).values.detach()
        return raw - raw_min + self.eps


def load_ckpt_into_model(model, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = (
        (ckpt.get("model") if isinstance(ckpt, dict) else None)
        or (ckpt.get("model_state") if isinstance(ckpt, dict) else None)
        or (ckpt.get("state_dict") if isinstance(ckpt, dict) else None)
        or ckpt
    )
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return ckpt


# =========================================================
# Correct inference (row-wise; handles coords dict)
# =========================================================
def run_dfl_inference(
    grid_size=(50, 50),
    large_features=True,
    instance_size=20,
    ckpt_dir="models/Final2",
    n_points=2000,
    device="cpu",
    out_dir="predictions",
    method="HYBRID"
):
    os.makedirs(out_dir, exist_ok=True)

    # checkpoint naming (ATSP example)
    feat_tag = "large" if large_features else "small"
    ckpt_path = os.path.join(ckpt_dir, f"DFL_{method}_{grid_size[0]}_{grid_size[1]}_{feat_tag}_{instance_size}.pt")
    if not os.path.exists(ckpt_path):
        print("[SKIP] missing checkpoint:", ckpt_path)
        return

    # load dict
    with open('dataset/data_dict_westmount_2.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    coords_obj = data_dict["coords"]

    # FIX: coords can be dict -> turn into list in order
    if isinstance(coords_obj, dict):
        coords = [coords_obj[i] for i in range(n_points)]
    else:
        coords = list(coords_obj)[:n_points]

    input_dim = 228 if large_features else 12

    # load static assets
    bbox = GetExtremePoints(GMApoly[2])
    grid = grid_corners(bbox, grid_size)
    padded_neighbors = get_padding_neighbors(grid)
    cell_centers = compute_cell_centers(grid)

    centers_dist = np.load(f'dataset/centers_dist_mat_{grid_size[0]}_{grid_size[1]}.npy')
    with open(f'dataset/key_index_dict_{grid_size[0]}_{grid_size[1]}.pkl', 'rb') as f:
        center_key_indexes = pickle.load(f)
    with open(f'dataset/grid_building_matchings_{grid_size[0]}_{grid_size[1]}.pkl', 'rb') as f:
        grid_building_matchings = pickle.load(f)

    with open('dataset/building_centroids.pkl', 'rb') as f:
        building_centroids = pickle.load(f)
    with open('dataset/building_centroid_dist_matrix_driving.pkl', 'rb') as f:
        building_dist = pickle.load(f)
    with open('dataset/ref_building_street.pkl', 'rb') as f:
        ref_streets = pickle.load(f)

    # model
    model = ArcCostPredictor(input_dim=input_dim, hidden_dim=1024, num_layers=6).to(device)
    load_ckpt_into_model(model, ckpt_path, device=device)
    model.eval()

    data_dict["predictions"] = {}

    print(f"\n=== DFL inference | grid={grid_size} | feat={feat_tag} | inst={instance_size} ===")
    print("ckpt:", ckpt_path)

    t0 = time.time()

    for i in tqdm(range(n_points), desc="Predicting rows"):
        c1 = coords[i]
        row_feats = []
        for j in range(n_points):
            c2 = coords[j]
            feats = compute_features_for_new_data_2(
                (c1[0], c1[1]),
                (c2[0], c2[1]),
                grid_size=grid_size,
                padded_neighbors=padded_neighbors,
                centers_dist_mat=centers_dist,
                cell_centers=cell_centers,
                bounding_box=bbox,
                center_keys=center_key_indexes,
                grid_building_matchings=grid_building_matchings,
                building_centroids=building_centroids,
                building_distance_matrix=building_dist,
                ref_streets=ref_streets,
                large_features=large_features,
            )
            row_feats.append(feats)

        x_row = torch.tensor(row_feats, dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, F]

        with torch.no_grad():
            row_pred = model(x_row).squeeze(0).cpu().numpy()  # [N]

        data_dict["predictions"][i] = {j: float(row_pred[j]) for j in range(n_points)}

    t1 = time.time()
    print(f"Total prediction time: {t1 - t0:.2f}s for {n_points*n_points:,} pairs")

    out_path = os.path.join(out_dir, f"new_large_data_dict_westmount_with_predictions_{method}_{feat_tag}_{grid_size[0]}_{grid_size[1]}_{instance_size}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)
    print("Saved:", out_path)

    # sanity
    for i in range(5):
        for j in range(5):
            print(data_dict["predictions"][i][j], data_dict["matrix"][i][j])


# =========================================================
# Sweep like your regression script
# =========================================================
if __name__ == "__main__":
    for grid_size in [(50,50),(100,100),(150,150),(200,200),(250,250)]:
        for large_features in [True, False]:
            for instance_size in [20, 50]:
                run_dfl_inference(
                    grid_size=grid_size,
                    large_features=large_features,
                    instance_size=instance_size,
                    ckpt_dir="models/Final2",
                    n_points=2000,
                    device="cpu",
                    out_dir="predictions",
                    method="HYBRID"
                )