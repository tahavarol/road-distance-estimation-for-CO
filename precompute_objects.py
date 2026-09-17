"""Precompute the notebook's map objects.

Edit grid_size and k below, then run:
    python precompute_objects.py

Requires numpy, requests, tqdm, shapely, geopandas, osmnx (2.x), and a running
OSRM server. Coordinates are (latitude, longitude); road distances are metres.
"""

import json
import math
import pickle
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import Polygon
from tqdm import tqdm

# Edit these values before running the script.
grid_size = (250, 250)
k = 1

OSRM = "http://localhost:5011"
PROFILE = "driving"
CHUNK = 100

BASE_DIR = Path(__file__).resolve().parent
POLYGON_FILE = BASE_DIR / "dataset" / "MontrealServiceArea.pkl"
OUTPUT_DIR = None  # Uses dataset/precomputed/grid_<nx>_<ny>_k<k>/.
R = 6_371_008.8


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


def fast_haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat  = lat2 - lat1
    dlon  = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def get_building_centroids(polygon_coords, tags=None):
    import osmnx as ox

    if tags is None:
        tags = {"building": True}

    # 1) build the polygon  (note: (lon, lat) order for shapely)
    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])

    # 2) download OSM geometries that match the tags
    gdf = ox.features_from_polygon(poly, tags=tags)

    # 3) compute centroids
    centroids = {i: (geom.centroid.y, geom.centroid.x)   # (lat, lon)
                 for i, geom in enumerate(gdf.geometry)}

    return centroids


def get_k_closest_buildings(cell_centers, building_centroids, k=1):
    """
    For each cell center, find the k closest building centroids.

    Parameters
    ----------
    cell_centers : dict
        Dictionary with keys as cell indices and values as (lat, lon) tuples.
    building_centroids : dict
        Dictionary with keys as building indices and values as (lat, lon) tuples.
    k : int
        Number of closest buildings to find.

    Returns
    -------
    dict
        Dictionary with keys as cell centers and values as lists of indices of the k closest buildings.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if k > len(building_centroids):
        raise ValueError(f"k={k} exceeds the number of available buildings ({len(building_centroids)}).")
    closest_buildings = {}
    
    for cell_key, cell_center in tqdm(cell_centers.items()):
        distances = []
        for building_key, building_center in building_centroids.items():
            dist = fast_haversine(cell_center[0], cell_center[1],
                                  building_center[0], building_center[1])
            distances.append((building_key, dist))
        
        # Sort by distance and take the first k elements
        distances.sort(key=lambda x: x[1])
        closest_buildings[cell_key] = [building_key for building_key, _ in distances[:k]]
    
    return closest_buildings


def filter_assigned_buildings(building_centroids, k_closest_buildings):
    """Retain assigned buildings and remap every object to contiguous building IDs."""
    assigned_building_ids = sorted({building_id
                                   for building_ids in k_closest_buildings.values()
                                   for building_id in building_ids})
    building_index_mapping = {old_index: new_index
                              for new_index, old_index in enumerate(assigned_building_ids)}
    building_centroids = {new_index: building_centroids[old_index]
                          for old_index, new_index in building_index_mapping.items()}
    k_closest_buildings = {cell_key: [building_index_mapping[building_id] for building_id in building_ids]
                          for cell_key, building_ids in k_closest_buildings.items()}
    return building_centroids, k_closest_buildings, building_index_mapping


def _norm(val):
    if isinstance(val, (list, tuple, np.ndarray)):
        flags = {_norm(item) for item in val}
        return flags.pop() if len(flags) == 1 else None
    if val in (True, "yes", "Yes", 1, "1"):
        return True
    if val in (False, "no", "No", 0, "0"):
        return False
    if val in ("-1", "reverse"):
        return True         # still one-way, just opposite direction
    return None


def build_street_network(polygon_coords):
    import osmnx as ox

    poly = Polygon([(lon, lat) for lat, lon in polygon_coords])
    G = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
    G = ox.project_graph(G)
    edges = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    oneway = edges["oneway"].apply(_norm)
    return G, oneway


def closest_street_oneway(latitudes, longitudes, G, oneway):
    """
    Vectorised nearest-street query.

    Parameters
    ----------
    latitudes, longitudes : 1-D sequences of float (EPSG:4326)

    Returns
    -------
    np.ndarray[object] – elements are True, False, or None
    """
    import geopandas as gpd
    import osmnx as ox

    # project coordinates in bulk
    pts = gpd.points_from_xy(longitudes, latitudes, crs=4326)
    pts_proj = gpd.GeoSeries(pts).to_crs(G.graph["crs"])
    xs, ys = pts_proj.x.values, pts_proj.y.values

    # one vectorised nearest-edge search → list[(u,v,key), ...]
    edge_tuples = ox.distance.nearest_edges(G, xs, ys, return_dist=False)

    # look up the ‘oneway’ flag
    return np.fromiter((oneway.get(t) for t in edge_tuples), dtype=object)


def distance_matrix(coords, chunk=100, base_url="http://localhost:5011",
                    profile="driving", timeout=120):
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer.")
    n = len(coords)
    matrix = np.empty((n, n), dtype=np.float32)
    blocks = math.ceil(n / chunk)

    def osrm_block(src_idx, dst_idx):
        block_coords = [coords[i] for i in src_idx] + [coords[j] for j in dst_idx]
        coord_str = ";".join(f"{lon},{lat}" for lat, lon in block_coords)
        src_param = ";".join(map(str, range(len(src_idx))))
        dst_param = ";".join(str(len(src_idx)+k) for k in range(len(dst_idx)))
        r = requests.get(f"{base_url.rstrip('/')}/table/v1/{profile}/{coord_str}",
                         params={"sources":src_param,
                                 "destinations":dst_param,
                                 "annotations":"distance"}, timeout=timeout)
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != "Ok" or "distances" not in payload:
            raise RuntimeError(f"OSRM table failed: {payload.get('code')}: {payload.get('message', '')}")
        block = np.array(payload["distances"], dtype=np.float32)
        if block.shape != (len(src_idx), len(dst_idx)):
            raise ValueError("OSRM returned an unexpected distance-table shape.")
        # OSRM null values denote unreachable directed pairs.
        block[np.isnan(block)] = np.inf
        return block

    with tqdm(total=blocks*blocks, desc="Building matrix") as bar:
        for br in range(blocks):
            src_idx = list(range(br*chunk, min((br+1)*chunk, n)))
            for bc in range(blocks):
                dst_idx = list(range(bc*chunk, min((bc+1)*chunk, n)))
                matrix[np.ix_(src_idx, dst_idx)] = osrm_block(src_idx, dst_idx)
                bar.update(1)
    return matrix


def save_pickle(path, value):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def precompute_objects(k, grid_size, polygon_file=BASE_DIR / "dataset" / "MontrealServiceArea.pkl",
                       output_dir=None, osrm_url="http://localhost:5011", profile="driving", chunk=100):
    """Run the notebook pipeline and return the directory containing its saved objects."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if len(grid_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid_size):
        raise ValueError("grid_size must contain two positive integers (nx, ny).")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer.")
    nx, ny = grid_size
    polygon_file = Path(polygon_file)
    with polygon_file.open("rb") as stream:
        GMApoly = pickle.load(stream)
    coordinates = np.asarray(GMApoly, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3
            or not np.isfinite(coordinates).all()):
        raise ValueError("The polygon file must contain a list of (latitude, longitude) vertices.")
    output_dir = (Path(output_dir) if output_dir is not None else
                  BASE_DIR / "dataset" / "precomputed" / f"grid_{nx}_{ny}_k{k}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / "config.json"
    metadata = {"k": k, "grid_size": list(grid_size), "polygon_file": str(polygon_file.resolve()),
                "osrm_url": osrm_url, "profile": profile, "chunk_size": chunk,
                "coordinate_order": "latitude, longitude", "distance_units": "metres",
                "tile_order": "W, NW, N, NE, E, SE, S, SW, center", "tile_padding": "edge clipping",
                "unreachable_distance": "positive_infinity", "complete": False}
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")

    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.cache_folder = BASE_DIR / "cache"

    bbox = GetExtremePoints(GMApoly)
    grid = grid_corners(bbox, grid_size)
    padded_neighbors = get_padding_neighbors(grid)
    cell_centers = compute_cell_centers(grid)
    print(f"Grid: {nx} x {ny}; k: {k}; output: {output_dir}")
    print("Downloading building data (cached when available)...")
    building_centroids = get_building_centroids(GMApoly)
    n_candidates = len(building_centroids)
    k_closest_buildings = get_k_closest_buildings(cell_centers, building_centroids, k=k)
    building_centroids, k_closest_buildings, building_index_mapping = filter_assigned_buildings(
        building_centroids, k_closest_buildings)
    print(f"Retained {len(building_centroids):,} of {n_candidates:,} buildings.")

    print("Computing street indicators for retained buildings...")
    G, oneway = build_street_network(GMApoly)
    lats = [building_centroids[key][0] for key in building_centroids]
    lons = [building_centroids[key][1] for key in building_centroids]
    res = list(closest_street_oneway(lats, lons, G, oneway))
    ref_streets = {key: res[idx] for idx, key in enumerate(building_centroids.keys())}
    key_index_dict = {key: idx for idx, key in enumerate(cell_centers.keys())}

    objects = {
        "ref_building_street.pkl": ref_streets,
        f"grid_building_matchings_{nx}_{ny}.pkl": k_closest_buildings,
        f"key_index_dict_{nx}_{ny}.pkl": key_index_dict,
        "building_centroids.pkl": building_centroids,
        "building_index_mapping.pkl": building_index_mapping,
        f"grid_{nx}_{ny}.pkl": grid,
        f"cell_centers_{nx}_{ny}.pkl": cell_centers,
        f"padded_neighbors_{nx}_{ny}.pkl": padded_neighbors,
    }
    for filename, value in objects.items():
        save_pickle(output_dir / filename, value)

    print("Computing the grid-center distance matrix...")
    cell_centers_array = np.array(list(cell_centers.values()))
    cell_centers_distance_matrix = distance_matrix(cell_centers_array, chunk=chunk,
                                                   base_url=osrm_url, profile=profile)
    np.save(output_dir / f"centers_dist_mat_{nx}_{ny}.npy", cell_centers_distance_matrix)
    del cell_centers_distance_matrix

    print("Computing the retained-building distance matrix...")
    building_centroids_array = np.array(list(building_centroids.values()))
    building_centroids_distance_matrix = distance_matrix(building_centroids_array, chunk=chunk,
                                                          base_url=osrm_url, profile=profile)
    save_pickle(output_dir / f"building_centroid_dist_matrix_{profile}.pkl", building_centroids_distance_matrix)
    metadata.update(complete=True, n_building_candidates=n_candidates, n_reference_buildings=len(building_centroids))
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved precomputation objects to {output_dir}")
    return output_dir


def main():
    return precompute_objects(k=k, grid_size=grid_size, polygon_file=POLYGON_FILE,
                              output_dir=OUTPUT_DIR, osrm_url=OSRM, profile=PROFILE,
                              chunk=CHUNK)


if __name__ == "__main__":
    main()
