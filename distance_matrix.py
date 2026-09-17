import numpy as np
from geopy.distance import geodesic   # pip install geopy
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import requests
import math
# ── DISTANCE MATRICES ─────────────────────────────────────────────────────
def geodesic_distance_matrix(coords: np.ndarray,
                             *,
                             unit: str = "m") -> np.ndarray:

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have shape (n, 2)")

    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)

    # compute only upper triangle, then mirror to save work
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic(coords[i], coords[j])
            D[i, j] = d.m if unit == "m" else d.km
            D[j, i] = D[i, j]

    return D





def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean (ℓ₂) distances."""
    coords = _validate_coords(coords)
    return squareform(pdist(coords, metric="euclidean"))

def manhattan_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Pairwise Manhattan (ℓ₁) distances."""
    coords = _validate_coords(coords)
    return squareform(pdist(coords, metric="cityblock"))


def _validate_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have shape (n, 2)")
    return coords


import numpy as np

def scaled_euclidean_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Pairwise `scaled_euclidean` distances for an array of lat/lon points.

    Parameters
    ----------
    coords : (n, 2) array_like
        Latitude-longitude pairs in decimal degrees:
        [[lat₀, lon₀],
         …,
         [latₙ₋₁, lonₙ₋₁]]

    Returns
    -------
    D : (n, n) ndarray[float]
        Symmetric distance matrix (metres) with zeros on the diagonal.

    Notes
    -----
    * Uses exactly the same scaling formula you defined:
          lat_scale = 111 000 m
          lon_scale = 111 000 m · cos( mean_latitude )
      which is the equirectangular approximation—excellent for
      neighbourhood-sized regions (≈ < 200 km across).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have shape (n, 2)")

    # Column vectors for broadcasting
    lat1 = coords[:, 0][:, None]        # shape (n, 1)
    lon1 = coords[:, 1][:, None]        # shape (n, 1)
    lat2 = coords[:, 0][None, :]        # shape (1, n)
    lon2 = coords[:, 1][None, :]        # shape (1, n)

    lat_scale = 111_000.0
    mean_lat = (lat1 + lat2) / 2.0
    lon_scale = 111_000.0 * np.cos(np.radians(mean_lat))

    dlat = (lat1 - lat2) * lat_scale
    dlon = (lon1 - lon2) * lon_scale
    return np.sqrt(dlat**2 + dlon**2)


# ── CONFIG ────────────────────────────────────────────────────────────────
OSRM      = "http://localhost:5010"   # your docker-ised OSRM
PROFILE   = "driving"
N_POINTS  = 10_000
CHUNK     = 100         
def distance_matrix(coords, max_table_size=100_000, chunk=CHUNK):
    n = len(coords)
    matrix = np.empty((n, n), dtype=np.float32)
    blocks = math.ceil(n / chunk)

    def osrm_block(src_idx, dst_idx):
        block_coords = [coords[i] for i in src_idx] + [coords[j] for j in dst_idx]
        coord_str = ";".join(f"{lon},{lat}" for lat, lon in block_coords)
        src_param = ";".join(map(str, range(len(src_idx))))
        dst_param = ";".join(str(len(src_idx)+k) for k in range(len(dst_idx)))
        r = requests.get(f"{OSRM}/table/v1/{PROFILE}/{coord_str}",
                         params={"sources":src_param,
                                 "destinations":dst_param,
                                 "annotations":"distance"})
        r.raise_for_status()
        return np.array(r.json()["distances"], dtype=np.float32)

    with tqdm(total=blocks*blocks, desc="Building matrix") as bar:
        for br in range(blocks):
            src_idx = list(range(br*chunk, min((br+1)*chunk, n)))
            for bc in range(blocks):
                dst_idx = list(range(bc*chunk, min((bc+1)*chunk, n)))
                matrix[np.ix_(src_idx, dst_idx)] = osrm_block(src_idx, dst_idx)
                bar.update(1)
    return matrix