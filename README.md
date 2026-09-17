# Learning Road Distances for Combinatorial Optimization

Code accompanying:

**Taha Varol and Okan Arslan, *Learning Road Distances for Combinatorial
Optimization*.**

This repository contains the code used to generate road-network distance
data, construct map-based features, train distance-surrogate models, and
evaluate them on downstream combinatorial optimization problems.

The experiments use a service area in Greater Montréal and consider:

-   Directed road-network distances obtained from OSRM
-   Grid-based spatial and road-network features
-   Decision-agnostic learning (DAL)
-   Decision-focused learning (DFL)
-   Asymmetric Traveling Salesman Problem (ATSP)
-   (p)-median problem

## 1. OSRM Setup

[OSRM](https://project-osrm.org/) is used as the road-network distance
oracle.

The experiments use a **pure shortest-path driving-distance
configuration**: routes minimize distance along the permitted directed
road network rather than travel time or OSRM's default routability
objective. Consequently, one-way streets and other driving restrictions
are respected, and the resulting distance matrix can be asymmetric.

The custom OSRM profile used for this configuration is provided at:

``` text
osrm/car_distance.lua
```

A Québec OpenStreetMap extract can be downloaded from
[Geofabrik](https://download.geofabrik.de/north-america/canada/quebec.html).

The experiments use OSRM with Contraction Hierarchies (CH). A basic
Docker setup is:

``` bash
OSRM_IMAGE=ghcr.io/project-osrm/osrm-backend:v6.0.0

mkdir -p osrm/data
```

Place the Québec `.osm.pbf` file in `osrm/data/`, then preprocess it
using the supplied distance profile:

``` bash
docker run --rm \
  -v "$PWD/osrm/data:/data" \
  -v "$PWD/osrm/car_distance.lua:/opt/car_distance.lua:ro" \
  "$OSRM_IMAGE" \
  osrm-extract -p /opt/car_distance.lua /data/quebec-latest.osm.pbf

docker run --rm \
  -v "$PWD/osrm/data:/data" \
  "$OSRM_IMAGE" \
  osrm-contract /data/quebec-latest.osrm
```

Start the routing server:

``` bash
docker run -d --name osrm-quebec-distance \
  -p 5011:5000 \
  -v "$PWD/osrm/data:/data:ro" \
  "$OSRM_IMAGE" \
  osrm-routed --algorithm ch --max-table-size 200 \
  /data/quebec-latest.osrm
```

The OSRM endpoint used by the code is:

``` text
http://localhost:5011
```

Distances returned by this server are shortest permitted driving
distances in metres.

## 2. Python Environment

The main dependencies are:

-   NumPy
-   pandas
-   SciPy
-   PyTorch
-   GeoPandas
-   Shapely
-   OSMnx
-   NetworkX
-   scikit-learn
-   Requests
-   tqdm
-   Matplotlib
-   elkai
-   gurobipy

A suitable environment can be created with:

``` bash
conda create -n road-distance python=3.9 pip
conda activate road-distance

pip install \
  numpy pandas scipy torch matplotlib requests tqdm \
  shapely geopandas pyproj osmnx networkx geopy scikit-learn \
  elkai gurobipy
```

Gurobi requires a valid license for the (p)-median experiments.

## 3. Data Generation and Precomputation

### Generate locations

Run:

``` bash
python generate_locations.py
```

This generates training and testing locations within the Montréal
service area.

### Precompute map objects

Configure the desired grid resolution and OSRM endpoint in
`precompute_objects.py`, for example:

``` python
grid_size = (250, 250)
k = 1
OSRM = "http://localhost:5011"
PROFILE = "driving"
CHUNK = 100
```

Then run:

``` bash
python precompute_objects.py
```

This script constructs the reusable map objects required for feature
generation, including:

-   Grid cells and centroids
-   Neighboring grid cells
-   Building references
-   Street-direction indicators
-   Directed centroid-to-centroid road distances
-   Directed building-to-building road distances

Precomputed objects are stored under:

``` text
dataset/precomputed/
```

## 4. Model Training and Evaluation

The repository contains scripts for decision-agnostic and
decision-focused distance estimation and their downstream evaluation.

Important scripts include:

``` text
generate_locations.py
precompute_objects.py
distance_matrix.py

DAL_trainer.py
DAL_inference.py
DFL_inference.py

generate_benchmark_problems.py

DAL_downstream_PMEDIAN.py
DFL_downstream_PMEDIAN.py

SNAP_downstream_optimization_ATSP.py
SNAP_downstream_optimization_PMEDIAN.py
```

The overall experimental pipeline is:

``` text
OSRM setup
    ↓
Location generation
    ↓
Map and distance precomputation
    ↓
Feature and label generation
    ↓
Model training
    ↓
Inference
    ↓
ATSP and p-median evaluation
```

Model and experiment settings can be modified directly in the
corresponding scripts.

## 5. Notes

OSRM coordinates are provided in:

``` text
longitude, latitude
```

while some Python objects in this repository store coordinates as:

``` text
(latitude, longitude)
```

Make sure the appropriate ordering is used when constructing OSRM
requests.

The road-network distances are **directed**. In particular,

\[ d(i,j) `\neq `{=tex}d(j,i) \]

may occur because of one-way streets and other road-network
restrictions.

Large grid resolutions can also require substantial memory because
centroid-to-centroid distance matrices are dense.

## 6. Citation

If you use this code, please cite:

**Taha Varol and Okan Arslan. *Learning Road Distances for Combinatorial
Optimization*.**

Road-network data are obtained from
[OpenStreetMap](https://www.openstreetmap.org/) and the Québec extract
is distributed by
[Geofabrik](https://download.geofabrik.de/north-america/canada/quebec.html).

Routing distances are computed using [Project
OSRM](https://project-osrm.org/).
