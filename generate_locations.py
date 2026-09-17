"""Generate 5,000 training and 2,000 testing locations in Montreal.

Run with: python generate_locations.py
Requires shapely. Outputs are dictionaries mapping IDs to (latitude, longitude).
"""

import pickle
import random
from pathlib import Path

from shapely.geometry import Point, Polygon


# Reproducible coordinate sampling configuration.
SEED = 42
N_TRAIN_LOCATIONS = 5_000
N_TEST_LOCATIONS = 2_000
N_LOCATIONS = N_TRAIN_LOCATIONS + N_TEST_LOCATIONS
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "dataset"
TRAIN_OUTPUT_FILE = DATA_DIR / "train_locations.pkl"
TEST_OUTPUT_FILE = DATA_DIR / "test_locations.pkl"
POLYGON_FILE = DATA_DIR / "MontrealServiceArea.pkl"


def main():
    with POLYGON_FILE.open("rb") as file:
        polygon_vertices = pickle.load(file)

    polygon_coords = [(longitude, latitude) for latitude, longitude in polygon_vertices]
    polygon = Polygon(polygon_coords)

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("The sampling polygon must have positive area.")

    print("Sampling coordinates inside the Montreal service area...")
    print(f"Polygon bounds: {polygon.bounds}")

    min_longitude, min_latitude, max_longitude, max_latitude = polygon.bounds
    rng = random.Random(SEED)
    locations = []
    seen_locations = set()

    while len(locations) < N_LOCATIONS:
        longitude = rng.uniform(min_longitude, max_longitude)
        latitude = rng.uniform(min_latitude, max_latitude)
        coordinate = (latitude, longitude)
        point = Point(longitude, latitude)

        if polygon.covers(point) and coordinate not in seen_locations:
            locations.append(coordinate)
            seen_locations.add(coordinate)

    # Separate, non-overlapping samples, each indexed from zero.
    train_locations = dict(enumerate(locations[:N_TRAIN_LOCATIONS]))
    test_locations = dict(enumerate(locations[N_TRAIN_LOCATIONS:]))

    print(f"Training locations: {len(train_locations)}")
    print(f"Test locations: {len(test_locations)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for output_file, sample in [
        (TRAIN_OUTPUT_FILE, train_locations),
        (TEST_OUTPUT_FILE, test_locations),
    ]:
        with output_file.open("wb") as file:
            pickle.dump(sample, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(sample)} locations to {output_file}")


if __name__ == "__main__":
    main()
