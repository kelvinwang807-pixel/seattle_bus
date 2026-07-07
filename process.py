import pandas as pd
import numpy as np
import json
import torch


def process_features() -> pd.DataFrame:
    stop_times = pd.read_csv("data/stop_times.txt")
    trips = pd.read_csv("data/trips.txt")

    gtfs = stop_times.merge(
        trips[["trip_id", "route_id", "direction_id"]],
        on="trip_id"
    )

    with open("data/tripupdates_pb.json") as f:
        data = json.load(f)


    rows = []
    for entity in data["entity"]:
        trip = entity["trip_update"]["trip"]

        for stop in entity["trip_update"]["stop_time_update"]:
            if stop["schedule_relationship"] != "SKIPPED":
                rows.append({
                    "trip_id": int(trip["trip_id"]),
                    "stop_sequence": stop["stop_sequence"],
                    "stop_id": int(stop["stop_id"]),
                    "delay": stop["arrival"]["delay"],
                    "timestamp": stop["arrival"]["time"]
                })

    rt = pd.DataFrame(rows)

    df = gtfs.merge(
        rt,
        on=["trip_id", "stop_sequence"],
        how="inner"
    )

    def to_mins(t):
        h, m, s = map(int, t.split(":"))
        return h*60 + m

    df["arrival_min"] = df["arrival_time"].apply(to_mins)
    route_to_idx = {
        r:i
        for i,r in enumerate(df.route_id.unique())
    }

    df["route_idx"] = df.route_id.map(route_to_idx)


    return df

def to_tensor(df: pd.DataFrame) -> tuple[torch.tensor, torch.tensor]:
    xs= []
    ys = []
    window = 6
    features = [
        "arrival_min",
        "shape_dist_traveled",
        "delay",
        "stop_sequence",
        "route_idx",
        "direction_id"
    ]
    trip = df.sort_values("stop_sequence")

    values = trip[features].to_numpy()

    for i in range(len(values)-window):

        xs.append(values[i:i+window])

        ys.append(values[i+window][2])
    X = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)

    return X,y

df = process_features()
X,y = to_tensor(df)
print(X.shape)
print(y.shape)