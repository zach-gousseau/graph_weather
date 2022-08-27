import torch
import numpy as np
from graph_weather.models.forecast import GraphWeatherForecaster
from graph_weather.models.losses import NormalizedMSELoss

BATCH_SIZE = 1

lat_lons = []
for lat in range(-90, 90, 1):
    for lon in range(0, 360, 1):
        lat_lons.append((lat, lon))

graph_nodes = []
for lat in range(-90, 90, 2):
    for lon in range(0, 360, 2):
        graph_nodes.append((lat, lon))

lat_lons_to_graph_map = {i: int(np.floor(i / 4)) for i in range(len(lat_lons))}

distances = {}
for idx in range(len(graph_nodes)):
    distance = 0
    distances[idx] = {}
    for neighbor in range(idx + 1, idx + 4):
        if neighbor >= len(graph_nodes):
            continue
        
        distance += 1
        try:
            distances[idx][neighbor] = (0, distance)
        except:
            pass


print('num lat/lon nodes:', len(lat_lons))
print('num graph nodes:', len(graph_nodes))
print('total:', len(lat_lons) + len(graph_nodes))
print('nun edges:', len(lat_lons_to_graph_map))

model = GraphWeatherForecaster(
    lat_lons,
    graph_nodes,
    lat_lons_to_graph_map, 
    distances, 
    aux_dim=0,
    output_dim=1,
    target_variables=[7],
    predict_delta=True,
    )

features = torch.randn((BATCH_SIZE, len(lat_lons), 78))

out = model(features)
criterion = NormalizedMSELoss(lat_lons=lat_lons, feature_variance=torch.randn((78,)))
loss = criterion(out, features)

loss.backward()