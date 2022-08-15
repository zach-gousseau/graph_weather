"""Decoders to decode from the Processor graph to the original graph with updated values

In the original paper the decoder is described as

The Decoder maps back to physical data defined on a latitude/longitude grid. The underlying graph is
again bipartite, this time mapping icosahedron→lat/lon.
The inputs to the Decoder come from the Processor, plus a skip connection back to the original
state of the 78 atmospheric variables onthe latitude/longitude grid.
The output of the Decoder is the predicted 6-hour change in the 78 atmospheric variables,
which is then added to the initial state to produce the new state. We found 6 hours to be a good
balance between shorter time steps (simpler dynamics to model but more iterations required during
rollout) and longer time steps (fewer iterations required during rollout but modeling
more complex dynamics)

"""
import einops
import h3
import numpy as np
import torch
from torch_geometric.data import Data

from .graph_net_block import MLP, GraphProcessor
from .utils import haversine

def create_decoder_graph(lat_lons: list, graph_nodes: list, lat_lons_to_graph_map: dict, input_dim: int):

    # graph_to_lat_lons_map = {v: k for k, v in lat_lons_to_graph_map.items()}
    graph_to_lat_lons_map_from = list(lat_lons_to_graph_map.values())
    graph_to_lat_lons_map_to = list(lat_lons_to_graph_map.keys())

    # Indices for each lat/lon node and graph node
    # lat_lon_index = list(range(0, len(lat_lons)))
    # graph_node_index = list(range(len(lat_lon_index), len(lat_lon_index) + len(graph_nodes)))
    graph_node_index = list(range(0, len(graph_nodes)))
    lat_lon_index = list(range(len(graph_node_index), len(graph_node_index) + len(lat_lons)))
    lat_lon_map = {idx: lat_lon for idx, lat_lon in zip(lat_lon_index, lat_lons)}
    graph_node_map = {idx: node for idx, node in zip(graph_node_index, graph_nodes)}

    # Create bipartite edges (from lat/lon to graph)
    edge_sources = []
    edge_targets = []
    for from_i, to_j in zip(graph_to_lat_lons_map_from, graph_to_lat_lons_map_to):
        edge_sources.append(graph_node_index[from_i])
        edge_targets.append(lat_lon_index[to_j])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    
    # Create tensor for each lat/lon and graph node
    nodes = torch.zeros((len(lat_lons) + len(graph_nodes), input_dim), dtype=torch.float)

    # Get distances between each lat/lon node and its corresponding graph node\
    distances = []
    for from_idx, to_idx in zip(edge_sources, edge_targets):
        from_graph_node = graph_node_map[from_idx]
        to_lat_lon = lat_lon_map[to_idx]
        # distance = haversine(from_graph_node[0], from_graph_node[1], to_lat_lon[0], to_lat_lon[1])
        distance = h3.point_dist(from_graph_node, to_lat_lon, unit="rads")
        distances.append([np.sin(distance), np.cos(distance)])
        
    distances = torch.tensor(distances, dtype=torch.float)

    return Data(x=nodes, edge_index=edge_index, edge_attr=distances)


class AssimilatorDecoder(torch.nn.Module):
    """Assimilator graph module"""

    def __init__(
        self,
        lat_lons: list,
        graph_nodes: list,
        lat_lons_to_graph_map: dict, 
        input_dim: int = 256,
        output_dim: int = 78,
        output_edge_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
    ):
        """
        Decoder from latent graph to lat/lon graph for assimilation of observation

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        self.num_latlons = len(lat_lons)
        self.num_graph_nodes = len(graph_nodes)
        
        self.latlon_nodes = torch.zeros((len(lat_lons), input_dim), dtype=torch.float)
        self.graph = create_decoder_graph(lat_lons, graph_nodes, lat_lons_to_graph_map, input_dim)

        self.edge_encoder = MLP(2, output_edge_dim, hidden_dim_processor_edge, 2, mlp_norm_type)
        self.graph_processor = GraphProcessor(
            mp_iterations=1,
            in_dim_node=input_dim,
            in_dim_edge=output_edge_dim,
            hidden_dim_node=hidden_dim_processor_node,
            hidden_dim_edge=hidden_dim_processor_edge,
            hidden_layers_node=hidden_layers_processor_node,
            hidden_layers_edge=hidden_layers_processor_edge,
            norm_type=mlp_norm_type,
        )
        self.node_decoder = MLP(
            input_dim, output_dim, hidden_dim_decoder, hidden_layers_decoder, None
        )

    def forward(self, processor_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]

        Returns:
            Updated features for model
        """
        self.graph = self.graph.to(processor_features.device)
        edge_attr = self.edge_encoder(self.graph.edge_attr)  # Update attributes based on distance
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=batch_size)

        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(batch_size)
            ],
            dim=1,
        )

        # Readd nodes to match graph node number
        self.latlon_nodes = self.latlon_nodes.to(processor_features.device)
        features = einops.rearrange(processor_features, "(b n) f -> b n f", b=batch_size)
        features = torch.cat(
            [features, einops.repeat(self.latlon_nodes, "n f -> b n f", b=batch_size)], dim=1
        )
        features = einops.rearrange(features, "b n f -> (b n) f")

        out, _ = self.graph_processor(features, edge_index, edge_attr)  # Message Passing
        # Remove the h3 nodes now, only want the latlon ones
        out = self.node_decoder(out)  # Decode to 78 from 256
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        test, out = torch.split(out, [self.num_graph_nodes, self.num_latlons], dim=1)
        return out
