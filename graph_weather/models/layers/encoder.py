"""Encoders to encode from the input graph to the latent graph

In the original paper the encoder is described as

The Encoder maps from physical data defined on a latitude/longitude grid to abstract latent features
defined on an icosahedron grid.  The Encoder GNN uses a bipartite graph(lat/lon→icosahedron) with
edges only between nodes in the lat/lon grid and nodes in the icosahedron grid. Put another way,
spatial and channel information in the local neighborhood of each icosahedron node is
gathered using connections to nearby lat/lon nodes.

The initial node features are the 78 atmospheric variables described in Section 2.1, plus solar
radiation, orography, land-sea mask, the day-of-year,sin(lat),cos(lat),sin(lon), and cos(lon).
The initial edge features are the positions of the lat/lon nodes connected to each icosahedron node.
These positions are provided in a local coordinate system that is defined relative
to each icosahedron node.

In further notes, they notice that there is some hexagon instabilities in long rollouts
One possible way to change that is to do the additative noise as in the original MeshGraphNet
or mildly randomize graph connectivity in encoder, as a kind of edge Dropout



"""
from typing import Tuple

import einops
import h3
import numpy as np
import torch
from torch_geometric.data import Data

from .graph_net_block import MLP, GraphProcessor
from .utils import haversine




def create_encoder_graph(lat_lons: list, graph_nodes: list, lat_lons_to_graph_map: dict, input_dim: int):

    # Indices for each lat/lon node and graph node
    lat_lon_index = list(range(0, len(lat_lons)))
    graph_node_index = list(range(len(lat_lon_index), len(lat_lon_index) + len(graph_nodes)))
    lat_lon_map = {idx: lat_lon for idx, lat_lon in zip(lat_lon_index, lat_lons)}
    graph_node_map = {idx: node for idx, node in zip(graph_node_index, graph_nodes)}

    # Create bipartite edges (from lat/lon to graph)
    edge_sources = []
    edge_targets = []
    for from_i, to_j in lat_lons_to_graph_map.items():
        edge_sources.append(lat_lon_index[from_i])
        edge_targets.append(graph_node_index[to_j])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    
    # Create tensor for each lat/lon and graph node
    nodes = torch.zeros((len(lat_lons) + len(graph_nodes), input_dim), dtype=torch.float)

    # Get distances between each lat/lon node and its corresponding graph node\
    distances = []
    for from_idx, to_idx in zip(edge_sources, edge_targets):
        from_lat_lon = lat_lon_map[from_idx]
        to_graph_node = graph_node_map[to_idx]
        # distance = haversine(from_lat_lon[0], from_lat_lon[1], to_graph_node[0], to_graph_node[1])
        distance = h3.point_dist(from_lat_lon, to_graph_node, unit="rads")
        distances.append([np.sin(distance), np.cos(distance)])
        
    distances = torch.tensor(distances, dtype=torch.float)

    return Data(x=nodes, edge_index=edge_index, edge_attr=distances)

def create_latent_graph(graph_nodes: list, distances: dict):
    """
    Copies over and generates a Data object for the processor to use

    Returns:
        The connectivity and edge attributes for the latent graph
    """
    edge_sources = []
    edge_targets = []
    edge_attrs = []
    for graph_node_idx in range(len(graph_nodes)):
        for neighbor in distances[graph_node_idx]:
            edge_sources.append(graph_node_idx)
            edge_targets.append(neighbor)
            edge_attrs.append(distances[graph_node_idx][neighbor])

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(edge_index=edge_index, edge_attr=edge_attrs)



class Encoder(torch.nn.Module):
    """Encoder graph model"""

    def __init__(
        self,
        lat_lons: list,
        graph_nodes: list,
        lat_lons_to_graph_map: dict, 
        distances: dict, 
        # resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 256,
        output_edge_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
    ):
        """
        Encode the lat/lon data inot the isohedron graph

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
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_latlons = len(lat_lons)

        self.graph = create_encoder_graph(lat_lons, graph_nodes, lat_lons_to_graph_map, input_dim)
        self.latent_graph = create_latent_graph(graph_nodes, distances)
        self.h3_nodes = torch.zeros((len(graph_nodes), input_dim), dtype=torch.float)  # Tensor w/ shape: (num_graph_nodes, num_features)

        # Output graph

        self.node_encoder = MLP(
            input_dim,
            output_dim,
            hidden_dim_processor_node,
            hidden_layers_processor_node,
            mlp_norm_type,
        )
        self.edge_encoder = MLP(
            2,
            output_edge_dim,
            hidden_dim_processor_edge,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )
        self.latent_edge_encoder = MLP(
            2,
            output_edge_dim,
            hidden_dim_processor_edge,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )
        self.graph_processor = GraphProcessor(
            1,
            output_dim,
            output_edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:
            Torch tensors of node features, latent graph edge index, and latent edge attributes
        """
        batch_size = features.shape[0]
        self.h3_nodes = self.h3_nodes.to(features.device)
        self.graph = self.graph.to(features.device)
        self.latent_graph = self.latent_graph.to(features.device)

        features = torch.cat(
            [features, einops.repeat(self.h3_nodes, "n f -> b n f", b=batch_size)], dim=1
        )
        # Cat with the h3 nodes to have correct amount of nodes, and in right order
        features = einops.rearrange(features, "b n f -> (b n) f")
        out = self.node_encoder(features)  # Encode to 256 from 78
        edge_attr = self.edge_encoder(self.graph.edge_attr)  # Update attributes based on distance
        # Copy attributes batch times
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=batch_size)
        # Expand edge index correct number of times while adding the proper number to the edge index
        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(batch_size)
            ],
            dim=1,
        )
        out, _ = self.graph_processor(out, edge_index, edge_attr)  # Message Passing
        # Remove the extra nodes (lat/lon) from the output
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        _, out = torch.split(out, [self.num_latlons, self.h3_nodes.shape[0]], dim=1)
        out = einops.rearrange(out, "b n f -> (b n) f")
        i = 1

        return (
            out,
            torch.cat(
                [
                    self.latent_graph.edge_index + i * torch.max(self.latent_graph.edge_index) + i
                    for i in range(batch_size)
                ],
                dim=1,
            ),
            self.latent_edge_encoder(
                einops.repeat(self.latent_graph.edge_attr, "e f -> (repeat e) f", repeat=batch_size)
            ),
        )  # New graph

    def create_latent_graph(self) -> Data:
        """
        Copies over and generates a Data object for the processor to use

        Returns:
            The connectivity and edge attributes for the latent graph
        """
        # Get connectivity of the graph
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for h3_index in self.base_h3_grid:
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                distance = h3.point_dist(h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads")
                edge_attrs.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.base_h3_map[h3_index])
                edge_targets.append(self.base_h3_map[h])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)
