"""Model for forecasting weather from NWP states"""
import torch
from huggingface_hub import PyTorchModelHubMixin

from . import Decoder, Encoder, Processor


class GraphWeatherForecaster(torch.nn.Module, PyTorchModelHubMixin):
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        graph_nodes: list,
        lat_lons_to_graph_map: dict, 
        distances: dict, 
        feature_dim: int = 78,
        output_dim: int = 78,
        target_variables: list = None,
        predict_delta: bool = False,
        aux_dim: int = 24,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        hidden_dim_decoder=128,
        hidden_layers_decoder=2,
        norm_type="LayerNorm",
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()

        if target_variables is None and output_dim != feature_dim:
            raise ValueError('If only predicting a subset of the input variables, you must specify the indices of the predicted\
                variable using target_variables.')
        elif target_variables is not None and output_dim != len(target_variables):
            raise ValueError(f'Size of target_variables {len(target_variables)} should match output_dim {output_dim}.')
        elif target_variables is None and output_dim == feature_dim:
            # Default
            self.target_variables = torch.tensor(list(range(output_dim))).type(torch.LongTensor)
        else:
            self.target_variables = torch.tensor(target_variables).type(torch.LongTensor)


        self.predict_delta = predict_delta

        self.encoder = Encoder(
            lat_lons,
            graph_nodes,
            lat_lons_to_graph_map, 
            distances, 
            input_dim=feature_dim+aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
        )
        self.processor = Processor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
        )
        self.decoder = Decoder(
            lat_lons,
            graph_nodes,
            lat_lons_to_graph_map, 
            input_dim=node_dim,
            output_dim=output_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        x, edge_idx, edge_attr = self.encoder(features)
        x = self.processor(x, edge_idx, edge_attr)

        if self.predict_delta:
            start_features = torch.zeros_like(features[..., self.target_variables])
        else:
            start_features = features[..., self.target_variables]

        x = self.decoder(x, start_features, features.shape[0])
        return x
