import math

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import init

from deepstruct.graph import LayeredGraph


class BaseRecurrentLayer(nn.Module):
    """
    Base class for recurrent layers which can be masked and have an additional unfold() operation.

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self, input_size: int, hidden_size: int, batch_first: bool = False, **kwargs
    ):
        super().__init__()
        assert input_size > 0
        assert hidden_size > 0

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._batch_first = True if batch_first else False

        self._initialize_parameters()
        self.reset_parameters()

    def _initialize_parameters(self):
        input_size = self._input_size
        hidden_size = self._hidden_size

        self._weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self._weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(hidden_size))
        self._bias_hh = Parameter(torch.randn(hidden_size))

        self.register_buffer(
            "_mask_i2h", torch.ones((hidden_size, input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((hidden_size, hidden_size), dtype=torch.bool)
        )

    def reset_parameters(self, keep_mask=False):
        # TODO should weight initialization be done here?
        stdv = 1.0 / math.sqrt(self._hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def set_i2h_mask(self, mask):
        self._mask_i2h = Variable(mask)

    def set_h2h_mask(self, mask):
        self._mask_h2h = Variable(mask)

    def unfold(self, input, hx):
        in_dim = 1 if self._batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []

        for i in range(n_seq):
            seq = input[:, i, :] if self._batch_first else input[i]
            hx = self.forward(seq, hx)
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)

    def extra_repr(self):
        s = "in_features={_input_size}, out_features={_hidden_size}"

        if self._batch_first:
            s += ", batch_first={_batch_first}"

        return s.format(**self.__dict__)


class MaskedRecurrentLayer(BaseRecurrentLayer):
    """
    Base class for layer initialization for Vanilla RNN.

    Args:
        nonlinearity: Can be a usual torch.nn.ReLU() or torch.nn.Tanh() or torch.nn.LogSigmoid() ..
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = False,
        nonlinearity=torch.nn.Tanh(),
    ):
        super().__init__(input_size, hidden_size, batch_first)
        assert callable(nonlinearity)
        self._nonlinearity = nonlinearity

    def forward(self, input, hx):
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        return self._nonlinearity(igate + hgate)


class MaskedGRULayer(BaseRecurrentLayer):
    """
    Base class for layer initialization for GRU.
    """

    def _initialize_parameters(self):
        input_size = self._input_size
        hidden_size = self._hidden_size

        gate_size = 3 * hidden_size

        self._weight_ih = Parameter(torch.randn(gate_size, input_size))
        self._weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(gate_size))
        self._bias_hh = Parameter(torch.randn(gate_size))

        self.register_buffer(
            "_mask_i2h", torch.ones((gate_size, input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((gate_size, hidden_size), dtype=torch.bool)
        )

    def forward(self, input, hx):
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        i_reset, i_input, i_new = igate.chunk(3, 1)
        h_reset, h_input, h_new = hgate.chunk(3, 1)

        reset_gate = torch.sigmoid(i_reset + h_reset)
        input_gate = torch.sigmoid(i_input + h_input)
        new_gate = torch.tanh(i_new + reset_gate * h_new)

        hx = new_gate + input_gate * (hx - new_gate)
        return hx


class MaskedLSTMLayer(BaseRecurrentLayer):
    """
    Base class for layer initialization for LSTM.
    """

    def _initialize_parameters(self):
        input_size = self._input_size
        hidden_size = self._hidden_size

        gate_size = 4 * hidden_size

        self._weight_ih = Parameter(torch.randn(gate_size, input_size))
        self._weight_hh = Parameter(torch.randn(gate_size, hidden_size))
        self._bias_ih = Parameter(torch.randn(gate_size))
        self._bias_hh = Parameter(torch.randn(gate_size))
        self.reset_parameters()

        self.register_buffer(
            "_mask_i2h", torch.ones((gate_size, input_size), dtype=torch.bool)
        )
        self.register_buffer(
            "_mask_h2h", torch.ones((gate_size, hidden_size), dtype=torch.bool)
        )

    def forward(self, input, hx):
        hx, cx = hx
        igate = torch.mm(input, (self._weight_ih * self._mask_i2h).t()) + self._bias_ih
        hgate = torch.mm(hx, (self._weight_hh * self._mask_h2h).t()) + self._bias_hh

        gates = igate + hgate

        input_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        cx = (forget_gate * cx) + (input_gate * cell_gate)
        hx = out_gate * torch.tanh(cx)
        return hx, cx

    def unfold(self, input, hx):
        in_dim = 1 if self._batch_first else 0
        n_seq = input.size(in_dim)
        outputs = []
        cx = hx.clone()

        for i in range(n_seq):
            seq = input[:, i, :] if self._batch_first else input[i]
            hx, cx = self.forward(seq, (hx, cx))
            outputs.append(hx.unsqueeze(in_dim))

        return torch.cat(outputs, dim=in_dim)


class BaseMaskModule(nn.Module):
    def apply_mask(self, threshold=0.0, i2h=False, h2h=False):
        """
        :param threshold: Amount of pruning to apply. Default '0.0'
        :param i2h: If True, then Input-to-Hidden layers will be masked. Default 'False'
        :param h2h: If True, then Hidden-to-Hidden layers will be masked. Default 'False'
        :type threshold: float
        :type i2h: bool
        :type h2h: bool
        """
        if not i2h and not h2h:
            return

        masks = self.__get_masks(threshold, i2h, h2h)
        for lay_idx, layer in enumerate(self._recurrent_layers):
            if i2h:
                layer.set_i2h_mask(masks[lay_idx][0])
            if h2h:
                layer.set_h2h_mask(masks[lay_idx][-1])

    def __get_masks(self, threshold, i2h, h2h):
        key = "" if i2h and h2h else "ih" if i2h else "hh" if h2h else None

        masks = {}
        for lay_idx, layer in enumerate(self._recurrent_layers):
            masks[lay_idx] = []
            for param, data in layer.named_parameters():
                if "bias" not in param and key in param:
                    mask = torch.ones(data.shape, dtype=torch.bool, device=data.device)
                    mask[torch.where(abs(data) < threshold)] = False
                    masks[lay_idx].append(mask)

        return masks


class MaskedDeepRNN(BaseMaskModule):
    """
    A deep Vanilla RNN model with maskable layers to allow for sparsity.

    Args:
        input_size: The number of expected features in the input
        hidden_layers: A list specifying number of expected features in each hidden layer
            (E.g, hidden_layers=[50, 50] specifies model consisting two hidden layers with 50 features each)
        build_recurrent_layer: Type of recurrent layer to use
            (MaskedRecurrentLayer or MaskedGRULayer or MaskedLSTMLayer)
        nonlinearity: Can be a usual non-linearity such as torch.nn.ReLU() or torch.nn.Tanh()
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
    """

    def __init__(
        self,
        input_size,
        hidden_layers: list,
        build_recurrent_layer=MaskedRecurrentLayer,
        nonlinearity=torch.nn.Tanh(),
        batch_first=False,
    ):
        super(MaskedDeepRNN, self).__init__()

        assert callable(nonlinearity)

        self._input_size = input_size
        self._hidden_layers = hidden_layers
        self._batch_first = batch_first

        layer_list = []
        for lay, hidden_size in enumerate(hidden_layers):
            input_size = input_size if lay == 0 else hidden_layers[lay - 1]
            layer_list.append(
                build_recurrent_layer(
                    input_size, hidden_size, batch_first, nonlinearity=nonlinearity
                )
            )
            # MaskedRecurrentLayer(input_size, hidden_size, nonlinearity, batch_first)
        self._recurrent_layers = nn.ModuleList(layer_list)

    def forward(self, input):
        batch_size = input.size(0) if self._batch_first else input.size(1)

        for layer, hidden_size in zip(self._recurrent_layers, self._hidden_layers):
            # TODO initialization of first hidden states should be configurable from outside
            hx = torch.zeros(
                batch_size, hidden_size, dtype=input.dtype, device=input.device
            )
            input = layer.unfold(input, hx)

        output = input[:, -1, :] if self._batch_first else input[-1]
        return output


class MaskedDeepRDAN(nn.Module):
    def __init__(
        self,
        input_size,
        structure: LayeredGraph,
        build_recurrent_layer=MaskedRecurrentLayer,
        nonlinearity=torch.nn.Tanh(),
        batch_first=False,
    ):
        super(MaskedDeepRDAN, self).__init__()

        self._input_size = input_size
        self._structure = structure
        self._batch_first = batch_first

        gates = (
            1
            if build_recurrent_layer == "MaskedRecurrentLayer"
            else 3
            if build_recurrent_layer == "MaskedGRULayer"
            else 4
        )

        assert callable(nonlinearity)
        assert structure.num_layers > 0

        layer_list = []
        for layer_idx, layer in enumerate(structure.layers):
            input_size = input_size if layer_idx == 0 else structure.get_layer_size(layer_idx - 1)
            layer_list.append(
                build_recurrent_layer(
                    input_size,
                    structure.get_layer_size(layer_idx),
                    batch_first,
                    nonlinearity=nonlinearity,
                )
            )

        self._recurrent_layers = nn.ModuleList(layer_list)

        for layer_idx, layer in zip(structure.layers[1:], self._recurrent_layers[1:]):
            mask = torch.zeros(
                structure.get_layer_size(layer_idx),
                structure.get_layer_size(layer_idx - 1),
            )
            for source_idx, source_vertex in enumerate(
                structure.get_vertices(layer_idx - 1)
            ):
                for target_idx, target_vertex in enumerate(
                    structure.get_vertices(layer_idx)
                ):
                    if structure.has_edge(source_vertex, target_vertex):
                        mask[target_idx][source_idx] = 1
            mask = np.repeat(mask, gates, 0)
            layer.set_i2h_mask(mask)

        skip_layers = []
        self._skip_targets = {}
        for target_layer in structure.layers[2:]:
            target_size = structure.get_layer_size(target_layer)
            for distant_source_layer in structure.layers[: target_layer - 1]:
                if structure.layer_connected(distant_source_layer, target_layer):
                    if target_layer not in self._skip_targets:
                        self._skip_targets[target_layer] = []

                    skip_layer = build_recurrent_layer(
                        structure.get_layer_size(distant_source_layer),
                        target_size,
                        batch_first,
                        nonlinearity=nonlinearity,
                    )
                    mask = torch.zeros(
                        structure.get_layer_size(target_layer),
                        structure.get_layer_size(distant_source_layer),
                    )
                    for source_idx, source_vertex in enumerate(
                        structure.get_vertices(distant_source_layer)
                    ):
                        for target_idx, target_vertex in enumerate(
                            structure.get_vertices(target_layer)
                        ):
                            if structure.has_edge(source_vertex, target_vertex):
                                mask[target_idx][source_idx] = 1
                    mask = np.repeat(mask, gates, 0)
                    skip_layer.set_i2h_mask(mask)

                    skip_layers.append(skip_layer)
                    self._skip_targets[target_layer].append(
                        {"layer": skip_layer, "source": distant_source_layer}
                    )
        self.skip_layers = nn.ModuleList(skip_layers)

    def forward(self, input):
        batch_size = input.size(0) if self._batch_first else input.size(1)

        layer_results = dict()
        for layer, layer_idx in zip(self._recurrent_layers, self._structure.layers):
            hx = torch.zeros(
                batch_size,
                self._structure.get_layer_size(layer_idx),
                dtype=input.dtype,
                device=input.device,
            )
            input = layer.unfold(input, hx)

            if layer_idx in self._skip_targets:
                for skip_target in self._skip_targets[layer_idx]:
                    source_layer = skip_target["layer"]
                    source_idx = skip_target["source"]

                    input += source_layer.unfold(layer_results[source_idx], hx)

            layer_results[layer_idx] = input

        output = input[:, -1, :] if self._batch_first else input[-1]
        return output
