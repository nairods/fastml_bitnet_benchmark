from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs quantization of weights and activations

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
        frac_bits (int, optional): Fractional bits for Q1.n fixed-point quantization [1, 16]. Default to None, using floating-point scales.
        quant_export (bool, optional): If True, state_dict exports int8 weights + scales instead of float weights. Defaults to False.
        quantize_bias (bool, optional): If True, quantize the bias when frac_bits is set. Defaults to True.
        beta_quant (str, optional): Beta quantization mode: None, "fixed", or "power2". Defaults to None.
        beta_shift_min (int, optional): Minimum allowed beta shift for power2 beta quantization. Defaults to None.
        beta_shift_max (int, optional): Maximum allowed beta shift for power2 beta quantization. Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
        frac_bits: Optional[int] = None,
        quant_export: bool = False,
        quantize_bias: bool = True,
        beta_quant: Optional[str] = None,
        beta_shift_min: Optional[int] = None,
        beta_shift_max: Optional[int] = None,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-8
        self.device = self.weight.device
        self.dtype = self.weight.dtype

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1) - 1.0
        self.beta = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.beta_shift = None
        self.gamma = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Check for fixed-point quantization
        if frac_bits is not None:
            # Ensure the user provided an integer
            if not isinstance(frac_bits, int):
                raise TypeError(
                    f"frac_bits must be an integer, got {type(frac_bits).__name__}"
                )
            # Ensure the value is within a meaningful hardware range
            if not (1 <= frac_bits <= 16):
                raise ValueError(f"frac_bits must be in [1, 16], got {frac_bits}")
            # Store fractional bits for fixed-point quantization
            self.frac_bits = frac_bits
            self.scale = 2.0**-self.frac_bits
            self.max_val = (2**self.frac_bits - 1) * self.scale
            # # Optional: conservative quantization mode against overflow
            # max_val = ((2 ** (self.frac_bits - 1)) - 1) * scale
        else:
            # No fixed-point quantization requested
            self.frac_bits = None

        if beta_quant is not None and beta_quant not in ("fixed", "power2"):
            raise ValueError(
                f'beta_quant must be one of None, "fixed", or "power2", got {beta_quant!r}'
            )
        if beta_quant == "fixed" and self.frac_bits is None:
            raise ValueError('beta_quant="fixed" requires frac_bits to be set')
        if beta_quant is None and self.frac_bits is not None:
            beta_quant = "fixed"
        for name, value in (
            ("beta_shift_min", beta_shift_min),
            ("beta_shift_max", beta_shift_max),
        ):
            if value is not None and not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
        if (
            beta_shift_min is not None
            and beta_shift_max is not None
            and beta_shift_min > beta_shift_max
        ):
            raise ValueError(
                f"beta_shift_min must be <= beta_shift_max, got "
                f"{beta_shift_min} > {beta_shift_max}"
            )
        self.beta_quant = beta_quant
        self.beta_shift_min = beta_shift_min
        self.beta_shift_max = beta_shift_max
        self.quantize_bias = quantize_bias

        # Flag to control whether to export quantized values in state_dict
        self.quant_export = quant_export

        # Placeholders for quantized parameters to be saved during export
        self.int_weight = None
        self.bias_q = None

    def quantize_fixed(self, x: Tensor) -> Tensor:
        """Symmetric fixed-point quantization in range [-1, max_val]."""
        q = torch.round(x / self.scale) * self.scale
        return q.clamp(-self.max_val - self.scale, self.max_val)

    def quantize_beta(self, beta: Tensor) -> Tensor:
        """Quantize beta according to the configured beta quantization mode."""
        self.beta_shift = None
        if self.beta_quant == "fixed":
            return self.quantize_fixed(beta)
        if self.beta_quant == "power2":
            self.beta_shift = torch.round(torch.log2(beta.clamp(min=self.eps)))
            if self.beta_shift_min is not None or self.beta_shift_max is not None:
                self.beta_shift = self.beta_shift.clamp(
                    min=self.beta_shift_min,
                    max=self.beta_shift_max,
                )
            return torch.pow(
                torch.tensor(2.0, device=beta.device, dtype=beta.dtype),
                self.beta_shift,
            )
        return beta

    def quantize_weights(self, w: Tensor) -> Tensor:
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        alpha = w.mean()
        self.beta = w.abs().mean().clamp_(min=self.eps)
        self.beta = self.quantize_beta(self.beta)
        quantized_weight = torch.sign(w - alpha)

        # Update quantized weight for export
        self.int_weight = quantized_weight.to(torch.int8)

        return quantized_weight * self.beta

    def quantize_activations(self, x: Tensor) -> Tensor:
        """
        Quantizes the activations of the layer.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        self.gamma = self.Q_b / x.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=self.eps
        )
        quantized_x = (x * self.gamma).round().clamp_(-(self.Q_b + 1), self.Q_b)

        return quantized_x / self.gamma

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # weight tensor with shape (in_features, out_features)
        w = self.weight

        # Quantize weights
        w_quant = w + (self.quantize_weights(w) - w).detach()

        # Quantize input
        x_quant = x + (self.quantize_activations(x) - x).detach()

        # Quantize bias if fixed-point quantization is enabled and bias exists
        if self.quantize_bias and self.frac_bits is not None and self.bias is not None:
            self.bias_q = self.quantize_fixed(self.bias)
            b_quant = self.bias + (self.bias_q - self.bias).detach()
        else:
            self.bias_q = None
            b_quant = self.bias

        # Perform linear transformation
        output = F.linear(x_quant, w_quant, b_quant)

        # Return dequantized output
        return output

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Overrides PyTorch's default parameter serialization when quant_export=True.

        Args:
            destination (dict): Shared state dict passed by PyTorch recursion
            prefix (str): Module name prefix (e.g., "layer1.")
            keep_vars (bool): If True, preserve autograd graph; if False, detach tensors before saving.

        Saved format:
            quant_export=True:  int_weight (int8), beta_scale or beta_shift, bias (float)
            quant_export=False: weight (float), bias (float)

        Compatible with: torch.save(model.state_dict(), path)
        """

        if self.quant_export and self.int_weight is not None:
            # Store tensors respecting keep_vars (autograd preservation vs detached copies)
            destination[prefix + "int_weight"] = (
                self.int_weight if keep_vars else self.int_weight.detach()
            )
            if self.beta_quant == "power2":
                beta_shift = self.beta_shift
                if beta_shift is None:
                    beta_shift = torch.round(torch.log2(self.beta.clamp(min=self.eps)))
                beta_shift = beta_shift.to(torch.int32)
                destination[prefix + "beta_shift"] = (
                    beta_shift if keep_vars else beta_shift.detach().cpu()
                )
            else:
                destination[prefix + "beta_scale"] = (
                    self.beta if keep_vars else self.beta.detach().cpu()
                )
            if self.bias is not None:
                # Use quantized bias if fixed-point quantization is enabled, otherwise use float bias
                bias = self.bias_q if self.bias_q is not None else self.bias
                destination[prefix + "bias"] = (
                    bias if keep_vars else bias.detach().cpu()
                )
        else:
            # Default behavior: save float weight/bias
            super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Custom deserialization for quantized state dictionary loading.

        Supported formats:
            Quantized: int_weight (int8), beta_scale (float) or beta_shift (int), bias (float)→ reconstructs float weight
            Float fallback: weight (float), bias (float)

        Args:
            state_dict (dict): Checkpoint state dict
            prefix (str): Module name prefix
            local_metadata (dict): Module metadata
            strict (bool): Strict loading mode
            missing_keys (list): List of missing keys
            unexpected_keys (list): List of unexpected keys
            error_msgs (list): Error message list

        Compatible with: model.load_state_dict(torch.load(path, map_location=device), strict=False)
        """

        # Quantized key names
        key_wq = prefix + "int_weight"
        key_beta = prefix + "beta_scale"
        key_beta_shift = prefix + "beta_shift"

        # Standard float key names
        key_w = prefix + "weight"
        key_bias = prefix + "bias"

        # Look for quantized keys
        if key_wq in state_dict and (
            key_beta in state_dict or key_beta_shift in state_dict
        ):

            if key_beta_shift in state_dict:
                beta = torch.pow(
                    torch.tensor(
                        2.0,
                        device=state_dict[key_wq].device,
                        dtype=torch.float32,
                    ),
                    state_dict[key_beta_shift].float(),
                )
            else:
                beta = state_dict[key_beta]

            # Reconstruct float weight = int8 * beta
            W_float = state_dict[key_wq].float() * beta

            # Copy into real module weight tensor
            self.weight.data.copy_(W_float.to(self.weight.device))
            if self.bias is not None and key_bias in state_dict:
                self.bias.data.copy_(state_dict[key_bias].to(self.bias.device))

            # Remove loaded keys from missing_keys to prevent false missing key errors
            if key_w in missing_keys:
                missing_keys.remove(key_w)
            if self.bias is not None and key_bias in missing_keys:
                missing_keys.remove(key_bias)

            return

        # If quantized keys are not found, fallback to default loading (float)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BitLinear158b(BitLinear):
    """
    BitLinear158b layer allowing for tertiar weights (-1,0,1). Rest is keeped
    as in BitLinear

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
        frac_bits (int, optional): Fractional bits for Q1.n fixed-point quantization [1, 16]. Default to None, using floating-point scales.
        quant_export (bool, optional): If True, state_dict exports int8 weights + scales instead of float weights. Defaults to False.
        quantize_bias (bool, optional): If True, quantize the bias when frac_bits is set. Defaults to True.
        beta_quant (str, optional): Beta quantization mode: None, "fixed", or "power2". Defaults to None.
        beta_shift_min (int, optional): Minimum allowed beta shift for power2 beta quantization. Defaults to None.
        beta_shift_max (int, optional): Maximum allowed beta shift for power2 beta quantization. Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
        frac_bits: Optional[int] = None,
        quant_export: bool = False,
        quantize_bias: bool = True,
        beta_quant: Optional[str] = None,
        beta_shift_min: Optional[int] = None,
        beta_shift_max: Optional[int] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            b,
            frac_bits,
            quant_export,
            quantize_bias,
            beta_quant,
            beta_shift_min,
            beta_shift_max,
        )

    def quantize_weights(self, w: Tensor):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = w.abs().mean().clamp_(min=self.eps)
        self.beta = self.quantize_beta(self.beta)
        quantized_weight = (w / self.beta).round().clamp_(-1, 1)

        # Update quantized weight for export
        self.int_weight = quantized_weight.to(torch.int8)

        return quantized_weight * self.beta
