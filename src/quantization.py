# src/quantization.py
"""Quantization utilities: PTQ, QAT (LSQ), and Timestep-Aware mixed precision."""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.config import config, device
from src.data import get_dataloader
from diffusers import DDPMScheduler


# ===================== Schedulers (shared) =====================

scheduler_ddpm = DDPMScheduler(
    num_train_timesteps=config['train_timesteps'],
    beta_schedule='linear'
)


# ===================== Low-level Quantization Functions =====================

class BinarizeFn(torch.autograd.Function):
    """Straight-through estimator for sign."""
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


def binarize_weight(w, per_channel=True):
    """Return binarized weight and scaling factor."""
    if per_channel:
        alpha = w.abs().mean(dim=list(range(1, w.dim())), keepdim=True)
    else:
        alpha = w.abs().mean()
    return BinarizeFn.apply(w) * alpha, alpha


class UniformQuantizeFn(torch.autograd.Function):
    """STE for uniform quantization (symmetric)."""
    @staticmethod
    def forward(ctx, x, n_bits, scale):
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
        x_rounded = torch.round(x_clipped)
        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize_weight_uniform(w, n_bits, per_channel=True):
    if n_bits == 32:
        return w
    if n_bits == 1:
        return binarize_weight(w, per_channel)[0]
    if per_channel:
        scale = torch.amax(w.abs(), dim=tuple(range(1, w.dim())), keepdim=True) / (2 ** (n_bits - 1) - 1)
    else:
        scale = w.abs().max() / (2 ** (n_bits - 1) - 1)
    return UniformQuantizeFn.apply(w, n_bits, scale)


# ===================== Quantized Conv2d Variants =====================

class QuantConv2d(nn.Conv2d):
    """Conv2d with quantized weights (and optionally activations)."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_bits=32, act_bits=32, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.per_channel = True
        self.act_scale = None

    def forward(self, x, **kwargs):
        w_q = quantize_weight_uniform(self.weight, self.weight_bits, self.per_channel)
        if self.act_bits < 32 and self.act_scale is not None:
            x = UniformQuantizeFn.apply(x, self.act_bits, self.act_scale)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)


class LSQWeightQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, n_bits):
        ctx.save_for_backward(x, scale)
        ctx.n_bits = n_bits
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
        x_rounded = torch.round(x_clipped)
        return x_rounded * scale

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        n_bits = ctx.n_bits
        x_scaled = x / scale
        x_clipped = torch.clamp(x_scaled, -2 ** (n_bits - 1), 2 ** (n_bits - 1) - 1)
        x_rounded = torch.round(x_clipped)
        grad_x = grad_output.clone()
        grad_scale = (grad_output * (x_clipped - x_rounded)).sum().view(-1) / (2 ** (n_bits - 1) - 1)
        return grad_x, grad_scale, None


class QuantConv2dLSQ(QuantConv2d):
    """Quantized Conv2d with learnable step size (LSQ)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, **kwargs):
        w_q = LSQWeightQuantizeFn.apply(self.weight, self.scale, self.weight_bits)
        if self.act_bits < 32 and self.act_scale is not None:
            x = UniformQuantizeFn.apply(x, self.act_bits, self.act_scale)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)


class TimestepAwareQuantConv2d(QuantConv2d):
    """Conv2d that changes bit-width based on the timestep."""
    def forward(self, x, t=None, **kwargs):
        bits = self.weight_bits
        if t is not None:
            norm_t = t[0].item() / config['train_timesteps']
            if norm_t < 0.3:
                bits = 1
            elif norm_t < 0.7:
                bits = 4
            else:
                bits = 8
        w_q = quantize_weight_uniform(self.weight, bits, self.per_channel)
        if self.act_bits < 32 and self.act_scale is not None:
            x = UniformQuantizeFn.apply(x, self.act_bits, self.act_scale)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding)


# ===================== Module Replacement Helpers =====================

def replace_module_by_type(model, module_type, replacement_factory, condition=None):
    """Recursively replace modules of a given type."""
    for name, child in model.named_children():
        if isinstance(child, module_type) and (condition is None or condition(name, child)):
            setattr(model, name, replacement_factory(child))
        else:
            replace_module_by_type(child, module_type, replacement_factory, condition)


def condition_quant(name, module):
    """Only quantize layers whose name contains any string in config['quant_layers']."""
    for ql in config['quant_layers']:
        if ql in name:
            return True
    return False


# ===================== Apply PTQ =====================

def apply_ptq(model, weight_bits, act_bits=32, condition=None):
    """Replace Conv2d with QuantConv2d, copy weights, calibrate activations."""
    def replacement_factory(layer):
        new_layer = QuantConv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            weight_bits=weight_bits, act_bits=act_bits,
            stride=layer.stride, padding=layer.padding, bias=layer.bias is not None
        ).to(device)
        new_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        return new_layer

    replace_module_by_type(model, nn.Conv2d, replacement_factory, condition)

    # Calibrate activation scales
    if act_bits < 32:
        dataloader = get_dataloader()
        model.eval()
        activations = defaultdict(list)
        hooks = []

        def hook_fn(name):
            def fn(module, input, output):
                activations[name].append(output.detach().abs().max().item())
            return fn

        for name, module in model.named_modules():
            if isinstance(module, QuantConv2d):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= config['ptq_calib_steps']:
                    break
                x = x.to(device)
                t = torch.randint(0, config['train_timesteps'], (x.shape[0],), device=device).long()
                _ = model(x, t)
        for hook in hooks:
            hook.remove()
        for name, module in model.named_modules():
            if isinstance(module, QuantConv2d) and name in activations:
                max_act = max(activations[name])
                module.act_scale = max_act / (2 ** (act_bits - 1) - 1) if max_act > 0 else 1.0
    return model


# ===================== Apply QAT =====================

def apply_qat(model, weight_bits=1, act_bits=32, condition=None, epochs=5, lr=1e-5):
    """Replace Conv2d with QuantConv2dLSQ and fine-tune with gradient clipping."""
    def replacement_factory(layer):
        new_layer = QuantConv2dLSQ(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            weight_bits=weight_bits, act_bits=act_bits,
            stride=layer.stride, padding=layer.padding, bias=layer.bias is not None
        ).to(device)
        new_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        with torch.no_grad():
            std = layer.weight.data.std()
            new_layer.scale.data = torch.tensor(
                [2.0 * std / (2 ** (weight_bits - 1))]
            ).to(device)
        return new_layer

    replace_module_by_type(model, nn.Conv2d, replacement_factory, condition)

    dataloader = get_dataloader()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x0, _ in tqdm(dataloader, leave=False, desc=f"QAT epoch {epoch + 1}"):
            x0 = x0.to(device)
            bs = x0.shape[0]
            t = torch.randint(0, config['train_timesteps'], (bs,), device=device).long()
            noise = torch.randn_like(x0)
            xt = scheduler_ddpm.add_noise(x0, noise, t)
            pred = model(xt, t).sample
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"QAT Epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}")
        if torch.isnan(torch.tensor(avg_loss)):
            print("WARNING: NaN loss detected – stopping early.")
            break

    return model


# ===================== Apply Timestep-Aware =====================

def apply_timestep_aware(model, condition=None):
    """Replace QuantConv2d with TimestepAwareQuantConv2d."""
    def replacement_factory(layer):
        new_layer = TimestepAwareQuantConv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size,
            weight_bits=layer.weight_bits, act_bits=layer.act_bits,
            stride=layer.stride, padding=layer.padding, bias=layer.bias is not None
        )
        new_layer.weight.data = layer.weight.data.clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data.clone()
        new_layer.act_scale = getattr(layer, 'act_scale', None)
        return new_layer

    replace_module_by_type(model, QuantConv2d, replacement_factory, condition)
    return model
