"""CUDA pair-force kernel reused across sequences; imported only on CUDA."""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from starling.minimizer.forcefield import MIN_DISTANCE, MpipiGG
from starling.minimizer.fire import ForceFunction
from starling.minimizer.parameters import BOND_FORCE_CONSTANT, BOND_LENGTH
from starling.minimizer.restraints import DistanceRestraints


# n is argument 9: changing sequence length must not specialize the kernel.
@triton.jit(do_not_specialize=[9])
def _pair_forces(
    x, sigma, cutoff, scale, qq, reference, index, constants, output, n, BLOCK: tl.constexpr, ACC64: tl.constexpr
):
    """Stream partner tiles and reduce forces without dense pair temporaries."""
    bead = tl.program_id(0)
    frame = tl.program_id(1)
    ref_frame = tl.load(index + frame)
    xyz = x + (frame * n + bead) * 3
    xi, yi, zi = tl.load(xyz), tl.load(xyz + 1), tl.load(xyz + 2)
    kappa = tl.load(constants)
    dh_cutoff = tl.load(constants + 1)
    stiffness = tl.load(constants + 2)
    tolerance = tl.load(constants + 3)
    min_separation = tl.load(constants + 4).to(tl.int32)
    bond_k = tl.load(constants + 5)
    bond_length = tl.load(constants + 6)
    min_distance = tl.load(constants + 7)
    accumulation = tl.float64 if ACC64 else tl.float32
    fx = tl.full((BLOCK,), 0.0, accumulation)
    fy = tl.full((BLOCK,), 0.0, accumulation)
    fz = tl.full((BLOCK,), 0.0, accumulation)
    for start in range(0, n, BLOCK):
        partner = start + tl.arange(0, BLOCK)
        valid = partner < n
        separation = tl.abs(bead - partner)
        other = x + (frame * n + partner) * 3
        dx = xi - tl.load(other, valid, 0.0)
        dy = yi - tl.load(other + 1, valid, 0.0)
        dz = zi - tl.load(other + 2, valid, 0.0)
        r = tl.maximum(libdevice.sqrt_rn(dx * dx + dy * dy + dz * dz), min_distance)
        inv_r = 1.0 / r
        pair = bead * n + partner
        ratio = tl.load(sigma + pair, valid, 0.0) * inv_r
        a = ratio * ratio
        a = a * a
        # mu=2, nu=1: C=(R/sigma)^4=3^4=81 and 1+2*nu*C=163.
        b = 81.0 * a
        derivative = (tl.load(scale + pair, valid, 0.0) * inv_r) * (b - 1.0) * (a * (3.0 * b - 163.0))
        derivative = tl.where(valid & (separation >= 2) & (r < tl.load(cutoff + pair, valid, 0.0)), derivative, 0.0)
        charge_product = tl.load(qq + pair, valid, 0.0)
        charged = valid & (charge_product != 0.0) & (r <= dh_cutoff)
        if tl.sum(charged.to(tl.int32), 0) > 0:
            u = charge_product * libdevice.exp(-kappa * r) * inv_r
            derivative += tl.where(charged, -u * (kappa + inv_r), 0.0)
        target = tl.load(reference + (ref_frame * n + bead) * n + partner, valid, 0.0)
        deviation = r - target
        excess = deviation - tl.minimum(tl.maximum(deviation, -tolerance), tolerance)
        derivative += tl.where(valid & (separation >= min_separation), stiffness * excess, 0.0)
        derivative += tl.where(valid & (separation == 1), bond_k * (r - bond_length), 0.0)
        g = tl.where(valid & (separation != 0), -derivative * inv_r, 0.0)
        fx += (g * dx).to(accumulation)
        fy += (g * dy).to(accumulation)
        fz += (g * dz).to(accumulation)
    result = output + (frame * n + bead) * 3
    tl.store(result, tl.sum(fx, 0))
    tl.store(result + 1, tl.sum(fy, 0))
    tl.store(result + 2, tl.sum(fz, 0))


def fused_forces(forcefield: MpipiGG, restraints: DistanceRestraints) -> ForceFunction:
    """Bind float32 Mpipi-GG parameters to the reusable CUDA kernel."""
    if forcefield.device.type != "cuda" or forcefield.dtype != torch.float32:
        raise ValueError("Fused forces require CUDA float32")
    if not isinstance(forcefield._two_mu, float) or not isinstance(forcefield._two_nu, float):
        raise ValueError("Fused forces require uniform Mpipi-GG exponents")
    if forcefield._two_mu != 4.0 or forcefield._two_nu != 2.0 or forcefield._cutoff_power != 81.0:
        raise ValueError("Fused forces require Mpipi-GG mu=2, nu=1")
    parameters = tuple(
        t.contiguous()
        for t in (
            forcefield._sigma,
            forcefield._wf_cutoff,
            forcefield._wf_derivative_scale,
            forcefield._qq,
            restraints._reference,
        )
    )
    constants = torch.tensor(
        [
            forcefield._kappa,
            forcefield.coulomb_cutoff,
            restraints.pair_force_constant,
            restraints.tolerance,
            restraints.min_separation,
            BOND_FORCE_CONSTANT,
            BOND_LENGTH,
            MIN_DISTANCE,
        ],
        dtype=torch.float32,
        device=forcefield.device,
    )

    def force(coordinates: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if coordinates.dtype != torch.float32 or coordinates.device != parameters[0].device:
            raise ValueError("Fused forces require float32 coordinates on the force field's CUDA device")
        if coordinates.ndim != 3 or coordinates.shape[1:] != (forcefield.n_residues, 3):
            raise ValueError(f"coordinates must have shape (batch, {forcefield.n_residues}, 3)")
        centered = (coordinates - coordinates.mean(dim=1, keepdim=True)).contiguous()
        output = torch.empty_like(centered)
        # Triton's launch adds compiler kwargs absent from the Python signature.
        launch: Any = _pair_forces[(coordinates.shape[1], coordinates.shape[0])]
        launch(
            centered,
            *parameters,
            index,
            constants,
            output,
            coordinates.shape[1],
            BLOCK=64,
            ACC64=False,
            num_warps=2,
            enable_fp_fusion=False,
        )
        return output

    return force
