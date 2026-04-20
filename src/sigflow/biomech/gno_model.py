"""BiomechGINO — GINO surrogate for MyoSim3D forward solver.

Maps a 23-dim muscle activation vector (sigflow convention: 0 = relaxed,
100 = max contracted) to per-atom displacement of a fixed .s3d tongue
mesh. The neural replacement for `solve_equilibrium`, ~2-5 ms per forward
on GPU vs ~seconds for the Python Jacobi solver.

Paper: Li et al., "Geometry-Informed Neural Operator for Large-Scale 3D
PDEs", NeurIPS 2023.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .types import MyoSim3D


def _dominant_muscle_per_atom(
    strut_pairs: np.ndarray,
    strut_muscles: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Per-atom dominant muscle ID (most-common incident muscle; -1 = none)."""
    incident: list[list[int]] = [[] for _ in range(n_atoms)]
    for (a, b), mid in zip(strut_pairs, strut_muscles):
        if mid < 0:
            continue
        incident[a].append(int(mid))
        incident[b].append(int(mid))
    out = np.full(n_atoms, -1, dtype=np.int64)
    for i, lst in enumerate(incident):
        if lst:
            out[i] = Counter(lst).most_common(1)[0][0]
    return out


def _midline_mask(positions: np.ndarray, axis: int = 0, frac: float = 0.01) -> np.ndarray:
    """Atoms within `frac` of the axis range of the symmetry plane."""
    ax_range = float(positions[:, axis].max() - positions[:, axis].min())
    thr = frac * ax_range if ax_range > 0 else 1e-3
    return np.abs(positions[:, axis]) < thr


class BiomechGINO(nn.Module):
    """GINO wrapper specialised for a fixed MyoSim3D mesh.

    Inputs during forward: `activations: (B, n_muscles) float32` in
    sigflow convention.
    Outputs: `(B, n_atoms, 3) float32` atom displacements in the model's
    native coordinate frame (same units as `model.positions`).
    """

    def __init__(
        self,
        model: MyoSim3D,
        *,
        latent_grid: int = 16,
        fno_n_modes: tuple[int, int, int] = (8, 8, 8),
        fno_hidden_channels: int = 32,
        fno_n_layers: int = 4,
        gno_radius: float = 0.05,
        muscle_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        from neuralop.models import GINO

        n_atoms = int(model.positions.shape[0])
        n_muscles = int(len(model.muscle_names))
        self.n_atoms = n_atoms
        self.n_muscles = n_muscles

        # Normalise positions into [0, 1]^3 for GNO/GINO kernel radius
        # to be meaningful. Keep bbox so we can un-normalise displacements.
        pos = np.asarray(model.positions, dtype=np.float32)
        bmin = pos.min(axis=0)
        bmax = pos.max(axis=0)
        size = (bmax - bmin).clip(min=1e-6)
        pos_norm = (pos - bmin) / size  # (N, 3) in [0, 1]

        self.register_buffer("bbox_min", torch.from_numpy(bmin.astype(np.float32)))
        self.register_buffer("bbox_size", torch.from_numpy(size.astype(np.float32)))
        self.register_buffer(
            "rest_positions", torch.from_numpy(pos.astype(np.float32))
        )
        # GINO expects (1, N, 3) for input/output geometry
        self.register_buffer(
            "input_geom", torch.from_numpy(pos_norm.astype(np.float32)).unsqueeze(0)
        )

        # Latent regular grid on [0, 1]^3.
        g = torch.linspace(0.0, 1.0, latent_grid)
        gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
        latent = torch.stack([gx, gy, gz], dim=-1)  # (G, G, G, 3)
        self.register_buffer("latent_queries", latent.unsqueeze(0))  # (1, G, G, G, 3)

        # Per-atom dominant-muscle ID (-1 for no-muscle) + midline/fixed flags.
        dm = _dominant_muscle_per_atom(
            model.strut_pairs, model.strut_muscles, n_atoms
        )
        self.register_buffer("dominant_muscle", torch.from_numpy(dm))  # int64
        self.register_buffer(
            "midline_mask",
            torch.from_numpy(_midline_mask(pos).astype(np.float32)),
        )
        self.register_buffer(
            "fixed_mask", torch.from_numpy(np.asarray(model.fixing, dtype=np.float32))
        )

        # Muscle ID embedding: index 0 = "no muscle", indices 1..n_muscles = muscles.
        self.muscle_embedding = nn.Embedding(n_muscles + 1, muscle_embed_dim)
        self.muscle_embed_dim = muscle_embed_dim

        # Channel count of per-atom input function x:
        # [per_atom_act (1), muscle_emb (K), midline (1), fixed (1), rest_pos_norm (3)]
        in_channels = 1 + muscle_embed_dim + 1 + 1 + 3  # = 14 at default

        self.gino = GINO(
            in_channels=in_channels,
            out_channels=3,  # displacement xyz
            gno_coord_dim=3,
            in_gno_radius=gno_radius,
            out_gno_radius=gno_radius,
            # "linear" transform requires fno_in_channels == in_channels
            fno_in_channels=in_channels,
            fno_n_modes=fno_n_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_n_layers=fno_n_layers,
        )

        self._in_channels = in_channels

    def build_x(self, activations: torch.Tensor) -> torch.Tensor:
        """Construct the GINO per-atom input function from a batch of activations.

        activations: (B, n_muscles) float32, sigflow convention (0 = rest).
        Returns: (B, n_atoms, in_channels).
        """
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        B = activations.shape[0]
        N = self.n_atoms
        device = self.input_geom.device
        dtype = self.input_geom.dtype

        dm = self.dominant_muscle  # (N,) int64, -1 for no-muscle
        safe_dm = dm.clamp(min=0)  # (N,), map -1 → 0 for gather
        no_muscle = (dm < 0).to(dtype)  # (N,)

        # Per-atom activation: activations[b, dm[n]] / 100, masked
        acts_norm = (activations.to(dtype) / 100.0).clamp(0.0, 1.0)  # (B, n_muscles)
        per_atom_act = acts_norm.index_select(1, safe_dm)  # (B, N)
        per_atom_act = per_atom_act * (1.0 - no_muscle.unsqueeze(0))  # zero no-muscle atoms

        # Muscle embedding (shared across batch)
        emb_idx = (dm + 1).to(torch.long)  # (N,), 0 = no-muscle, 1..n_muscles
        muscle_emb = self.muscle_embedding(emb_idx)  # (N, K)
        muscle_emb_b = muscle_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, K)

        midline_b = self.midline_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        fixed_b = self.fixed_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        pos_b = self.input_geom.expand(B, -1, -1)  # (B, N, 3)

        x = torch.cat(
            [
                per_atom_act.unsqueeze(-1),  # (B, N, 1)
                muscle_emb_b,                # (B, N, K)
                midline_b,                   # (B, N, 1)
                fixed_b,                     # (B, N, 1)
                pos_b,                       # (B, N, 3)
            ],
            dim=-1,
        )
        return x

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """activations: (B, 23) sigflow. Returns per-atom displacement (B, N, 3).

        Displacement is in **native model units** (same scale as
        `rest_positions`). Add to `rest_positions` to get absolute
        equilibrium coordinates.
        """
        x = self.build_x(activations)  # (B, N, 14)
        B = x.shape[0]
        # GINO forward; geometry buffers are shape (1, ...) and broadcast.
        disp_norm = self.gino(
            input_geom=self.input_geom,
            output_queries=self.input_geom,
            latent_queries=self.latent_queries,
            x=x,
        )  # (B, N, 3) in normalised [0,1] units
        disp = disp_norm * self.bbox_size  # un-normalise to model units
        return disp

    @torch.no_grad()
    def predict_positions(self, activations: torch.Tensor) -> torch.Tensor:
        """Convenience: rest + disp. activations: (B, 23) or (23,)."""
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        disp = self.forward(activations)
        return self.rest_positions.unsqueeze(0) + disp


class BiomechStepGNO(nn.Module):
    """GINO learning the force-balance-relaxation step operator.

    Maps `(current_pose, activations)` → `Δpose` per one inner iteration
    of the MyoSim3D solver. Inference rolls this forward N times
    (starting from rest) until `||Δ|| < tol` — same semantics as
    `solve_equilibrium`, but learned.

    Architecture differs from `BiomechGINO` only in two places:
    - Per-atom features include the CURRENT pose (3 extra channels).
    - `input_geom` / `output_queries` follow the current pose, not the
      rest pose — so the GNO radius and latent-grid query positions
      track the deformed mesh.

    Supervised as `‖ GNO(pose_k, a) − (pose_{k+1}_true − pose_k) ‖`
    with targets pulled from `solve_equilibrium_rollout`.
    """

    def __init__(
        self,
        model: MyoSim3D,
        *,
        latent_grid: int = 16,
        fno_n_modes: tuple[int, int, int] = (8, 8, 8),
        fno_hidden_channels: int = 32,
        fno_n_layers: int = 4,
        gno_radius: float = 0.05,
        muscle_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        from neuralop.models import GINO

        n_atoms = int(model.positions.shape[0])
        n_muscles = int(len(model.muscle_names))
        self.n_atoms = n_atoms
        self.n_muscles = n_muscles

        pos = np.asarray(model.positions, dtype=np.float32)
        bmin = pos.min(axis=0)
        bmax = pos.max(axis=0)
        size = (bmax - bmin).clip(min=1e-6)
        self.register_buffer("bbox_min", torch.from_numpy(bmin.astype(np.float32)))
        self.register_buffer("bbox_size", torch.from_numpy(size.astype(np.float32)))
        self.register_buffer(
            "rest_positions", torch.from_numpy(pos.astype(np.float32))
        )

        g = torch.linspace(0.0, 1.0, latent_grid)
        gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
        latent = torch.stack([gx, gy, gz], dim=-1)
        self.register_buffer("latent_queries", latent.unsqueeze(0))

        dm = _dominant_muscle_per_atom(
            model.strut_pairs, model.strut_muscles, n_atoms
        )
        self.register_buffer("dominant_muscle", torch.from_numpy(dm))
        self.register_buffer(
            "midline_mask",
            torch.from_numpy(_midline_mask(pos).astype(np.float32)),
        )
        self.register_buffer(
            "fixed_mask", torch.from_numpy(np.asarray(model.fixing, dtype=np.float32))
        )

        self.muscle_embedding = nn.Embedding(n_muscles + 1, muscle_embed_dim)
        self.muscle_embed_dim = muscle_embed_dim

        # Extra 3 channels vs BiomechGINO: CURRENT normalised pose.
        in_channels = 1 + muscle_embed_dim + 1 + 1 + 3 + 3  # = 17 at default

        self.gino = GINO(
            in_channels=in_channels,
            out_channels=3,
            gno_coord_dim=3,
            in_gno_radius=gno_radius,
            out_gno_radius=gno_radius,
            fno_in_channels=in_channels,
            fno_n_modes=fno_n_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_n_layers=fno_n_layers,
        )
        self._in_channels = in_channels

    def _normalise_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Model-space pose → [0, 1]^3 via the rest bbox."""
        return (pose - self.bbox_min) / self.bbox_size

    def build_x(
        self, pose: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """(B, N, 3) pose + (B, n_muscles) acts → (B, N, in_channels) features."""
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        if pose.dim() == 2:
            pose = pose.unsqueeze(0)
        B = activations.shape[0]
        dtype = self.rest_positions.dtype

        dm = self.dominant_muscle
        safe_dm = dm.clamp(min=0)
        no_muscle = (dm < 0).to(dtype)

        acts_norm = (activations.to(dtype) / 100.0).clamp(0.0, 1.0)
        per_atom_act = acts_norm.index_select(1, safe_dm)
        per_atom_act = per_atom_act * (1.0 - no_muscle.unsqueeze(0))

        emb_idx = (dm + 1).to(torch.long)
        muscle_emb = self.muscle_embedding(emb_idx)
        muscle_emb_b = muscle_emb.unsqueeze(0).expand(B, -1, -1)

        midline_b = self.midline_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)
        fixed_b = self.fixed_mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, 1)

        # BiomechGINO used input_geom (rest) here. BiomechStepGNO uses
        # CURRENT pose for both the positional feature and the
        # input_geom — the operator should see where the atoms actually
        # are, not where they started.
        pose_norm = self._normalise_pose(pose.to(dtype))            # (B, N, 3)
        rest_norm = self._normalise_pose(self.rest_positions)        # (N, 3)
        rest_b = rest_norm.unsqueeze(0).expand(B, -1, -1)            # (B, N, 3)

        x = torch.cat(
            [
                per_atom_act.unsqueeze(-1),  # (B, N, 1)
                muscle_emb_b,                # (B, N, K)
                midline_b,                   # (B, N, 1)
                fixed_b,                     # (B, N, 1)
                rest_b,                      # (B, N, 3)   — static anchor
                pose_norm,                   # (B, N, 3)   — current pose
            ],
            dim=-1,
        )
        return x

    def forward(
        self, pose: torch.Tensor, activations: torch.Tensor
    ) -> torch.Tensor:
        """One step: `(pose, acts) → Δpose` in model units.

        Arguments:
            pose: (B, N, 3) current atom positions (model units).
            activations: (B, n_muscles) in MyoSim3D convention
                (100 = rest, 0 = fully contracted).
        Returns:
            (B, N, 3) displacement to apply this step. `next_pose = pose + return`.

        Note: the underlying `neuralop.GINO` neighbor search assumes
        shared geometry across the batch. Because `input_geom` here
        follows the *current* pose (different per batch item), we
        iterate batch items sequentially. Modest cost (~B× per-item
        compute) that's still far cheaper than one force-balance solve.
        """
        if pose.dim() == 2:
            pose = pose.unsqueeze(0)
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        x = self.build_x(pose, activations)                          # (B, N, C)
        pose_norm = self._normalise_pose(pose)                       # (B, N, 3)

        outputs = []
        for i in range(pose.shape[0]):
            dpose_norm_i = self.gino(
                input_geom=pose_norm[i : i + 1],
                output_queries=pose_norm[i : i + 1],
                latent_queries=self.latent_queries,
                x=x[i : i + 1],
            )                                                         # (1, N, 3)
            outputs.append(dpose_norm_i)
        dpose_norm = torch.cat(outputs, dim=0)                        # (B, N, 3)
        return dpose_norm * self.bbox_size                            # back to model units

    @torch.no_grad()
    def rollout(
        self,
        activations: torch.Tensor,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        pose_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Iterate `forward` from rest (or `pose_init`) to equilibrium.

        Matches `solve_equilibrium` inference semantics. Returns the
        terminal pose `(B, N, 3)` only — callers who want the full
        trajectory should loop `forward` manually.
        """
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        B = activations.shape[0]
        if pose_init is None:
            pose = self.rest_positions.unsqueeze(0).expand(B, -1, -1).clone()
        else:
            pose = pose_init.clone()
        for _ in range(max_iter):
            dpose = self.forward(pose, activations)
            pose = pose + dpose
            if float(dpose.norm()) < tol:
                break
        return pose


__all__ = [
    "BiomechGINO",
    "BiomechStepGNO",
    "_dominant_muscle_per_atom",
    "_midline_mask",
]
