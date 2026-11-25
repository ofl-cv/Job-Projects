#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDEM Geometry Replicator + Clog Advisor
=======================================

Copies geometry sections in an Altair EDEM deck (.dem) and places each copy
at a user-chosen distance from the original. Beyond plain replication it can:

  * write the copies straight into the deck through the documented EDEMpy
    editing API (Deck(path, mode='w') + deck.createGeometry), material
    included, with an automatic timestamped backup of the deck first;
  * export binary STL files instead, for manual import in EDEM Creator;
  * snapshot everything it can reach of the Creator / Simulator / Analysis
    data (geometries, materials, particles, time step, grid, domain, ...)
    into the JSON manifest so every run is fully documented;
  * copy each section's particle factories (e.g. a virtual Polygon inlet
    with "New Factory 1") to the copies: deck.addFactory attaches a fresh
    factory to each copy through the documented API, then the original
    factory's parameter values are mirrored onto it at HDF5 level, and the
    source section's physical/virtual type flags are mirrored too
    (--copy-factories);
  * copy each section's existing kinematics (Linear Rotation, Linear
    Translation, ...) to the copies, with every point field (start/end
    points, point of application) shifted by the copy offset
    (--copy-kinematics);
  * replay recorded rigid-body motion onto the copies as Linear Translation /
    Linear Rotation kinematics via deck.createKinematic, from a CSV with
    columns  timestep, body, x, y, z, psi, theta, phi  (based on the
    MotionSolve-to-EDEM transfer approach by J. Handel);
  * check the setup against the particle size and warn about clogging:
    measured gap between original and copy, arching limits at openings,
    Rayleigh time step and grid cell size -- and with --optimize it pushes
    the copies apart until the gap clears a safety factor times the largest
    particle diameter.

Usage
-----
    python edem_geometry_replicator.py DECK.dem --distance 0.5
    python edem_geometry_replicator.py DECK.dem --offset 0 0 0.5 --count 3
    python edem_geometry_replicator.py DECK.dem --distance 0.4 --direction 0 1 0 \\
        --write-deck --material Steel --optimize --particle-diameter 0.006
    python edem_geometry_replicator.py DECK.dem --offset 0 0.8 0 --count 2 \\
        --write-deck --material Steel --copy-factories --copy-kinematics
    python edem_geometry_replicator.py DECK.dem --list-factories
    python edem_geometry_replicator.py DECK.dem --lattice 4 3 1 --spacing 1 0.8 0
    python edem_geometry_replicator.py DECK.dem --rotate 0 0 30 --count 11 \\
        --pivot 0 0 0 --write-deck --material Steel
    python edem_geometry_replicator.py DECK.dem --list-geometries

Run it with no transform flags in a terminal and it simply asks how far away
the copy should go and how many copies to make (copy k is placed at k times
the offset), so it also works without memorising any flags. Invalid answers
are re-asked in a loop, and decimal commas ("0,5") are accepted.

Conventions: metres and degrees everywhere; STL files are metres too, so
import them in Creator with units = metres. After every run, read
replication_manifest.json in the output directory -- it records each copy,
its exact transform, the deck snapshot, the clogging advice and the few
steps that still have no public EDEMpy API (Setup Selections in particular).

Notes on the EDEMpy API surface used here:
  * reading meshes (getXCoords/getYCoords/getZCoords/getTriangleNodes),
    deck.geometryNames, deck.materialNames, Deck(path, mode='w'),
    deck.createGeometry and deck.createKinematic are all in Altair's
    official EDEMpy documentation (2022.2+);
  * only the LAST timestep of a deck can be modified per the docs, so use
    --write-deck on setup decks (before running the simulation);
  * deck.addFactory(geom_name, factory_name) is documented (2022.1+), but
    there is NO read API for factory or kinematic parameters, so
    --copy-factories/--copy-kinematics mirror the parameter values between
    same-schema HDF5 groups inside the deck; always verify in Creator, and
    use --list-factories / --list-kinematics to inspect first;
  * Setup Selections have no read or write API at all -- the read-only HDF5
    scan lists their names so you can recreate them in Creator.

Requires: Python 3.8+ and numpy (any version from 1.17 up, including 2.x --
but note that edempy itself currently requires numpy < 2, so in the same
environment as edempy stick to numpy 1.x). Runs on Windows, macOS and Linux;
edempy is only needed for deck access, h5py only for the selection scan.

Author: Olle Ericsson Gustafsson
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib
import json
import logging
import math
import os
import re
import shutil
import struct
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Sequence

import numpy as np

# Static type checkers see the precise numpy.typing aliases; at runtime the
# plain ndarray is used instead, so old numpy versions (some EDEM-bundled
# Pythons ship pre-1.20 without numpy.typing) still work.
if TYPE_CHECKING:
    import numpy.typing as npt
    FloatArray = npt.NDArray[np.float64]
    IntArray = npt.NDArray[np.int64]
else:
    FloatArray = np.ndarray
    IntArray = np.ndarray

LOG = logging.getLogger("edem_geometry_replicator")

_I3: Final = np.eye(3)


# ---------------------------------------------------------------------------
# Errors. ConfigError = user mistake (exit 2), the rest = deck/environment
# problems (exit 1), so batch files can tell them apart.
# ---------------------------------------------------------------------------
class ReplicatorError(Exception):
    """Base class for every error this module raises on purpose."""


class ConfigError(ReplicatorError):
    """The user's input or configuration is invalid."""


class DeckReadError(ReplicatorError):
    """The deck could not be opened, or its contents could not be read."""


class DeckWriteError(ReplicatorError):
    """Writing into the deck via the EDEMpy editing API failed."""


class ApiMismatchError(DeckReadError):
    """The installed EDEMpy exposes none of the accessor names known here.

    The message lists what your version actually has, so the candidate
    tables in EdemDeck can be extended instead of debugging blind.
    """


# ---------------------------------------------------------------------------
# Rigid-body transforms.  v' = R (v - pivot) + pivot + translation.
# Pure translation is R = I; the pivot may also be "each mesh's centroid".
# ---------------------------------------------------------------------------
def rotation_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> FloatArray:
    """R = Rz @ Ry @ Rx: rotate about fixed global X, then Y, then Z (degrees)."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], float)
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], float)
    r_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], float)
    return r_z @ r_y @ r_x


def rotation_from_axis_angle_deg(axis: Sequence[float], angle_deg: float) -> FloatArray:
    """Rodrigues formula; the axis may have any nonzero length."""
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(a))
    if length < 1e-12:
        raise ConfigError(f"rotation axis {list(axis)} has (near) zero length")
    a /= length
    theta = math.radians(angle_deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return _I3 + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def _rotation_vector(r: FloatArray) -> FloatArray:
    """Axis * angle (radians) from a rotation matrix. Used for motion replay."""
    angle = math.acos(max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0)))
    if angle < 1e-10:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis / (2.0 * math.sin(angle)) * angle


def _validated_rotation(matrix: Any) -> FloatArray:
    """Rotation matrices must be orthonormal and proper (det = +1): a mirror
    would silently flip triangle winding and turn geometry inside out for
    contact detection, so it is rejected."""
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape != (3, 3):
        raise ConfigError(f"rotation matrix has shape {r.shape}, expected (3, 3)")
    if not np.allclose(r @ r.T, _I3, atol=1e-8):
        raise ConfigError("rotation matrix is not orthonormal (R @ R.T != I)")
    if not np.isclose(np.linalg.det(r), 1.0, atol=1e-8):
        raise ConfigError("rotation matrix determinant is not +1 (mirroring rejected)")
    return r


@dataclass(frozen=True)
class RigidTransform:
    """One rigid-body transform. meta remembers HOW it was specified (Euler
    angles, lattice index, ...) so the manifest stays human-readable."""

    rotation: FloatArray
    translation: FloatArray
    pivot: FloatArray | None = None          # None = origin
    pivot_is_centroid: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        r = _validated_rotation(self.rotation)
        t = np.asarray(self.translation, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(t)):
            raise ConfigError("translation contains non-finite values")
        p = self.pivot
        if p is not None:
            p = np.asarray(p, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(p)):
                raise ConfigError("pivot contains non-finite values")
        if self.pivot_is_centroid and p is not None:
            raise ConfigError("give either an explicit pivot or centroid, not both")
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation", t)
        object.__setattr__(self, "pivot", p)

    @classmethod
    def translation_only(
        cls, offset: "Sequence[float] | FloatArray",
        meta: dict[str, Any] | None = None,
    ) -> "RigidTransform":
        return cls(_I3.copy(), np.asarray(offset, dtype=np.float64),
                   meta=meta or {"offset_m": [float(v) for v in offset]})

    @classmethod
    def from_parts(
        cls,
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        euler_deg: Sequence[float] | None = None,
        axis: Sequence[float] | None = None,
        angle_deg: float | None = None,
        matrix: Any = None,
        pivot: Sequence[float] | None = None,
        pivot_centroid: bool = False,
    ) -> "RigidTransform":
        """Build a transform from any ONE of the rotation styles (or none)."""
        styles = [euler_deg is not None,
                  axis is not None or angle_deg is not None,
                  matrix is not None]
        if sum(styles) > 1:
            raise ConfigError("specify the rotation ONE way: euler_deg, "
                              "axis+angle_deg, or matrix")
        meta: dict[str, Any] = {"offset_m": [float(v) for v in offset]}
        if euler_deg is not None:
            e = [float(v) for v in euler_deg]
            if len(e) != 3:
                raise ConfigError("euler_deg needs exactly three angles")
            rot = rotation_from_euler_xyz_deg(*e)
            meta["euler_xyz_deg"] = e
        elif axis is not None or angle_deg is not None:
            if axis is None or angle_deg is None:
                raise ConfigError("axis and angle_deg must be given together")
            rot = rotation_from_axis_angle_deg(axis, float(angle_deg))
            meta["axis"] = [float(v) for v in axis]
            meta["angle_deg"] = float(angle_deg)
        elif matrix is not None:
            rot = _validated_rotation(matrix)
            meta["rotation_matrix"] = rot.tolist()
        else:
            rot = _I3.copy()
        if pivot_centroid:
            meta["pivot"] = "centroid"
        elif pivot is not None:
            meta["pivot"] = [float(v) for v in pivot]
        return cls(rot, np.asarray(offset, dtype=np.float64),
                   None if pivot is None else np.asarray(pivot, dtype=np.float64),
                   pivot_centroid, meta)

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.rotation, _I3, atol=1e-14)
                    and np.all(self.translation == 0.0))

    def with_translation(
        self, translation: "Sequence[float] | FloatArray"
    ) -> "RigidTransform":
        """Same rotation/pivot, different translation (used by the optimizer)."""
        return RigidTransform(self.rotation.copy(), np.asarray(translation, float),
                              None if self.pivot is None else self.pivot.copy(),
                              self.pivot_is_centroid, dict(self.meta))

    def as_affine(self) -> tuple[FloatArray, FloatArray]:
        """Rewrite as v' = R v + d, needed to compose transforms (--count)."""
        if self.pivot_is_centroid:
            raise ConfigError(
                "--count composes the transform with itself, which needs a "
                "FIXED pivot; --pivot-centroid varies per mesh. Use --pivot.")
        p = self.pivot if self.pivot is not None else np.zeros(3)
        return self.rotation.copy(), p - self.rotation @ p + self.translation

    def apply_points(self, points: FloatArray) -> FloatArray:
        """Transform an (n, 3) point array. Vectorized, O(n)."""
        if self.is_identity:
            return points
        if np.allclose(self.rotation, _I3, atol=1e-14):
            return points + self.translation
        if self.pivot_is_centroid:
            p = points.mean(axis=0)          # vertex mean, not area-weighted
        else:
            p = self.pivot if self.pivot is not None else np.zeros(3)
        return (points - p) @ self.rotation.T + p + self.translation

    def apply(self, mesh: "TriangleMesh") -> "TriangleMesh":
        """New mesh with this transform applied. Connectivity is shared with
        the source (indices do not change under rigid motion)."""
        if self.is_identity:
            return mesh
        return TriangleMesh(self.apply_points(mesh.vertices), mesh.triangles)

    def describe(self) -> dict[str, Any]:
        return {
            "translation_m": self.translation.tolist(),
            "rotation_matrix": self.rotation.tolist(),
            "pivot": ("centroid" if self.pivot_is_centroid
                      else self.pivot.tolist() if self.pivot is not None else "origin"),
            "specified_as": self.meta,
        }


def cumulative_transforms(base: RigidTransform, count: int) -> tuple[RigidTransform, ...]:
    """--count semantics: copy k gets the base transform applied k times.
    Pure translation -> evenly spaced row; rotation about a pivot -> circular
    array; both -> helix. Composed exactly: R_(k+1) = R R_k, d_(k+1) = R d_k + d."""
    if count < 1:
        raise ConfigError("--count must be >= 1")
    r_base, d_base = base.as_affine()
    out: list[RigidTransform] = []
    r_k, d_k = _I3.copy(), np.zeros(3)
    for k in range(1, count + 1):
        r_k = r_base @ r_k
        d_k = r_base @ d_k + d_base
        out.append(RigidTransform(r_k.copy(), d_k.copy(),
                                  meta={"cumulative_step": k, "base": dict(base.meta)}))
    return tuple(out)


def lattice_transforms(
    counts: Sequence[int], spacing: Sequence[float], include_origin: bool = False
) -> tuple[RigidTransform, ...]:
    """n x m x k grid of pure translations. The (0,0,0) point is skipped by
    default because the original geometry already sits there."""
    n = np.asarray(counts, dtype=np.int64).reshape(3)
    s = np.asarray(spacing, dtype=np.float64).reshape(3)
    if np.any(n < 1):
        raise ConfigError(f"lattice counts must all be >= 1, got {n.tolist()}")
    if not np.all(np.isfinite(s)):
        raise ConfigError("lattice spacing contains non-finite values")
    bad = (s == 0.0) & (n > 1)
    if np.any(bad):
        raise ConfigError(
            f"lattice spacing is zero along axis/axes {np.where(bad)[0].tolist()} "
            "while more than one copy is requested there (overlapping copies)")
    grid = np.stack(
        np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing="ij"),
        axis=-1).reshape(-1, 3)
    if not include_origin:
        grid = grid[np.any(grid != 0, axis=1)]
    if grid.shape[0] > 5000:
        LOG.warning("lattice generates %d copies; this will be slow to import",
                    grid.shape[0])
    return tuple(
        RigidTransform.translation_only(
            (idx * s).astype(np.float64),
            meta={"lattice_index": idx.tolist(), "offset_m": (idx * s).tolist()})
        for idx in grid)


# ---------------------------------------------------------------------------
# Mesh data model. All numerical sanity checks happen once, here, so the
# rest of the code can trust every TriangleMesh completely.
# ---------------------------------------------------------------------------
def _normalize_triangle_indices(raw: Any, n_vertices: int) -> IntArray:
    """Coerce EDEMpy's connectivity into validated 0-based (m, 3) int64.
    Handles flat arrays and 1-based indexing (detected when the smallest
    index is >= 1 AND the largest equals the vertex count); anything outside
    the valid range raises rather than being clamped."""
    tri = np.asarray(raw)
    if tri.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if tri.ndim == 1:
        if tri.size % 3 != 0:
            raise DeckReadError(
                f"flat triangle array of size {tri.size} is not divisible by 3")
        tri = tri.reshape(-1, 3)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise DeckReadError(f"triangle array has shape {tri.shape}, expected (m, 3)")
    tri = tri.astype(np.int64, copy=False)
    tmin, tmax = int(tri.min()), int(tri.max())
    if tmin >= 1 and tmax == n_vertices:
        LOG.debug("triangle connectivity detected as 1-based; shifting")
        tri = tri - 1
        tmin, tmax = tmin - 1, tmax - 1
    if n_vertices == 0 or tmin < 0 or tmax >= n_vertices:
        raise DeckReadError(
            f"triangle indices span [{tmin}, {tmax}], outside the valid "
            f"vertex range [0, {n_vertices - 1}]")
    return tri


@dataclass(frozen=True)
class TriangleMesh:
    """Immutable triangulated surface: vertices (n,3) float64 in metres,
    triangles (m,3) int64 with 0-based indices."""

    vertices: FloatArray
    triangles: IntArray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.ndim == 1 and v.size % 3 == 0:
            v = v.reshape(-1, 3)             # some accessors return flat xyzxyz
        if v.ndim != 2 or v.shape[1] != 3:
            raise DeckReadError(f"vertex array has shape {v.shape}, expected (n, 3)")
        if v.size and not np.all(np.isfinite(v)):
            raise DeckReadError("vertex array contains non-finite values")
        v = np.ascontiguousarray(v)
        t = _normalize_triangle_indices(self.triangles, v.shape[0])
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "triangles", t)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.triangles.shape[0])


# ---------------------------------------------------------------------------
# EDEMpy access layer. The only class that talks to edempy for reading;
# unknown API variants raise ApiMismatchError listing what IS available.
# ---------------------------------------------------------------------------
def _edempy_search_roots() -> list[Path]:
    """Places where an EDEM installation (and its EDEMpy) may live."""
    roots: list[Path] = []
    for env_var in ("EDEMPY_PATH", "EDEM_INSTALL"):
        value = os.environ.get(env_var, "").strip().strip('"')
        if value:
            roots.append(Path(value))
    roots += [Path(__file__).resolve().parent, Path.cwd()]
    if sys.platform == "win32":
        roots += [Path(r"C:\Program Files\Altair"),
                  Path(r"C:\Program Files (x86)\Altair"),
                  Path(r"C:\Altair")]
    else:
        roots += [Path("/opt/Altair"), Path("/usr/local/Altair")]
    return [r for r in roots if r.is_dir()]


_EDEMPY_GLOBS: Final = ("EDEMpy*", "*/EDEMpy*", "*/EDEM*/EDEMpy*",
                        "*/EDEM*/embeddedPython/Lib/site-packages")


def _edempy_candidates() -> tuple[list[Path], list[Path]]:
    """Directories that hold the edempy package, and edempy wheel files."""
    dirs: list[Path] = []
    wheels: list[Path] = []
    for root in _edempy_search_roots():
        hits = [root]
        for pattern in _EDEMPY_GLOBS:
            hits += sorted(root.glob(pattern), reverse=True)   # newest first
        for hit in hits:
            if not hit.is_dir():
                continue
            if (hit / "edempy" / "__init__.py").is_file():
                dirs.append(hit)
            wheels += sorted(hit.glob("edempy*.whl"), reverse=True)

    def _uniq(seq: list[Path]) -> list[Path]:
        return list(dict.fromkeys(p.resolve() for p in seq))

    return _uniq(dirs), _uniq(wheels)


def _import_edempy_deck() -> Any:
    """Import edempy, searching the usual EDEM install locations if needed.

    edempy is not on PyPI -- it ships inside the EDEM installation. To keep
    this runnable by non-programmers the import tries, in order: the current
    environment, the folders named in EDEMPY_PATH / EDEM_INSTALL, the
    script's own folder (drop the edempy wheel next to this file and it just
    works), and the standard Altair installation folders. Wheel files are
    loaded directly via zipimport when possible.
    """
    try:
        from edempy import Deck
        return Deck
    except ImportError:
        pass
    dirs, wheels = _edempy_candidates()
    for candidate in [*dirs, *wheels]:
        entry = str(candidate)
        if entry in sys.path:
            continue
        sys.path.insert(0, entry)
        importlib.invalidate_caches()
        try:
            from edempy import Deck
            LOG.info("found EDEMpy automatically at %s", candidate)
            return Deck
        except ImportError:
            sys.path.remove(entry)
    found = ""
    if wheels:
        found = (f" A wheel was found at {wheels[0]} but could not be loaded "
                 f'directly; install it with:  pip install "{wheels[0]}"')
    raise DeckReadError(
        "edempy is not importable and no EDEM installation was found "
        "automatically. edempy ships WITH EDEM (it is not on PyPI); fix ONE "
        "of these:\n"
        "  1. copy the edempy wheel (edempy-*.whl, found in "
        "<EDEM install>\\EDEMpy) into the same folder as this script -- it "
        "is picked up automatically;\n"
        "  2. set the EDEMPY_PATH environment variable to that folder;\n"
        "  3. pip install the wheel into your Python environment." + found)


def _close_deck_quietly(deck: Any) -> None:
    closer = getattr(deck, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Deck.close() raised: %s", exc)


def _probe_attrs(obj: Any, attrs: Sequence[str]) -> dict[str, Any]:
    """Best-effort read of simple attributes for the deck snapshot. Values
    are converted to JSON-friendly types; anything that errors is skipped."""
    out: dict[str, Any] = {}
    for name in attrs:
        try:
            value = getattr(obj, name, None)
            if callable(value):
                value = value()
            if value is None:
                continue
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                value = [str(v) if not isinstance(v, (int, float, bool)) else v
                         for v in list(value)[:64]]
            elif isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif not isinstance(value, (int, float, str, bool)):
                value = str(value)
            out[name] = value
        except Exception:  # noqa: BLE001 - snapshot must never break a run
            continue
    return out


class EdemDeck:
    """Read-only wrapper around edempy.Deck with defensive method lookup."""

    _TRIANGLE_FNS: Final = ("getTriangleNodes", "getTriangles")
    _COORD_FNS: Final = ("getXCoords", "getYCoords", "getZCoords")
    _RADIUS_FNS: Final = ("getSphereRadii", "getRadii", "getRadius",
                          "getPhysicalRadius")
    # Attributes worth capturing per EDEM tab; all optional, all best-effort.
    _CREATOR_ATTRS: Final = ("geometryNames", "materialNames", "particleNames",
                             "numGeoms", "numMaterials", "numParticleTypes",
                             "gravity", "domainMin", "domainMax")
    _SIMULATOR_ATTRS: Final = ("timeStepSize", "timeStep", "fixedTimeStep",
                               "gridCellSize", "cellSize", "numGridCells",
                               "simulationTime", "totalTime", "saveInterval",
                               "targetSaveInterval")
    _ANALYSIS_ATTRS: Final = ("numTimesteps", "timestepKeys", "timestepValues")

    def __init__(self, deck_path: Path, timestep_index: int = 0) -> None:
        self._deck_path = deck_path
        self._timestep_index = timestep_index
        self._deck: Any = None
        self._mesh_cache: dict[str, TriangleMesh] = {}

    def __enter__(self) -> "EdemDeck":
        deck_cls = _import_edempy_deck()
        try:
            self._deck = deck_cls(str(self._deck_path))   # default: read-only
        except OSError as exc:
            raise DeckReadError(
                f"could not open {self._deck_path}: {exc}. If the EDEM GUI has "
                "this deck open it holds a file lock; close EDEM first.") from exc
        except Exception as exc:  # noqa: BLE001 - edempy error types vary
            raise DeckReadError(f"edempy failed to open the deck: {exc}") from exc
        LOG.info("opened deck read-only: %s", self._deck_path)
        return self

    def __exit__(self, *exc_info: object) -> None:
        _close_deck_quietly(self._deck)
        self._deck = None
        self._mesh_cache.clear()

    @property
    def deck(self) -> Any:
        if self._deck is None:
            raise DeckReadError("deck is not open; use EdemDeck as a context manager")
        return self._deck

    def _data_entries(self):
        """Yield candidate containers for creator/timestep data, newest API
        first: creatorData[t], creatorData (older, non-indexed), timestep[t]."""
        cd = getattr(self.deck, "creatorData", None)
        if cd is not None:
            try:
                yield cd[self._timestep_index]
            except TypeError:
                yield cd
            except Exception:  # noqa: BLE001
                pass
        try:
            yield self.deck.timestep[self._timestep_index]
        except Exception:  # noqa: BLE001
            pass

    def _container(self, kind: str) -> Any:
        for entry in self._data_entries():
            container = getattr(entry, kind, None)
            if container is not None:
                return container
        raise ApiMismatchError(
            f"neither creatorData nor timestep exposes a {kind!r} container; "
            f"deck attributes: "
            f"{sorted(a for a in dir(self.deck) if not a.startswith('_'))}")

    @staticmethod
    def _api_mismatch(obj: Any, tried: Sequence[str], what: str) -> ApiMismatchError:
        public = sorted(a for a in dir(obj) if not a.startswith("_"))
        return ApiMismatchError(
            f"none of the known accessors {list(tried)} for {what} exist on "
            f"{type(obj).__name__}. Available public attributes: {public}.")

    # -- public API ---------------------------------------------------------
    def geometry_names(self) -> list[str]:
        names = getattr(self.deck, "geometryNames", None)
        if names is not None:
            return [str(n) for n in names]
        keys = getattr(self._container("geometry"), "keys", None)
        if callable(keys):
            return [str(k) for k in keys()]
        raise self._api_mismatch(self.deck, ("geometryNames", "geometry.keys()"),
                                 "geometry section names")

    def material_names(self) -> list[str]:
        names = getattr(self.deck, "materialNames", None)
        if names is None:
            raise self._api_mismatch(self.deck, ("materialNames",), "material names")
        return [str(n) for n in names]

    def mesh(self, name: str) -> TriangleMesh:
        """Read one section's mesh; each section hits the disk exactly once."""
        cached = self._mesh_cache.get(name)
        if cached is not None:
            return cached
        try:
            geom = self._container("geometry")[name]
        except ReplicatorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DeckReadError(f"geometry section {name!r} not accessible: {exc}") from exc

        coord_fns = [getattr(geom, n, None) for n in self._COORD_FNS]
        if all(callable(f) for f in coord_fns):
            coords = np.column_stack(
                [np.asarray(f(), dtype=np.float64)
                 for f in coord_fns if callable(f)])
        elif callable(getattr(geom, "getCoords", None)):
            coords = np.asarray(geom.getCoords(), dtype=np.float64)
        else:
            raise self._api_mismatch(geom, (*self._COORD_FNS, "getCoords"),
                                     f"node coordinates of {name!r}")
        for cand in self._TRIANGLE_FNS:
            fn = getattr(geom, cand, None)
            if callable(fn):
                triangles = fn()
                break
        else:
            raise self._api_mismatch(geom, self._TRIANGLE_FNS,
                                     f"triangle connectivity of {name!r}")
        mesh = TriangleMesh(coords, triangles)
        LOG.debug("read %r: %d vertices, %d triangles",
                  name, mesh.n_vertices, mesh.n_triangles)
        self._mesh_cache[name] = mesh
        return mesh

    def particle_diameters(self) -> dict[str, float]:
        """Best-effort particle sphere diameters, for the clog advisor.
        Returns {} when this EDEMpy version keeps them somewhere unknown."""
        out: dict[str, float] = {}
        try:
            container = self._container("particle")
        except ReplicatorError:
            return out
        names = getattr(self.deck, "particleNames", None)
        if names is None:
            keys = getattr(container, "keys", None)
            names = list(keys()) if callable(keys) else []
        for name in names:
            try:
                part = container[str(name)]
            except Exception:  # noqa: BLE001
                continue
            for cand in self._RADIUS_FNS:
                fn = getattr(part, cand, None)
                if not callable(fn):
                    continue
                try:
                    r = float(np.max(np.asarray(fn(), dtype=np.float64)))
                except Exception:  # noqa: BLE001
                    continue
                if r > 0:
                    out[str(name)] = 2.0 * r
                    break
        return out

    def snapshot(self) -> dict[str, Any]:
        """Everything reachable of the Creator/Simulator/Analysis data, for
        the manifest. Read-only and never fatal: missing attributes are
        simply absent from the result."""
        deck = self.deck
        snap = {
            "creator": _probe_attrs(deck, self._CREATOR_ATTRS),
            "simulator": _probe_attrs(deck, self._SIMULATOR_ATTRS),
            "analysis": _probe_attrs(deck, self._ANALYSIS_ATTRS),
        }
        try:
            snap["creator"].setdefault("geometryNames", self.geometry_names())
        except ReplicatorError:
            pass
        try:
            diam = self.particle_diameters()
            if diam:
                snap["creator"]["particle_diameters_m"] = diam
        except Exception:  # noqa: BLE001
            pass
        for entry in self._data_entries():
            sim = _probe_attrs(entry, self._SIMULATOR_ATTRS)
            for k, v in sim.items():
                snap["simulator"].setdefault(k, v)
            break
        return snap


# ---------------------------------------------------------------------------
# Binary STL export. Little-endian: 80-byte header, uint32 triangle count,
# then 50 bytes per triangle (normal, 3 vertices, attribute). One structured
# dtype per chunk keeps it a single tobytes() call and peak memory flat.
# ---------------------------------------------------------------------------
_STL_RECORD: Final = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
_STL_CHUNK: Final = 1_000_000
_AREA_EPS: Final = 1e-30


def write_binary_stl(path: Path, mesh: TriangleMesh, solid_name: str = "") -> None:
    m = mesh.n_triangles
    degenerate = 0
    # Header must NOT start with 'solid' or some readers misparse it as ASCII.
    header = f"EDEMREP {solid_name}".encode("ascii", errors="replace")[:80]
    with path.open("wb") as fh:
        fh.write(header.ljust(80, b"\0"))
        fh.write(struct.pack("<I", m))
        for start in range(0, m, _STL_CHUNK):
            tri = mesh.triangles[start:start + _STL_CHUNK]
            corners = mesh.vertices[tri]                       # (k, 3, 3)
            normals = np.cross(corners[:, 1] - corners[:, 0],
                               corners[:, 2] - corners[:, 0])
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            degenerate += int(np.count_nonzero(lengths <= _AREA_EPS))
            unit = np.divide(normals, lengths, out=np.zeros_like(normals),
                             where=lengths > _AREA_EPS)
            records = np.zeros(tri.shape[0], dtype=_STL_RECORD)
            records["normal"] = unit.astype(np.float32)
            records["vertices"] = corners.astype(np.float32)
            fh.write(records.tobytes())
    if degenerate:
        LOG.warning("%s: %d degenerate (zero-area) triangle(s) written with "
                    "zero normal", path.name, degenerate)


# ---------------------------------------------------------------------------
# Motion replay: recorded rigid-body motion -> deck.createKinematic on the
# copies. CSV columns: timestep, body, x, y, z, psi, theta, phi (angles as
# MotionSolve zyx Euler). Credit: based on J. Handel's MS-to-EDEM script.
# ---------------------------------------------------------------------------
def load_motion_table(path: Path, angle_unit: str = "deg") -> dict[str, FloatArray]:
    """Parse the motion CSV into {body: (n, 7) array of t,x,y,z,psi,theta,phi},
    sorted by time, angles normalised to degrees. Header row optional."""
    if not path.is_file():
        raise ConfigError(f"motion CSV not found: {path}")
    rows: dict[str, list[list[float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            if not row or not row[0].strip():
                continue
            try:
                t = float(row[0])
            except ValueError:
                if lineno == 1:
                    continue                  # header row
                raise ConfigError(f"{path}:{lineno}: cannot parse timestep {row[0]!r}")
            if len(row) < 8:
                raise ConfigError(
                    f"{path}:{lineno}: need 8 columns "
                    "(timestep, body, x, y, z, psi, theta, phi), got "
                    f"{len(row)}")
            try:
                values = [float(v) for v in row[2:8]]
            except ValueError as exc:
                raise ConfigError(f"{path}:{lineno}: {exc}") from exc
            rows.setdefault(row[1].strip(), []).append([t, *values])
    if not rows:
        raise ConfigError(f"motion CSV {path} contains no data rows")
    table: dict[str, FloatArray] = {}
    for body, data in rows.items():
        arr = np.asarray(data, dtype=np.float64)
        arr = arr[np.argsort(arr[:, 0], kind="stable")]
        if angle_unit == "rad":
            arr[:, 4:7] = np.rad2deg(arr[:, 4:7])
        table[body] = arr
    return table


def _euler_zyx_ms(psi: float, theta: float, phi: float) -> FloatArray:
    """MotionSolve euler ('zyx', [psi, -theta, -phi]) as a matrix:
    Rz(psi) @ Ry(-theta) @ Rx(-phi)."""
    return rotation_from_euler_xyz_deg(-phi, -theta, psi)


class MotionWriter:
    """Turns consecutive motion samples into Linear Translation / Linear
    Rotation kinematics on a (copied) body, POA shifted with the copy."""

    def __init__(self, deck: Any) -> None:
        self._deck = deck

    def replay(self, rows: FloatArray, body: str, transform: RigidTransform) -> int:
        rotate_copy = not np.allclose(transform.rotation, _I3, atol=1e-12)
        if rotate_copy:
            LOG.info("copy %r is rotated; motion axes/POA are rotated with it", body)
        created = 0
        for i in range(len(rows) - 1):
            t0, t1 = float(rows[i, 0]), float(rows[i + 1, 0])
            if t1 <= t0:
                continue
            created += self._translation_segment(body, i, t0, t1,
                                                 rows[i, 1:4], rows[i + 1, 1:4])
            created += self._rotation_segment(body, i, t0, t1,
                                              rows[i], rows[i + 1], transform,
                                              rotate_copy)
        return created

    def _create(self, body: str, **kwargs: Any) -> None:
        try:
            self._deck.createKinematic(body, **kwargs)
        except AttributeError as exc:
            raise DeckWriteError(
                "deck.createKinematic() does not exist in this EDEMpy version; "
                "kinematic replay needs EDEMpy with write support (2022.2+). "
                f"Original error: {exc}") from exc

    def _translation_segment(self, body: str, i: int, t0: float, t1: float,
                             c0: FloatArray, c1: FloatArray) -> int:
        step = c1 - c0
        dist = float(np.linalg.norm(step))
        if dist == 0.0:
            return 0
        self._create(
            body,
            kinematic_name=f"{body}_tr_{i:04d}",
            kinematic_type="Linear Translation",
            start_time=t0,
            end_time=t1,
            translation_init_acceleration_value=0,
            translation_init_velocity_value=dist / (t1 - t0),
            moves_with_body=False,            # global coordinates
            start_point=np.zeros(3),
            end_point=step,                   # direction of travel
        )
        return 1

    def _rotation_segment(self, body: str, i: int, t0: float, t1: float,
                          row0: FloatArray, row1: FloatArray,
                          transform: RigidTransform, rotate_copy: bool) -> int:
        m0 = _euler_zyx_ms(row0[4], row0[5], row0[6])
        m1 = _euler_zyx_ms(row1[4], row1[5], row1[6])
        vec = _rotation_vector(m1 @ m0.T)
        angle = float(np.linalg.norm(vec))     # radians
        if angle < 1e-12:
            return 0
        # Axis correction from the MotionSolve export convention: express the
        # axis relative to the body's start orientation, x flipped.
        axis = m0.T @ (vec / angle)
        axis[0] = -axis[0]
        if rotate_copy:
            axis = transform.rotation @ axis
        # POA follows the copy. Centroid pivots depend on the mesh, so fall
        # back to the plain translation there.
        c0 = row0[1:4]
        if transform.pivot_is_centroid:
            poa = c0 + transform.translation
        else:
            poa = transform.apply_points(c0.reshape(1, 3))[0]
        self._create(
            body,
            kinematic_name=f"{body}_rot_{i:04d}",
            kinematic_type="Linear Rotation",
            start_time=t0,
            end_time=t1,
            rotation_init_angular_acceleration_value=0,
            rotation_init_angular_velocity_value=angle / (t1 - t0),  # rad/s
            moves_with_body=False,
            rotation_start_point=axis,
            rotation_end_point=np.array([0.0, axis[1], axis[2]]),
            rotation_active_poa=np.array([False, True, False]),
            rotation_poa=poa,
        )
        return 1


# ---------------------------------------------------------------------------
# Clog advisor / optimizer. Deterministic checks against the particle size:
# no simulation is run, but the classic DEM rules of thumb are applied and
# the copy offsets can be grown until the measured gap is safe.
# ---------------------------------------------------------------------------
_GAP_SAMPLE: Final = 6000        # vertices used per side for the gap estimate


def rayleigh_timestep(diameter: float, density: float,
                      shear_modulus: float, poisson: float) -> float:
    """Rayleigh critical time step for the smallest sphere; simulations
    usually run at 20-40 % of this."""
    r = 0.5 * diameter
    return math.pi * r * math.sqrt(density / shear_modulus) / (0.1631 * poisson + 0.8766)


def min_gap(a: FloatArray, b: FloatArray, seed: int = 1) -> float:
    """Approximate smallest distance between two vertex clouds. Both sides
    are subsampled to _GAP_SAMPLE points (fixed seed, reproducible) and
    compared brute-force in chunks; accuracy is bounded by mesh resolution,
    which is fine for a safety-margin check."""
    rng: Any
    try:
        rng = np.random.default_rng(seed)
    except AttributeError:               # numpy < 1.17
        rng = np.random.RandomState(seed)

    def pick(v: FloatArray) -> FloatArray:
        if len(v) <= _GAP_SAMPLE:
            return v
        return v[rng.choice(len(v), _GAP_SAMPLE, replace=False)]

    a, b = pick(np.asarray(a, float)), pick(np.asarray(b, float))
    if not len(a) or not len(b):
        return math.inf
    best = math.inf
    chunk = max(1, 20_000_000 // max(len(b), 1))
    for i in range(0, len(a), chunk):
        d2 = ((a[i:i + chunk, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        best = min(best, float(d2.min()))
    return math.sqrt(best)


@dataclass(frozen=True)
class ParticleSpec:
    """What the advisor needs to know about the (largest) particle."""

    diameter: float                  # m
    density: float = 2500.0          # kg/m3
    shear_modulus: float = 1.0e8     # Pa (typical softened DEM value)
    poisson: float = 0.25

    def __post_init__(self) -> None:
        if not (self.diameter > 0 and self.density > 0
                and self.shear_modulus > 0 and 0 < self.poisson < 0.5):
            raise ConfigError("particle spec values must be positive "
                              "(and 0 < poisson < 0.5)")


def clogging_advice(particle: ParticleSpec, gap_factor: float) -> dict[str, Any]:
    """Rules of thumb against mechanical arching and unstable integration."""
    d = particle.diameter
    t_r = rayleigh_timestep(d, particle.density, particle.shear_modulus,
                            particle.poisson)
    return {
        "particle_diameter_m": d,
        "min_gap_between_geometries_m": gap_factor * d,
        "min_opening_free_flowing_m": 6.0 * d,
        "min_opening_cohesive_m": 10.0 * d,
        "rayleigh_timestep_s": t_r,
        "recommended_sim_timestep_s": 0.2 * t_r,
        "recommended_grid_cell_size_m": 2.5 * (d / 2.0),
        "notes": [
            f"Keep every opening particles must pass at >= 6x the particle "
            f"diameter ({6 * d:.4g} m) for free-flowing material, >= 10x "
            f"({10 * d:.4g} m) if cohesive, to avoid mechanical arching.",
            "Run the simulator at 20-40 % of the Rayleigh time step.",
            "Grid cell size around 2.5x the minimum particle radius keeps "
            "contact detection efficient.",
            "The gap check compares mesh vertices, so treat it as a safety "
            "margin estimate, not an exact clearance.",
        ],
    }


def optimise_transforms(
    transforms: Sequence[RigidTransform],
    source_vertices: FloatArray,
    particle: ParticleSpec,
    gap_factor: float,
    max_iterations: int = 12,
) -> tuple[tuple[RigidTransform, ...], list[dict[str, Any]]]:
    """Grow each copy's translation until the vertex-cloud gap to the source
    geometry reaches gap_factor * particle diameter. Rotation and pivot are
    left untouched; only the translation is pushed along its own direction
    (or away from the source centroid when there is none)."""
    target = gap_factor * particle.diameter
    adjusted: list[RigidTransform] = []
    reports: list[dict[str, Any]] = []
    for index, tr in enumerate(transforms, start=1):
        gap0 = min_gap(source_vertices, tr.apply_points(source_vertices))
        current, gap = tr, gap0
        iterations = 0
        while gap < target and iterations < max_iterations:
            direction = current.translation.copy()
            length = float(np.linalg.norm(direction))
            if length < 1e-12:
                moved_c = current.apply_points(source_vertices).mean(axis=0)
                direction = moved_c - source_vertices.mean(axis=0)
                length = float(np.linalg.norm(direction))
            if length < 1e-12:
                direction, length = np.array([1.0, 0.0, 0.0]), 1.0
            push = (target - gap) + 0.05 * target      # 5 % overshoot margin
            current = current.with_translation(
                current.translation + push * direction / length)
            gap = min_gap(source_vertices, current.apply_points(source_vertices))
            iterations += 1
        if gap < target:
            LOG.warning("copy %d: gap %.4g m still below target %.4g m after "
                        "%d iterations (rotation/pivot constraints?)",
                        index, gap, target, iterations)
        elif iterations:
            LOG.info("copy %d: offset grown %s -> %s to clear the %.4g m gap",
                     index, np.round(tr.translation, 6).tolist(),
                     np.round(current.translation, 6).tolist(), target)
        adjusted.append(current)
        reports.append({
            "copy_index": index,
            "target_gap_m": target,
            "initial_gap_m": gap0,
            "final_gap_m": gap,
            "initial_translation_m": tr.translation.tolist(),
            "final_translation_m": current.translation.tolist(),
            "adjusted": bool(iterations),
        })
    return tuple(adjusted), reports


# ---------------------------------------------------------------------------
# Writer backends: STL files, or straight into the .dem via the documented
# editing API (with backup, material validation and name de-duplication).
# ---------------------------------------------------------------------------
@dataclass
class CopyJob:
    """One planned copy. The transform is applied lazily at write time so
    only one transformed vertex array lives in memory at a time."""

    source_name: str
    copy_name: str
    transform: RigidTransform
    source_mesh: TriangleMesh


class StlExportBackend:
    """Default backend: one binary STL per copy, importable in Creator."""

    kinematics_created = 0

    def write_batch(self, deck_path: Path, out_dir: Path,
                    jobs: Sequence[CopyJob]) -> list[str]:
        destinations = []
        for job in jobs:
            path = out_dir / f"{job.copy_name}.stl"
            write_binary_stl(path, job.transform.apply(job.source_mesh),
                             solid_name=job.copy_name)
            destinations.append(str(path))
            LOG.info("wrote %s", path)
        return destinations


class EdempyDeckWriteBackend:
    """Writes copies directly into the .dem, optionally replaying recorded
    motion onto them. A timestamped backup is made first (unless disabled),
    the material is validated up front, and names are de-duplicated so
    nothing is silently overwritten."""

    def __init__(self, material: str, backup: bool = True,
                 motion_table: dict[str, FloatArray] | None = None,
                 factory_plan: dict[str, list[FactoryInfo]] | None = None,
                 known_factory_names: set[str] | None = None) -> None:
        self._material = material
        self._backup = backup
        self._motion_table = motion_table or {}
        self._factory_plan = factory_plan or {}
        self._factory_names = set(known_factory_names or set())
        self.kinematics_created = 0
        self.factory_plans_created: list[FactoryCopyPlan] = []

    def write_batch(self, deck_path: Path, out_dir: Path,
                    jobs: Sequence[CopyJob]) -> list[str]:
        if self._backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = deck_path.with_name(f"{deck_path.stem}_backup_{stamp}.dem")
            shutil.copy2(deck_path, backup)
            LOG.info("deck backup written: %s", backup)

        deck = self._open_writable(deck_path)
        destinations: list[str] = []
        try:
            materials = [str(n) for n in getattr(deck, "materialNames", [])]
            if materials and self._material not in materials:
                raise DeckWriteError(
                    f"material {self._material!r} does not exist in the deck; "
                    f"available materials: {materials}")
            existing = {str(n) for n in getattr(deck, "geometryNames", [])}
            motion = MotionWriter(deck) if self._motion_table else None

            for job in jobs:
                name, k = job.copy_name, 1
                while name in existing:
                    name = f"{job.copy_name}_{k}"
                    k += 1
                self._create_geometry(deck, name, job.transform.apply(job.source_mesh))
                existing.add(name)
                destinations.append(f"{deck_path.name}::{name}")
                LOG.info("created geometry %r in deck (material %r)",
                         name, self._material)
                self._add_factories(deck, job, name)
                rows = self._motion_table.get(job.source_name)
                if motion is not None and rows is not None:
                    n = motion.replay(rows, name, job.transform)
                    self.kinematics_created += n
                    LOG.info("replayed %d kinematic segment(s) onto %r", n, name)
        finally:
            _close_deck_quietly(deck)
        return destinations

    def _add_factories(self, deck: Any, job: CopyJob, copy_name: str) -> None:
        """Attach one fresh factory per original factory of the source
        section, via the documented deck.addFactory. Parameter values are
        mirrored onto them afterwards, in the HDF5 stage."""
        for info in self._factory_plan.get(job.source_name, ()):
            fname, k = f"{info.display_name}__{copy_name}", 1
            while fname in self._factory_names:
                fname = f"{info.display_name}__{copy_name}_{k}"
                k += 1
            try:
                deck.addFactory(copy_name, fname)
            except AttributeError as exc:
                raise DeckWriteError(
                    "deck.addFactory() does not exist in this EDEMpy version; "
                    "--copy-factories needs the factory editing API "
                    "(documented since EDEM 2022.1). Original error: "
                    f"{exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise DeckWriteError(
                    f"deck.addFactory({copy_name!r}, {fname!r}) failed: {exc}. "
                    "Check help(deck.addFactory) on your installation."
                ) from exc
            self._factory_names.add(fname)
            self.factory_plans_created.append(FactoryCopyPlan(
                info, fname, job.source_name, copy_name,
                job.transform.translation.tolist()))
            LOG.info("attached factory %r to %r (from %r)",
                     fname, copy_name, info.display_name)

    @staticmethod
    def _open_writable(deck_path: Path) -> Any:
        deck_cls = _import_edempy_deck()
        try:
            # updateDeckVersion migrates older decks on write, when supported.
            return deck_cls(str(deck_path), mode="w", updateDeckVersion=True)
        except TypeError:
            pass
        try:
            return deck_cls(str(deck_path), mode="w")
        except TypeError as exc:
            raise DeckWriteError(
                "this EDEMpy version does not accept Deck(..., mode='w'); deck "
                "editing needs the write support Altair documents for 2022.2+. "
                f"Use the default STL backend instead. Original error: {exc}"
            ) from exc
        except OSError as exc:
            raise DeckWriteError(
                f"could not open {deck_path} for writing: {exc}. If the EDEM "
                "GUI has this deck open it holds a file lock; close EDEM.") from exc

    def _create_geometry(self, deck: Any, name: str, mesh: TriangleMesh) -> None:
        """deck.createGeometry(name, material, coords, triangles). Public docs
        leave the exact array layout open, so try stacked (n,3)/(m,3) first
        and fall back to flat; if both fail, report both errors."""
        attempts = (("stacked (n,3)/(m,3)", mesh.vertices, mesh.triangles),
                    ("flat 1-D", mesh.vertices.ravel(), mesh.triangles.ravel()))
        errors = []
        for label, coords, tris in attempts:
            try:
                deck.createGeometry(name, self._material, coords, tris)
                return
            except AttributeError as exc:
                raise DeckWriteError(
                    "deck.createGeometry() does not exist in this EDEMpy "
                    "version; deck editing needs EDEMpy with write support "
                    f"(2022.2+). Original error: {exc}") from exc
            except Exception as exc:  # noqa: BLE001 - report, try next layout
                errors.append(f"{label} layout: {exc}")
        raise DeckWriteError(
            f"createGeometry rejected geometry {name!r} in both known array "
            f"layouts. Errors: {errors}. Check help(deck.createGeometry) on "
            "your installation and adjust _create_geometry() accordingly.")


# ---------------------------------------------------------------------------
# Factory + kinematics transfer (--copy-factories / --copy-kinematics).
#
# EDEMpy's documented editing API can ATTACH a new factory to a geometry
# section (deck.addFactory, documented since 2022.1) but has no read API for
# the parameters of an existing factory or kinematic. The transfer therefore
# works in two layers:
#
#   1. documented API (edempy): deck.createGeometry builds the copy and
#      deck.addFactory attaches a fresh factory to it, so EDEM itself creates
#      the correct internal structure with the right name and geometry link;
#   2. HDF5 mirroring (h5py): the parameter VALUES of the original factory /
#      kinematic are then copied dataset-by-dataset into the structure EDEM
#      just created, skipping every field whose value equals the original's
#      name or the original's geometry name (those fields already point at
#      the copy and must not be overwritten).
#
# Mirroring between two same-schema sibling groups inside one deck is far
# safer than hand-building undocumented structures, but the deck schema is
# still officially undocumented: run --list-factories / --list-kinematics
# first to see what the deck contains, and always open the result in EDEM
# Creator to verify. Only pure-translation copies are supported; rotated
# copies would also need axis vectors rewritten, which is refused rather
# than silently guessed.
# ---------------------------------------------------------------------------
_FACTORY_PATH_PAT: Final = re.compile(r"factor", re.I)
_KINEMATIC_KEY_PAT: Final = re.compile(r"kinemat|dynamic|motion", re.I)
_POINTLIKE_KEY_PAT: Final = re.compile(r"point|position|poa", re.I)
_SECTION_FLAG_PAT: Final = re.compile(r"virtual|physical|type", re.I)
_NAME_KEY_PAT: Final = re.compile(r"name", re.I)
_MAX_STRING_ELEMS: Final = 4      # larger datasets are never decoded as text


def _import_h5py() -> Any:
    """Import h5py or explain how to get it (needed for the transfer layer)."""
    try:
        import h5py
        return h5py
    except ImportError:
        raise ReplicatorError(
            "h5py is required for --copy-factories/--copy-kinematics and the "
            "--list-factories/--list-kinematics inspection; install it with "
            "'pip install h5py' (the installer .bat does this automatically)."
        ) from None


def _h5_decode(value: Any) -> str | None:
    """Best-effort decode of an HDF5 scalar (or 1-element array) to str."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, str):
        return value
    return None


def _string_values(h5py: Any, group: Any) -> dict[str, str]:
    """Small string-valued datasets and attributes directly on a group,
    as {key: text}. Large datasets are skipped so mesh arrays are never
    read into memory here."""
    out: dict[str, str] = {}
    for key, item in group.items():
        if not isinstance(item, h5py.Dataset) or item.size > _MAX_STRING_ELEMS:
            continue
        if item.dtype.kind in ("S", "O", "U"):
            text = _h5_decode(item[()])
            if text:
                out[key] = text
    for key, value in group.attrs.items():
        text = _h5_decode(value)
        if text:
            out[f"@{key}"] = text
    return out


@dataclass(frozen=True)
class FactoryInfo:
    """One factory found in the deck: its HDF5 group path, display name
    (e.g. 'New Factory 1') and the geometry section it belongs to."""

    path: str
    display_name: str
    geometry: str | None
    string_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class TransferPrescan:
    """Read-only map of the deck tree used by the transfer stage: where each
    geometry section lives in HDF5, which factories belong to which section,
    and which kinematics subtrees each section carries."""

    geom_groups: dict[str, str]                       # name -> HDF5 path
    factories_by_geom: dict[str, list[FactoryInfo]]
    kinematic_keys_by_geom: dict[str, list[str]]      # name -> subtree keys
    all_factories: list[FactoryInfo]


def _map_geometry_groups(h5py: Any, fh: Any, names: Sequence[str]) -> dict[str, str]:
    """Locate each geometry section's HDF5 group by name.

    A group is a candidate for name N if its own key equals N or it directly
    holds a small string dataset/attribute equal to N. Factory groups also
    contain geometry-name strings, so candidates then vote by parent: the
    parent group that covers the most distinct section names is taken as THE
    geometry container, and candidates under it win.
    """
    name_set = set(names)
    hits: dict[str, list[str]] = {}

    def visit(path: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        key = path.rsplit("/", 1)[-1]
        matched = key if key in name_set else None
        if matched is None:
            for text in _string_values(h5py, obj).values():
                if text in name_set:
                    matched = text
                    break
        if matched is not None:
            hits.setdefault(matched, []).append("/" + path)

    fh.visititems(visit)
    parent_cover: dict[str, set[str]] = {}
    for name, paths in hits.items():
        for p in paths:
            parent_cover.setdefault(p.rsplit("/", 1)[0], set()).add(name)
    container = max(parent_cover, key=lambda k: len(parent_cover[k]), default=None)
    out: dict[str, str] = {}
    for name, paths in hits.items():
        preferred = [p for p in paths if p.rsplit("/", 1)[0] == container]
        out[name] = (preferred or sorted(paths, key=len))[0]
    return out


def _find_factories(h5py: Any, fh: Any,
                    geom_groups: dict[str, str]) -> list[FactoryInfo]:
    """Factories = groups on a 'factor*' path that carry a name-keyed string.
    Nested subgroups of an already-found factory are not reported again. The
    owning geometry is taken from the enclosing geometry group, or from a
    string field equal to a known geometry name."""
    found: list[tuple[str, dict[str, str]]] = []

    def visit(path: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group) or not _FACTORY_PATH_PAT.search(path):
            return
        fields = _string_values(h5py, obj)
        if any(_NAME_KEY_PAT.search(k) for k in fields):
            found.append(("/" + path, fields))

    fh.visititems(visit)
    found.sort(key=lambda item: item[0].count("/"))
    kept: list[tuple[str, dict[str, str]]] = []
    for path, fields in found:
        if any(path.startswith(prev + "/") for prev, _ in kept):
            continue                       # subgroup of a factory already kept
        kept.append((path, fields))

    def _display_name(fields: dict[str, str]) -> str:
        """The factory's OWN name field: an exact 'name' key wins, then any
        name-like key that is not a reference to something else (geometry,
        particle, material), then any name-like key, then any string."""
        def rank(key: str) -> int:
            k = key.lstrip("@").lower()
            if k == "name":
                return 0
            if _NAME_KEY_PAT.search(k):
                return 2 if re.search(r"geom|particle|material|section", k) else 1
            return 3
        best = min(fields, key=lambda k: (rank(k), k))
        return fields[best]

    geom_names = set(geom_groups)
    infos: list[FactoryInfo] = []
    for path, fields in kept:
        display = _display_name(fields)
        geometry = next((g for g, gp in geom_groups.items()
                         if path.startswith(gp + "/")), None)
        if geometry is None:
            geometry = next((v for v in fields.values() if v in geom_names), None)
        infos.append(FactoryInfo(path, display, geometry, fields))
    return infos


def _kinematic_keys(h5py: Any, fh: Any, geom_group_path: str) -> list[str]:
    """Direct children of a geometry group that look like kinematics."""
    try:
        group = fh[geom_group_path]
    except KeyError:
        return []
    return [key for key, item in group.items()
            if isinstance(item, h5py.Group) and _KINEMATIC_KEY_PAT.search(key)]


def prescan_deck_tree(deck_path: Path, geometry_names: Sequence[str]) -> TransferPrescan:
    """Read-only h5py pass over the deck, mapping geometry groups, factories
    and kinematics. Raises DeckReadError when the file cannot be opened
    (typically because the EDEM GUI holds a lock on it)."""
    h5py = _import_h5py()
    try:
        fh = h5py.File(deck_path, "r")
    except OSError as exc:
        raise DeckReadError(
            f"could not open {deck_path} as HDF5 for the factory/kinematics "
            f"scan: {exc}. Close EDEM if it has this deck open.") from exc
    with fh:
        geom_groups = _map_geometry_groups(h5py, fh, geometry_names)
        factories = _find_factories(h5py, fh, geom_groups)
        by_geom: dict[str, list[FactoryInfo]] = {}
        for info in factories:
            if info.geometry is not None:
                by_geom.setdefault(info.geometry, []).append(info)
        kin = {name: _kinematic_keys(h5py, fh, path)
               for name, path in geom_groups.items()}
    return TransferPrescan(geom_groups, by_geom, kin, factories)


@dataclass
class FactoryCopyPlan:
    """One factory created via deck.addFactory during the write session,
    waiting for its parameter values to be mirrored from the original."""

    source: FactoryInfo
    new_factory_name: str
    source_geometry: str
    copy_geometry: str
    translation: list[float]


def _mirror_group(h5py: Any, src: Any, dst: Any, skip_values: set[str],
                  notes: list[str], prefix: str = "") -> int:
    """Recursively copy dataset values and attributes from src into dst.

    Same-named existing datasets are overwritten in place; missing datasets
    and subgroups are copied wholesale. String values found in skip_values
    (the original's own name and geometry link) are left untouched so the
    mirrored object keeps pointing at the copy. Returns the number of
    datasets written; problems are appended to notes instead of raising, so
    one odd field never aborts the whole transfer.
    """
    written = 0
    for key, item in src.items():
        label = f"{prefix}{key}"
        if isinstance(item, h5py.Group):
            written += _mirror_group(h5py, item, dst.require_group(key),
                                     skip_values, notes, f"{label}/")
            continue
        if item.size <= _MAX_STRING_ELEMS and item.dtype.kind in ("S", "O", "U"):
            text = _h5_decode(item[()])
            if text is not None and text in skip_values:
                continue
        try:
            if key in dst:
                try:
                    dst[key][...] = item[()]
                except Exception:  # noqa: BLE001 - dtype/shape change
                    del dst[key]
                    src.copy(key, dst)
            else:
                src.copy(key, dst)
            written += 1
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not mirror dataset {label!r}: {exc}")
    for key, value in src.attrs.items():
        text = _h5_decode(value)
        if text is not None and text in skip_values:
            continue
        try:
            dst.attrs[key] = value
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not mirror attribute {prefix}@{key}: {exc}")
    return written


def _shift_point_datasets(h5py: Any, group: Any, offset: FloatArray,
                          notes: list[str]) -> int:
    """Translate every float dataset under group whose key looks like a point
    (point/position/POA) and whose last dimension is 3. Direction-like fields
    (axes, velocities) are deliberately left alone: translating a copy moves
    its points but not its directions. Returns the number of shifted arrays."""
    shifted = 0

    def visit(path: str, obj: Any) -> None:
        nonlocal shifted
        if not isinstance(obj, h5py.Dataset) or obj.dtype.kind != "f":
            return
        leaf = path.rsplit("/", 1)[-1]
        if not _POINTLIKE_KEY_PAT.search(leaf):
            return
        if not (obj.ndim >= 1 and obj.shape[-1] == 3):
            return
        try:
            obj[...] = np.asarray(obj[()], dtype=np.float64) + offset
            shifted += 1
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not shift point dataset {path!r}: {exc}")

    group.visititems(visit)
    return shifted


def apply_hdf5_transfers(
    deck_path: Path,
    all_geometry_names: Sequence[str],
    factory_plans: Sequence[FactoryCopyPlan],
    kinematic_jobs: Sequence[tuple[str, str, FloatArray]],
    mirror_section_flags: bool,
) -> dict[str, Any]:
    """Second write stage, after the edempy session has closed: mirror
    factory parameters onto the factories addFactory just created, mirror
    the physical/virtual flags of the source sections onto their copies, and
    clone + translate the kinematics subtrees. Returns a manifest report."""
    h5py = _import_h5py()
    notes: list[str] = []
    report: dict[str, Any] = {"performed": True, "factories": [],
                              "kinematics": [], "section_flags": [],
                              "notes": notes}
    try:
        fh = h5py.File(deck_path, "r+")
    except OSError as exc:
        raise DeckWriteError(
            f"could not reopen {deck_path} for the factory/kinematics "
            f"transfer: {exc}. Close EDEM if it has this deck open.") from exc
    with fh:
        geom_groups = _map_geometry_groups(h5py, fh, all_geometry_names)
        factories = _find_factories(h5py, fh, geom_groups)
        by_display: dict[str, FactoryInfo] = {}
        for info in factories:
            by_display.setdefault(info.display_name, info)

        for plan in factory_plans:
            target = by_display.get(plan.new_factory_name)
            if target is None:
                notes.append(
                    f"factory {plan.new_factory_name!r} was created via "
                    "addFactory but not found again in the HDF5 tree; its "
                    "parameters were NOT mirrored -- set them in Creator")
                continue
            skip = {plan.source.display_name, plan.source_geometry}
            n = _mirror_group(h5py, fh[plan.source.path], fh[target.path],
                              skip, notes)
            shifted = _shift_point_datasets(
                h5py, fh[target.path],
                np.asarray(plan.translation, dtype=np.float64), notes)
            report["factories"].append({
                "source_factory": plan.source.display_name,
                "new_factory": plan.new_factory_name,
                "copy_geometry": plan.copy_geometry,
                "datasets_mirrored": n,
                "point_fields_shifted": shifted,
            })
            LOG.info("factory %r -> %r: %d parameter dataset(s) mirrored",
                     plan.source.display_name, plan.new_factory_name, n)

        if mirror_section_flags:
            done: set[tuple[str, str]] = set()
            pairs = ({(p.source_geometry, p.copy_geometry) for p in factory_plans}
                     | {(src, dst) for src, dst, _ in kinematic_jobs})
            for src_name, dst_name in sorted(pairs):
                if (src_name, dst_name) in done:
                    continue
                done.add((src_name, dst_name))
                src_p, dst_p = geom_groups.get(src_name), geom_groups.get(dst_name)
                if not src_p or not dst_p:
                    notes.append(f"geometry group for {src_name!r} or "
                                 f"{dst_name!r} not found; type flags not mirrored")
                    continue
                mirrored: list[str] = []
                src_g, dst_g = fh[src_p], fh[dst_p]
                for key, item in src_g.items():
                    if (isinstance(item, h5py.Dataset)
                            and item.size <= _MAX_STRING_ELEMS
                            and _SECTION_FLAG_PAT.search(key)
                            and key in dst_g):
                        try:
                            dst_g[key][...] = item[()]
                            mirrored.append(key)
                        except Exception as exc:  # noqa: BLE001
                            notes.append(f"flag {key!r} on {dst_name!r}: {exc}")
                report["section_flags"].append(
                    {"source": src_name, "copy": dst_name, "mirrored": mirrored})
                if mirrored:
                    LOG.info("section flags %s mirrored %s -> %s",
                             mirrored, src_name, dst_name)

        for src_name, dst_name, offset in kinematic_jobs:
            src_p, dst_p = geom_groups.get(src_name), geom_groups.get(dst_name)
            if not src_p or not dst_p:
                notes.append(f"geometry group for {src_name!r} or {dst_name!r} "
                             "not found; kinematics not copied")
                continue
            keys = _kinematic_keys(h5py, fh, src_p)
            if not keys:
                notes.append(f"no kinematics subtree found under {src_name!r} "
                             "(nothing to copy); run --list-kinematics to check")
                continue
            copied, shifted = 0, 0
            for key in keys:
                dst_sub = fh[dst_p].require_group(key)
                copied += _mirror_group(h5py, fh[src_p][key], dst_sub,
                                        set(), notes, f"{key}/")
                shifted += _shift_point_datasets(
                    h5py, dst_sub, np.asarray(offset, dtype=np.float64), notes)
            report["kinematics"].append({
                "source": src_name, "copy": dst_name, "subtrees": keys,
                "datasets_mirrored": copied, "point_fields_shifted": shifted})
            LOG.info("kinematics %s: %d dataset(s) mirrored, %d point "
                     "field(s) shifted, %s -> %s",
                     keys, copied, shifted, src_name, dst_name)
    for note in notes:
        LOG.warning("%s", note)
    return report


# ---------------------------------------------------------------------------
# Setup Selection scan (strictly read-only). No public API reads or writes
# selections, but the .dem is HDF5: listing every node whose path mentions
# 'selection' at least tells you WHICH selections to recreate in Creator.
# ---------------------------------------------------------------------------
@dataclass
class SelectionScan:
    performed: bool
    note: str
    matches: list[dict[str, Any]] = field(default_factory=list)


def scan_selections(deck_path: Path, max_matches: int = 200) -> SelectionScan:
    try:
        import h5py
    except ImportError:
        return SelectionScan(False, "h5py not installed; run 'pip install h5py' "
                                    "to enable the selection scan")
    try:
        fh = h5py.File(deck_path, "r")       # read-only, guaranteed
    except OSError as exc:
        return SelectionScan(False, "deck could not be opened as HDF5 (locked "
                                    f"by a running EDEM GUI, or an older deck "
                                    f"format): {exc}")
    matches: list[dict[str, Any]] = []

    def visit(path: str, obj: Any) -> None:
        if len(matches) >= max_matches or "select" not in path.lower():
            return
        entry: dict[str, Any] = {"path": path}
        if hasattr(obj, "shape"):
            entry.update(kind="dataset", shape=list(obj.shape), dtype=str(obj.dtype))
        else:
            entry.update(kind="group", children=list(obj.keys())[:50])
        matches.append(entry)

    with fh:
        fh.visititems(visit)
    note = (f"found {len(matches)} selection-related HDF5 node(s); use the "
            "names to recreate Setup Selections in Creator (the schema is "
            "undocumented, never a write target)"
            if matches else
            "no HDF5 nodes matching 'selection' found; this EDEM version may "
            "store selections under a different name")
    return SelectionScan(True, note, matches)


# ---------------------------------------------------------------------------
# Configuration + manifest.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplicationConfig:
    """Everything one replication run needs; validate() fails fast with
    actionable messages before any disk work."""

    deck_path: Path
    transforms: tuple[RigidTransform, ...]        # one entry per copy
    geometry_patterns: tuple[str, ...] = ("*",)
    timestep_index: int = 0
    output_dir: Path = Path("replicated_geometries")
    name_template: str = "{name}__copy{idx:02d}"
    inspect_selections: bool = True
    backend_kind: str = "stl"                     # "stl" or "deck"
    material: str | None = None
    backup_deck: bool = True
    # clog advisor / optimizer
    optimize: bool = False
    gap_factor: float = 6.0
    particle_diameter: float | None = None        # None = try the deck
    particle_density: float = 2500.0
    shear_modulus: float = 1.0e8
    poisson: float = 0.25
    # motion replay
    motion_csv: Path | None = None
    motion_angle_unit: str = "deg"
    # factory / kinematics transfer
    copy_factories: bool = False
    copy_kinematics: bool = False

    def validate(self) -> None:
        if self.deck_path.suffix.lower() != ".dem":
            raise ConfigError(f"expected a .dem deck, got: {self.deck_path}")
        if not self.deck_path.is_file():
            raise ConfigError(f"deck not found: {self.deck_path}")
        if not self.transforms:
            raise ConfigError("at least one transform (offset/rotation) is required")
        n_ident = sum(1 for t in self.transforms if t.is_identity)
        if n_ident:
            LOG.warning("%d transform(s) are the identity; those copies will "
                        "sit exactly on top of their source geometry", n_ident)
        if self.timestep_index < 0:
            raise ConfigError("timestep index must be >= 0")
        if not self.geometry_patterns or not all(
                isinstance(p, str) and p.strip() for p in self.geometry_patterns):
            raise ConfigError("geometry patterns must be non-empty strings")
        if self.backend_kind not in ("stl", "deck"):
            raise ConfigError(f"unknown backend {self.backend_kind!r}")
        if self.backend_kind == "deck" and not self.material:
            raise ConfigError(
                "--write-deck requires --material NAME (createGeometry assigns "
                "the material at creation time; --list-materials shows valid names)")
        if self.motion_csv is not None and self.backend_kind != "deck":
            raise ConfigError("--motion-csv needs --write-deck: kinematics can "
                              "only be created inside the deck, not in STL files")
        if self.motion_angle_unit not in ("deg", "rad"):
            raise ConfigError("--motion-angles must be 'deg' or 'rad'")
        if self.copy_factories or self.copy_kinematics:
            if self.backend_kind != "deck":
                raise ConfigError(
                    "--copy-factories/--copy-kinematics need --write-deck: "
                    "factories and kinematics live inside the deck and cannot "
                    "be represented in STL files")
            if self.motion_csv is not None and self.copy_kinematics:
                raise ConfigError(
                    "--copy-kinematics and --motion-csv both create kinematics "
                    "on the copies; pick one of them")
            rotated = [i for i, t in enumerate(self.transforms, start=1)
                       if not np.allclose(t.rotation, _I3, atol=1e-12)]
            if rotated:
                raise ConfigError(
                    "--copy-factories/--copy-kinematics currently support "
                    "pure translations only (rotated copies would also need "
                    f"axis vectors rewritten); transform(s) {rotated} rotate. "
                    "Drop --rotate, or copy without these flags and adjust "
                    "in Creator.")
        if self.gap_factor <= 0:
            raise ConfigError("--gap-factor must be > 0")
        if self.particle_diameter is not None and self.particle_diameter <= 0:
            raise ConfigError("--particle-diameter must be > 0 (metres)")
        try:
            probe = self.name_template.format(name="probe", idx=1)
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(f"invalid name template {self.name_template!r}: {exc}") from exc
        if "probe" not in probe:
            raise ConfigError("name template must contain the {name} field")


@dataclass
class CopyRecord:
    source_geometry: str
    copy_name: str
    transform: dict[str, Any]
    destination: str
    n_vertices: int
    n_triangles: int


def write_manifest(path: Path, config: ReplicationConfig,
                   copies: list[CopyRecord], scan: SelectionScan | None,
                   snapshot: dict[str, Any], advice: dict[str, Any] | None,
                   gap_reports: list[dict[str, Any]],
                   kinematics_created: int,
                   transfer_report: dict[str, Any] | None = None) -> None:
    """The manifest is the handover document: where each copy came from and
    went, its exact transform, the deck snapshot, the clogging advice, the
    factory/kinematics transfer report, and a checklist of anything manual."""
    if config.backend_kind == "deck":
        checklist = [
            "Geometry and material were written into the deck automatically.",
            ("Section type flags were mirrored from the sources; verify "
             "physical/virtual type of the new geometries in Creator anyway."
             if config.copy_factories else
             "Verify physical/virtual type of the new geometries in Creator."),
            ("Factories were copied; open Creator and verify every parameter "
             "of the new factories (transfer_report lists what was mirrored)."
             if config.copy_factories else
             "Factories are NOT copied by default; rerun with "
             "--copy-factories, or recreate them in Creator."),
            ("Kinematics were copied from the source geometries; verify "
             "start/end points and POA in Creator."
             if config.copy_kinematics else
             "Kinematics were replayed from the motion CSV."
             if kinematics_created else
             "Recreate kinematics on copies of moving geometries (or rerun "
             "with --copy-kinematics / --motion-csv)."),
            "Recreate Setup Selections on the copies; candidate names appear "
            "under selection_scan.matches[].children.",
        ]
    else:
        checklist = [
            "EDEM Creator > File > Import Geometry: select all STL files at "
            "once, units = metres.",
            "Assign material and physical/virtual type to each copy (STL "
            "carries neither).",
            "Recreate kinematics on copies of moving geometries.",
            "Recreate Setup Selections on the copies; candidate names appear "
            "under selection_scan.matches[].children.",
        ]
    payload = {
        "tool": "edem_geometry_replicator",
        "author": "Olle Ericsson Gustafsson",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_deck": str(config.deck_path),
        "timestep_index": config.timestep_index,
        "backend": config.backend_kind,
        "material": config.material,
        "units": "metres; angles were specified in degrees",
        "deck_snapshot": snapshot,
        "copies": [asdict(c) for c in copies],
        "clog_advice": advice,
        "gap_reports": gap_reports,
        "kinematics_created": kinematics_created,
        "transfer_report": (transfer_report if transfer_report is not None
                            else {"performed": False}),
        "selection_scan": (
            {"performed": scan.performed, "note": scan.note, "matches": scan.matches}
            if scan is not None
            else {"performed": False, "note": "scan disabled via --no-inspect"}),
        "checklist": checklist,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    LOG.info("manifest written: %s", path)


# ---------------------------------------------------------------------------
# Pipeline: phase 1 reads everything with the deck open read-only, phase 2
# hands all jobs to the backend (which may reopen the deck in write mode).
# ---------------------------------------------------------------------------
_FILENAME_UNSAFE: Final = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    cleaned = _FILENAME_UNSAFE.sub("_", name).strip("._")
    return cleaned or "geometry"


def match_geometry_names(names: Sequence[str], patterns: Sequence[str]) -> list[str]:
    """Case-insensitive shell-style wildcards, deck order preserved."""
    lowered = [p.lower() for p in patterns]
    return [n for n in names
            if any(fnmatch.fnmatch(n.lower(), p) for p in lowered)]


class ReplicationPipeline:
    def __init__(self, config: ReplicationConfig) -> None:
        self._config = config

    def run(self) -> Path:
        cfg = self._config
        cfg.validate()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- phase 1: read -------------------------------------------------
        with EdemDeck(cfg.deck_path, cfg.timestep_index) as reader:
            all_names = reader.geometry_names()
            selected = match_geometry_names(all_names, cfg.geometry_patterns)
            if not selected:
                raise ConfigError(
                    f"no geometry matched patterns {list(cfg.geometry_patterns)}; "
                    f"available sections: {all_names}")
            meshes = {name: reader.mesh(name) for name in selected}
            snapshot = reader.snapshot()
            deck_diameters = reader.particle_diameters()

        meshes = {n: m for n, m in meshes.items() if m.n_triangles > 0}
        for name in selected:
            if name not in meshes:
                LOG.warning("skipping %r: empty mesh (0 triangles)", name)
        selected = [n for n in selected if n in meshes]
        if not meshes:
            raise DeckReadError("every selected geometry section is empty")

        particle = self._particle_spec(deck_diameters)
        advice = clogging_advice(particle, cfg.gap_factor) if particle else None
        transforms = cfg.transforms
        gap_reports: list[dict[str, Any]] = []
        if cfg.optimize:
            if particle is None:
                raise ConfigError(
                    "--optimize needs the particle size: pass "
                    "--particle-diameter (metres); it could not be read from "
                    "this deck automatically")
            combined = np.vstack([m.vertices for m in meshes.values()])
            transforms, gap_reports = optimise_transforms(
                transforms, combined, particle, cfg.gap_factor)

        LOG.info("replicating %d of %d geometry section(s), %d cop(y/ies) "
                 "each via the %s backend",
                 len(selected), len(all_names), len(transforms), cfg.backend_kind)

        jobs: list[CopyJob] = []
        used_names: set[str] = set()
        for name in selected:
            for idx, transform in enumerate(transforms, start=1):
                stem = sanitize_filename(cfg.name_template.format(name=name, idx=idx))
                # two source names can sanitize to the same stem; keep unique
                k, candidate = 1, stem
                while candidate in used_names:
                    candidate = f"{stem}_{k}"
                    k += 1
                used_names.add(candidate)
                jobs.append(CopyJob(name, candidate, transform, meshes[name]))

        # ---- phase 1.5: read-only scan for factories / kinematics -----------
        prescan: TransferPrescan | None = None
        if cfg.copy_factories or cfg.copy_kinematics:
            prescan = prescan_deck_tree(cfg.deck_path, all_names)
            if cfg.copy_factories:
                for name in selected:
                    n = len(prescan.factories_by_geom.get(name, ()))
                    LOG.info("%r has %d factor(y/ies) to copy", name, n)
            if cfg.copy_kinematics:
                for name in selected:
                    keys = prescan.kinematic_keys_by_geom.get(name, [])
                    if not keys:
                        LOG.warning("no kinematics subtree found under %r; "
                                    "check with --list-kinematics", name)

        # ---- phase 2: write ------------------------------------------------
        if cfg.backend_kind == "deck":
            motion = (load_motion_table(cfg.motion_csv, cfg.motion_angle_unit)
                      if cfg.motion_csv else None)
            factory_plan = (prescan.factories_by_geom
                            if prescan and cfg.copy_factories else None)
            known = ({f.display_name for f in prescan.all_factories}
                     if prescan else None)
            backend: Any = EdempyDeckWriteBackend(
                cfg.material or "", cfg.backup_deck, motion,
                factory_plan, known)
        else:
            backend = StlExportBackend()
        destinations = backend.write_batch(cfg.deck_path, cfg.output_dir, jobs)

        copies = [CopyRecord(job.source_name, job.copy_name,
                             job.transform.describe(), dest,
                             job.source_mesh.n_vertices, job.source_mesh.n_triangles)
                  for job, dest in zip(jobs, destinations)]

        # ---- phase 2.5: mirror factory params + kinematics (HDF5) -----------
        transfer_report: dict[str, Any] | None = None
        if cfg.backend_kind == "deck" and (cfg.copy_factories or cfg.copy_kinematics):
            copy_names = [d.split("::", 1)[1] for d in destinations]
            kin_jobs: list[tuple[str, str, FloatArray]] = []
            if cfg.copy_kinematics:
                kin_jobs = [(job.source_name, cname, job.transform.translation)
                            for job, cname in zip(jobs, copy_names)]
            transfer_report = apply_hdf5_transfers(
                cfg.deck_path, [*all_names, *copy_names],
                getattr(backend, "factory_plans_created", []),
                kin_jobs, mirror_section_flags=cfg.copy_factories)

        # scan after all deck handles are closed
        scan = scan_selections(cfg.deck_path) if cfg.inspect_selections else None
        if scan is not None:
            LOG.info("selection scan: %s", scan.note)

        manifest_path = cfg.output_dir / "replication_manifest.json"
        write_manifest(manifest_path, cfg, copies, scan, snapshot, advice,
                       gap_reports, backend.kinematics_created, transfer_report)
        return manifest_path

    def _particle_spec(self, deck_diameters: dict[str, float]) -> ParticleSpec | None:
        cfg = self._config
        diameter = cfg.particle_diameter
        if diameter is None and deck_diameters:
            name, diameter = max(deck_diameters.items(), key=lambda kv: kv[1])
            LOG.info("largest particle in deck: %r, diameter %.4g m", name, diameter)
        if diameter is None:
            return None
        return ParticleSpec(diameter, cfg.particle_density,
                            cfg.shear_modulus, cfg.poisson)


# ---------------------------------------------------------------------------
# Command line interface. Exit codes: 0 ok, 1 deck/environment problem,
# 2 user input problem.
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edem_geometry_replicator",
        description="Copy EDEM deck geometries a chosen distance away, as STL "
                    "files or straight into the deck, with optional motion "
                    "replay and a particle-based clogging check.")
    parser.add_argument("deck", type=Path, nargs="?", default=None,
                        help="path to the .dem deck (asked for interactively "
                             "when omitted in a terminal)")

    grp_a = parser.add_argument_group("style A: offsets and rotations")
    grp_a.add_argument("--distance", type=float, metavar="D",
                       help="place ONE copy D metres away along --direction "
                            "(shorthand for a single --offset)")
    grp_a.add_argument("--direction", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                       metavar=("DX", "DY", "DZ"),
                       help="direction for --distance (default +X; any length)")
    grp_a.add_argument("--offset", nargs=3, type=float, action="append",
                       metavar=("DX", "DY", "DZ"),
                       help="translation in metres for ONE copy; repeatable")
    grp_a.add_argument("--rotate", nargs=3, type=float, action="append",
                       metavar=("RX", "RY", "RZ"),
                       help="Euler rotation in DEGREES (about fixed global X, "
                            "then Y, then Z); repeatable; a single --rotate "
                            "broadcasts over all offsets")
    grp_a.add_argument("--pivot", nargs=3, type=float, metavar=("X", "Y", "Z"),
                       help="rotate about this point (default: origin)")
    grp_a.add_argument("--pivot-centroid", action="store_true",
                       help="rotate each mesh about its own centroid")
    grp_a.add_argument("--count", type=int, default=1,
                       help="repeat ONE base transform N times cumulatively "
                            "(row, circular array, or helix)")

    grp_b = parser.add_argument_group("style B: lattice")
    grp_b.add_argument("--lattice", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                       help="grid of copies (pure translations)")
    grp_b.add_argument("--spacing", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                       help="grid spacing in metres (required with --lattice)")
    grp_b.add_argument("--include-origin", action="store_true",
                       help="also emit the (0,0,0) grid copy")

    grp_c = parser.add_argument_group("style C: full control")
    grp_c.add_argument("--transforms-json", type=Path, metavar="FILE",
                       help="JSON list of per-copy transforms, see module docstring")

    grp_be = parser.add_argument_group("backend")
    grp_be.add_argument("--write-deck", action="store_true",
                        help="write copies directly into the .dem via the "
                             "documented EDEMpy editing API")
    grp_be.add_argument("--material", type=str, default=None,
                        help="material for the new geometries (required with "
                             "--write-deck)")
    grp_be.add_argument("--no-backup", action="store_true",
                        help="skip the automatic .dem backup before writing")
    grp_be.add_argument("--motion-csv", type=Path, metavar="FILE",
                        help="replay recorded motion onto the copies as "
                             "kinematics (CSV: timestep, body, x, y, z, psi, "
                             "theta, phi); needs --write-deck")
    grp_be.add_argument("--motion-angles", choices=("deg", "rad"), default="deg",
                        help="angle unit in the motion CSV (default deg)")
    grp_be.add_argument("--copy-factories", action="store_true",
                        help="also copy each source section's particle "
                             "factories to its copies (needs --write-deck; "
                             "pure translations only): a fresh factory is "
                             "attached via the documented deck.addFactory and "
                             "the original's parameters are mirrored onto it")
    grp_be.add_argument("--copy-kinematics", action="store_true",
                        help="also copy each source section's kinematics "
                             "(e.g. Linear Rotation) to its copies, with all "
                             "point fields shifted by the copy offset (needs "
                             "--write-deck; pure translations only)")

    grp_o = parser.add_argument_group("clogging check / optimizer")
    grp_o.add_argument("--optimize", action="store_true",
                       help="grow copy offsets until the gap to the original "
                            "clears --gap-factor times the particle diameter")
    grp_o.add_argument("--gap-factor", type=float, default=6.0,
                       help="required gap as a multiple of the particle "
                            "diameter (default 6, the arching rule of thumb)")
    grp_o.add_argument("--particle-diameter", type=float, metavar="D",
                       help="largest particle diameter in METRES (default: "
                            "read from the deck when possible)")
    grp_o.add_argument("--particle-density", type=float, default=2500.0,
                       help="particle density in kg/m3 (default 2500)")
    grp_o.add_argument("--shear-modulus", type=float, default=1.0e8,
                       help="particle shear modulus in Pa (default 1e8)")
    grp_o.add_argument("--poisson", type=float, default=0.25,
                       help="particle Poisson ratio (default 0.25)")

    parser.add_argument("--geometry", action="append", default=None,
                        metavar="PATTERN",
                        help="case-insensitive wildcard; repeatable (default: all)")
    parser.add_argument("--timestep", type=int, default=0,
                        help="timestep to read meshes from (default 0 = initial)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("replicated_geometries"))
    parser.add_argument("--name-template", default="{name}__copy{idx:02d}")
    parser.add_argument("--no-inspect", action="store_true",
                        help="skip the selection scan")
    parser.add_argument("--inspect-only", action="store_true",
                        help="only run the selection scan, then exit")
    parser.add_argument("--list-geometries", action="store_true",
                        help="list geometry sections in the deck, then exit")
    parser.add_argument("--list-materials", action="store_true",
                        help="list material names in the deck, then exit")
    parser.add_argument("--list-factories", action="store_true",
                        help="list the factories found in the deck (name, "
                             "owning geometry, parameter fields), then exit")
    parser.add_argument("--list-kinematics", action="store_true",
                        help="list the kinematics subtrees found under each "
                             "geometry section, then exit")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def _transforms_from_json(path: Path) -> tuple[RigidTransform, ...]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"could not read transforms JSON {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ConfigError("transforms JSON must be a non-empty list of objects")
    transforms = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ConfigError(f"transforms JSON entry {i} is not an object")
        pivot_raw = entry.get("pivot")
        pivot_centroid = pivot_raw == "centroid"
        transforms.append(RigidTransform.from_parts(
            offset=entry.get("offset", (0.0, 0.0, 0.0)),
            euler_deg=entry.get("euler_deg"),
            axis=entry.get("axis"),
            angle_deg=entry.get("angle_deg"),
            matrix=entry.get("matrix"),
            pivot=None if pivot_centroid else pivot_raw,
            pivot_centroid=pivot_centroid))
    return tuple(transforms)


def _distance_offset(distance: float, direction: Sequence[float]) -> list[float]:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        raise ConfigError("--direction has (near) zero length")
    return (distance * d / length).tolist()


def _transforms_from_style_a(args: argparse.Namespace) -> tuple[RigidTransform, ...]:
    """--offset/--rotate pairing: equal counts pair 1:1, a single --rotate
    broadcasts; --distance contributes one more offset; --count > 1 needs
    exactly one base transform and repeats it cumulatively."""
    offsets: list[list[float]] = list(args.offset or [])
    if args.distance is not None:
        offsets.append(_distance_offset(args.distance, args.direction))
    rotates: list[Any] = list(args.rotate or [])
    n = max(len(offsets), len(rotates))
    if not offsets:
        offsets = [[0.0, 0.0, 0.0]] * n
    if not rotates:
        rotates = [None] * n
    elif len(rotates) == 1 and n > 1:
        rotates = rotates * n
    if len(offsets) != len(rotates):
        raise ConfigError(
            f"{len(offsets)} offset(s) cannot be paired with "
            f"{len(rotates)} --rotate flag(s); give equal counts, or a single "
            "--rotate to broadcast")
    base = [RigidTransform.from_parts(offset=off, euler_deg=rot,
                                      pivot=args.pivot,
                                      pivot_centroid=args.pivot_centroid)
            for off, rot in zip(offsets, rotates)]
    if args.count > 1:
        if len(base) != 1:
            raise ConfigError("--count > 1 requires exactly one base transform")
        return cumulative_transforms(base[0], args.count)
    if args.count < 1:
        raise ConfigError("--count must be >= 1")
    return tuple(base)


def _ask(prompt: str) -> str:
    """input() that turns Ctrl+C / closed stdin into a clean ConfigError
    (exit code 2) instead of a traceback in the console window."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise ConfigError("avbrutet, ingen inmatning gavs") from None


def _parse_floats(raw: str) -> list[float]:
    """Space-separated numbers; Swedish decimal commas ('0,5') accepted."""
    return [float(part) for part in raw.replace(",", ".").split()]


def _ask_transforms_interactively() -> tuple[RigidTransform, ...]:
    """Friendly fallback when the script starts without any transform flags
    in a terminal (typically double-clicked via the .bat launcher). Asks for
    the offset and the number of copies, and loops until each answer parses
    instead of dying on the first typo. Copy k is placed at k * offset, the
    same semantics as --count on the command line."""
    while True:
        raw = _ask("Avstand att placera kopian pa, i meter "
                   "(ett tal for +X, eller 'dx dy dz'): ")
        try:
            values = _parse_floats(raw)
        except ValueError:
            print(f"  Kunde inte tolka {raw!r} som tal, forsok igen.")
            continue
        if len(values) == 1:
            offset = [values[0], 0.0, 0.0]
            break
        if len(values) == 3:
            offset = values
            break
        print("  Ange ETT tal (avstand langs +X) eller TRE (dx dy dz).")
    while True:
        raw = _ask("Hur manga kopior ska goras? [1]: ")
        if not raw:
            count = 1
            break
        try:
            count = int(raw)
        except ValueError:
            print(f"  Kunde inte tolka {raw!r} som ett heltal, forsok igen.")
            continue
        if count >= 1:
            break
        print("  Antalet kopior maste vara minst 1.")
    base = RigidTransform.translation_only(offset)
    if count == 1:
        return (base,)
    LOG.info("kopia k placeras pa k * %s m fran originalet, k = 1..%d",
             offset, count)
    return cumulative_transforms(base, count)


def _transforms_from_args(args: argparse.Namespace) -> tuple[RigidTransform, ...]:
    style_a = bool(args.offset or args.rotate or args.distance is not None)
    style_b = args.lattice is not None
    style_c = args.transforms_json is not None
    chosen = sum([style_a, style_b, style_c])
    if chosen == 0:
        if sys.stdin.isatty():
            return _ask_transforms_interactively()
        raise ConfigError("no transform given: use --distance, "
                          "--offset/--rotate, --lattice, or --transforms-json")
    if chosen > 1:
        raise ConfigError("pick ONE transform style: --distance/--offset/"
                          "--rotate, OR --lattice, OR --transforms-json")
    if style_b:
        if args.spacing is None:
            raise ConfigError("--lattice requires --spacing DX DY DZ")
        if args.pivot is not None or args.pivot_centroid:
            raise ConfigError("--pivot flags apply to --rotate, not --lattice")
        return lattice_transforms(args.lattice, args.spacing, args.include_origin)
    if style_c:
        if args.pivot is not None or args.pivot_centroid:
            raise ConfigError("--pivot flags do not combine with "
                              "--transforms-json; set the pivot per entry")
        return _transforms_from_json(args.transforms_json)
    return _transforms_from_style_a(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s")
    try:
        if args.deck is None:
            # Run from an IDE's Run button (no arguments) this used to die
            # with a bare SystemExit; asking is friendlier.
            if not sys.stdin.isatty():
                raise ConfigError("missing deck argument: pass the path to "
                                  "the .dem file")
            while True:
                raw = _ask("Sokvag till .dem-filen (dra garna in filen "
                           "i det har fonstret): ").strip('"')
                if not raw:
                    print("  Ingen sokvag angiven, forsok igen.")
                    continue
                candidate = Path(raw).expanduser()
                if candidate.is_file():
                    args.deck = candidate
                    break
                print(f"  Hittar ingen fil pa {raw!r}, forsok igen.")
        deck_path = args.deck.expanduser().resolve()

        if args.list_geometries or args.list_materials:
            with EdemDeck(deck_path, args.timestep) as reader:
                if args.list_geometries:
                    for name in reader.geometry_names():
                        LOG.info("geometry section: %s", name)
                if args.list_materials:
                    for name in reader.material_names():
                        LOG.info("material: %s", name)
            return 0
        if args.list_factories or args.list_kinematics:
            with EdemDeck(deck_path, args.timestep) as reader:
                names = reader.geometry_names()
            prescan = prescan_deck_tree(deck_path, names)
            if args.list_factories:
                if not prescan.all_factories:
                    LOG.info("no factories found; if the deck definitely has "
                             "factories, the HDF5 layout differs from the "
                             "known pattern -- run --inspect-only and share "
                             "the output")
                for info in prescan.all_factories:
                    LOG.info("factory %r on geometry %s at %s",
                             info.display_name, info.geometry or "<unknown>",
                             info.path)
                    for key, value in info.string_fields.items():
                        LOG.info("    %s = %r", key, value)
            if args.list_kinematics:
                for name in names:
                    keys = prescan.kinematic_keys_by_geom.get(name, [])
                    LOG.info("geometry %r: kinematics subtrees %s",
                             name, keys or "<none found>")
            return 0
        if args.inspect_only:
            scan = scan_selections(deck_path)
            LOG.info("%s", scan.note)
            for entry in scan.matches:
                LOG.info("  %s", entry)
            return 0

        config = ReplicationConfig(
            deck_path=deck_path,
            transforms=_transforms_from_args(args),
            geometry_patterns=tuple(args.geometry) if args.geometry else ("*",),
            timestep_index=args.timestep,
            output_dir=args.output_dir,
            name_template=args.name_template,
            inspect_selections=not args.no_inspect,
            backend_kind="deck" if args.write_deck else "stl",
            material=args.material,
            backup_deck=not args.no_backup,
            optimize=args.optimize,
            gap_factor=args.gap_factor,
            particle_diameter=args.particle_diameter,
            particle_density=args.particle_density,
            shear_modulus=args.shear_modulus,
            poisson=args.poisson,
            motion_csv=args.motion_csv,
            motion_angle_unit=args.motion_angles,
            copy_factories=args.copy_factories,
            copy_kinematics=args.copy_kinematics)
        manifest = ReplicationPipeline(config).run()
        LOG.info("done; see %s for the checklist", manifest)
        return 0
    except ConfigError as exc:
        LOG.error("configuration error: %s", exc)
        return 2
    except ReplicatorError as exc:
        LOG.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())