"""
Data access for astron_mi.

Two sources, deliberately in this priority order:

1. THE AUTHORS' TEST FIXTURES (`aion-repo/tests/test_data/`). Real survey data with
   reference tokens: 64 Legacy stamps (9, 96, 96) with channel masks, 128 DESI spectra
   on the native 7800-point grid, plus encoded/decoded references for ~37 modalities.
   Small, instant, and they double as a regression suite. Enough for C0 and for most of
   the mechanics of B1/D0. Start here.

2. `data/provabgs_desi_ls.hdf5` — 121 GB, PROVABGS galaxy properties + DESI spectra +
   Legacy imaging. Needed for anything requiring physical labels at scale.

   The AION Tutorial notebook's loader is BROKEN for the current file: it does
   `Table({k: np.array(f[k]) for k in f.keys()})`, which assumes flat keys and would
   try to materialise 121 GB. The real layout is three nested groups:
   `desi`, `legacysurvey`, `provabgs_desi_edr_sv3`. ProvabgsReader below reads lazily
   by row and never materialises the file.

h5py is NOT installed in the astron-mi venv. ProvabgsReader raises a clear error saying
so; install it there, or use the anaconda python which already has it.
"""

from __future__ import annotations

import os
import pathlib
from typing import Iterator, Sequence

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = pathlib.Path(
    os.environ.get(
        "AION_TEST_DATA",
        "/private/tmp/claude-501/-Volumes-Crucial-X6-MacBook-Code-astron-mi"
        "/90bc6a25-1a2c-4caa-84ed-b5000b74adb2/scratchpad/aion-repo/tests/test_data",
    )
)
PROVABGS = ROOT / "data" / "provabgs_desi_ls.hdf5"


#: PROVABGS uses -999 as its missing-value sentinel. Several columns
#: (PROVABGS_LOGMSTAR_BF, n_dust, f_burst, MAG_R, SCHLEGEL_COLOR, f_fiber) carry it.
#: Left unmasked it silently poisons every probe and correlation.
MISSING_SENTINELS = (-999.0, -9999.0)


def mask_sentinels(a: np.ndarray, sentinels=MISSING_SENTINELS) -> np.ndarray:
    """Replace missing-value sentinels with NaN."""
    a = np.asarray(a, dtype="float64").copy()
    for s in sentinels:
        a[np.isclose(a, s)] = np.nan
    return a


def _load(name: str):
    p = FIXTURES / name
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Set AION_TEST_DATA to the AION repo's tests/test_data dir "
            "(clone https://github.com/PolymathicAI/AION if needed)."
        )
    return torch.load(p, weights_only=False)


# ------------------------------------------------------------------- fixtures
def load_fixture_images() -> dict[str, torch.Tensor]:
    """64 real Legacy Survey stamps. array (64, 9, 96, 96), channel_mask (64, 9).

    The 9 channels are the union frame the image codec always works in
    (Legacy grz+W1.. and HSC grizy slots); channel_mask says which are populated.
    """
    d = _load("image_codec_input_batch.pt")["image"]
    return {"array": d["array"], "channel_mask": d["channel_mask"]}


def load_fixture_spectra() -> dict[str, torch.Tensor]:
    """128 real DESI spectra: flux/ivar/lsf_sigma/lambda/mask, each (128, 7800)."""
    return dict(_load("SPECTRUM_input_batch.pt")["spectrum"])


def reference_tokens(which: str) -> torch.Tensor:
    """Authors' reference token tensors. 'image' -> (64, 24, 24) int32,
    'spectrum' -> (128, 273) int64."""
    return _load(
        {"image": "image_codec_encoded_batch.pt", "spectrum": "SPECTRUM_encoded_batch.pt"}[which]
    )


def check_regression(codec_manager=None, verbose: bool = True) -> dict[str, bool]:
    """
    Assert that this install reproduces the authors' reference tokens bit-exactly.

    This is the guard against the version skew: the installed polymathic-aion 0.0.2 lacks
    repo HEAD's `pad_spectrum`/`pad_length`, and the two produce different spectrum tokens.
    0.0.2 matches these fixtures exactly (verified: 0/273 tokens differ). Run this before
    trusting any wavelength-resolved result.
    """
    from aion.modalities import DESISpectrum

    from .loading import load_codecs

    cm = codec_manager or load_codecs("cpu")
    out = {}

    sp = load_fixture_spectra()
    got = cm.encode(
        DESISpectrum(
            flux=sp["flux"].clone(),
            ivar=sp["ivar"].clone(),
            mask=sp["mask"].clone().bool(),
            wavelength=sp["lambda"].clone(),
        )
    )
    got = list(got.values())[0] if isinstance(got, dict) else got
    ref = reference_tokens("spectrum")
    out["spectrum"] = bool((got.reshape(ref.shape).long() == ref.long()).all())

    if verbose:
        for k, v in out.items():
            print(f"  regression {k}: {'PASS' if v else 'FAIL'}")
        if not all(out.values()):
            print("  -> token generation does not match the reference. Check the aion version.")
    return out


# ------------------------------------------------------------------ provabgs
class ProvabgsReader:
    """
    Lazy row reader for provabgs_desi_ls.hdf5 (121 GB, three nested groups).

    Never materialises the file. Open once, index rows, close. Usage:

        with ProvabgsReader() as r:
            print(r.schema())
            batch = r.rows(range(16))
    """

    def __init__(self, path: os.PathLike | str = PROVABGS):
        self.path = pathlib.Path(path)
        self._f = None

    def __enter__(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is not installed in this environment. Either\n"
                "  pip install h5py    (run from a neutral cwd, LC_ALL=en_US.UTF-8)\n"
                "or use a python that has it."
            ) from exc
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} not found (121 GB download).")
        self._f = h5py.File(self.path, "r")
        return self

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
            self._f = None
        return False

    def schema(self, max_depth: int = 3) -> dict:
        """Full group/dataset tree with shapes and dtypes."""
        import h5py

        tree: dict = {}

        def walk(node, prefix, depth):
            if depth > max_depth:
                return
            for k in node.keys():
                obj = node[k]
                path = f"{prefix}/{k}" if prefix else k
                if isinstance(obj, h5py.Group):
                    tree[path] = {"type": "group", "n_keys": len(obj.keys())}
                    walk(obj, path, depth + 1)
                else:
                    tree[path] = {
                        "type": "dataset",
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                    }

        walk(self._f, "", 0)
        return tree

    def n_rows(self) -> int:
        """Row count, inferred from the first dataset found."""
        for meta in self.schema().values():
            if meta["type"] == "dataset" and meta["shape"]:
                return int(meta["shape"][0])
        raise RuntimeError("no datasets found")

    def rows(self, idx: Sequence[int], paths: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """Read specific rows. `paths` are '/'-joined dataset paths from schema()."""
        idx = np.asarray(sorted(set(int(i) for i in idx)))
        if paths is None:
            paths = [p for p, m in self.schema().items() if m["type"] == "dataset"]
        out = {}
        for p in paths:
            ds = self._f[p]
            try:
                out[p] = ds[idx]
            except Exception:
                out[p] = np.stack([ds[i] for i in idx])
        return out

    def iter_batches(self, batch_size: int = 16, limit: int | None = None,
                     paths: Sequence[str] | None = None) -> Iterator[dict]:
        n = self.n_rows() if limit is None else min(limit, self.n_rows())
        for start in range(0, n, batch_size):
            yield self.rows(range(start, min(start + batch_size, n)), paths)

    # ------------------------------------------------------- AION-ready batches
    #: PROVABGS physical labels worth probing for. Keys are short names.
    LABELS = {
        "z": "provabgs_desi_edr_sv3/Z_HP",
        "log_mstar": "provabgs_desi_edr_sv3/PROVABGS_LOGMSTAR_BF",
        "log_Z_mw": "provabgs_desi_edr_sv3/LOG_Z_MW",
        "t_age_mw": "provabgs_desi_edr_sv3/TAGE_MW",
        "log_sSFR": "provabgs_desi_edr_sv3/LOG_sSFR",
        "avg_sfr": "provabgs_desi_edr_sv3/avgSFR",
        "n_dust": "provabgs_desi_edr_sv3/n_dust",
        "f_burst": "provabgs_desi_edr_sv3/f_burst",
        "mag_r": "provabgs_desi_edr_sv3/MAG_R",
        "schlegel_color": "provabgs_desi_edr_sv3/SCHLEGEL_COLOR",
        # depth / observing-condition proxies available in-file. The full
        # PSFSIZE_*/PSFDEPTH_*/GALDEPTH_* set needed for the mandatory systematics
        # control is NOT here — it comes from the Legacy Survey sweep catalogues,
        # joined on (ra, dec).
        "tsnr2_bgs": "provabgs_desi_edr_sv3/TSNR2_BGS",
        "ebv": "legacysurvey/EBV",
        "f_fiber": "provabgs_desi_edr_sv3/f_fiber",
    }

    SCALARS = {  # AION scalar modalities present in-file
        "FLUX_G": "legacysurvey/FLUX_G", "FLUX_R": "legacysurvey/FLUX_R",
        "FLUX_I": "legacysurvey/FLUX_I", "FLUX_Z": "legacysurvey/FLUX_Z",
        "FLUX_W1": "legacysurvey/FLUX_W1", "FLUX_W2": "legacysurvey/FLUX_W2",
        "FLUX_W3": "legacysurvey/FLUX_W3", "FLUX_W4": "legacysurvey/FLUX_W4",
        "EBV": "legacysurvey/EBV",
    }

    def aion_batch(self, idx: Sequence[int], with_image: bool = True,
                   with_spectrum: bool = True, with_segmap: bool = False) -> dict:
        """
        Read rows and return tensors ready for astron_mi.encode_{image,spectrum}.

        Notes on the raw arrays:
        - `legacysurvey/image_array` is (N, 4, 160, 160); the codec centre-crops to 96.
        - `legacysurvey/image_band` gives the per-object band names as bytes; they are
          passed through so the bands[i] <-> channel i contract is never guessed.
        - `desi/spectrum_*` is 7781 pixels with NO sentinel. encode_spectrum appends it.
        - `legacysurvey/object_mask` is the pipeline segmentation map, the input to the
          D2 "does flux follow the mask or the light?" intervention.
        """
        want = ["desi/spectrum_flux", "desi/spectrum_ivar", "desi/spectrum_lambda",
                "desi/spectrum_mask"] if with_spectrum else []
        if with_image:
            want += ["legacysurvey/image_array", "legacysurvey/image_band"]
        if with_segmap:
            want += ["legacysurvey/object_mask"]
        want += list(self.LABELS.values()) + list(self.SCALARS.values())
        want += ["legacysurvey/ra", "legacysurvey/dec"]
        raw = self.rows(idx, sorted(set(want)))

        out: dict = {"n": len(idx), "labels": {}, "scalars": {}}
        for short, path in self.LABELS.items():
            if path in raw:
                out["labels"][short] = mask_sentinels(np.asarray(raw[path]))
        for short, path in self.SCALARS.items():
            if path in raw:
                out["scalars"][short] = np.asarray(raw[path])
        out["ra"] = np.asarray(raw["legacysurvey/ra"])
        out["dec"] = np.asarray(raw["legacysurvey/dec"])

        if with_spectrum:
            out["spectrum"] = {
                "flux": torch.from_numpy(np.asarray(raw["desi/spectrum_flux"])).float(),
                "ivar": torch.from_numpy(np.asarray(raw["desi/spectrum_ivar"])).float(),
                "mask": torch.from_numpy(np.asarray(raw["desi/spectrum_mask"])).bool(),
                "wavelength": torch.from_numpy(np.asarray(raw["desi/spectrum_lambda"])).float(),
            }
        if with_image:
            arr = np.asarray(raw["legacysurvey/image_array"])
            bands_raw = np.asarray(raw["legacysurvey/image_band"])
            bands = [b.decode() if isinstance(b, bytes) else str(b) for b in bands_raw[0]]
            out["image"] = {"flux": torch.from_numpy(arr).float(), "bands": bands}
        if with_segmap:
            out["segmap"] = torch.from_numpy(
                np.asarray(raw["legacysurvey/object_mask"])
            ).float()
        return out
