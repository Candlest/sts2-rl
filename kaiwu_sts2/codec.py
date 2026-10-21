"""msgpack codec with numpy support (kaiwu default protocol is msgpack)."""

from __future__ import annotations

import numpy as np
import msgpack


def _encode(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__nd__": True,
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
            b"data": obj.tobytes(),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"cannot encode {type(obj)}")


def _decode(obj):
    if obj.get(b"__nd__"):
        arr = np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))
        return arr.reshape(obj[b"shape"])
    return obj


def packb(obj) -> bytes:
    return msgpack.packb(obj, default=_encode, use_bin_type=True)


def unpackb(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)
