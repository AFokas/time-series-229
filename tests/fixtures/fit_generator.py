"""Minimal FIT binary file generator for testing.

Implements just enough of the Garmin FIT format to produce valid .fit and
.fit.gz files that fitparse can parse — specifically files containing only
`record` messages (global message number 20) with the fields used by the
pipeline.

FIT file structure
------------------
1. File header (14 bytes)
2. Definition record for local message type 0 → global message 20 (record)
3. N data records
4. File CRC (2 bytes, little-endian)

All multi-byte integers are little-endian unless noted otherwise.
Timestamps are seconds since the FIT epoch: 1989-01-01 00:00:00 UTC.
"""

from __future__ import annotations

import gzip
import io
import struct
from datetime import datetime, timezone

import numpy as np

# Seconds between Unix epoch (1970-01-01) and FIT epoch (1989-01-01)
FIT_EPOCH_OFFSET = 631065600

# CRC table copied from fitparse.records.Crc
_CRC_TABLE = (
    0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
    0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400,
)


def _crc16(data: bytes, crc: int = 0) -> int:
    for byte in data:
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[byte & 0xF]
        tmp = _CRC_TABLE[crc & 0xF]
        crc = (crc >> 4) & 0x0FFF
        crc = crc ^ tmp ^ _CRC_TABLE[(byte >> 4) & 0xF]
    return crc


# ---- FIT field definitions --------------------------------------------------
# Each entry: (def_num, fmt, base_type_byte, scale, offset)
#   scale/offset: how raw ints relate to real-world values
#   raw = (value + offset) * scale   →   value = raw / scale - offset
#
# def_num 253 = timestamp (uint32)  raw = unix_seconds - FIT_EPOCH_OFFSET
# def_num   0 = position_lat  (sint32 semicircles, scale=(2^31)/180)
# def_num   1 = position_long (sint32 semicircles)
# def_num   2 = altitude      (uint16, scale=5, offset=500)  meters
# def_num   3 = heart_rate    (uint8)   bpm
# def_num   4 = cadence       (uint8)   spm per foot (×2 for total)
# def_num   5 = distance      (uint32, scale=100)  meters
# def_num   6 = speed         (uint16, scale=1000) m/s
# def_num  13 = temperature   (sint8)   °C
# def_num  78 = enhanced_altitude (uint32, scale=5, offset=500)  meters

SEMICIRCLES_PER_DEGREE = (2**31) / 180.0

FIELD_DEFS = [
    # (def_num, struct_fmt, base_type_byte)
    (253, "I", 0x86),   # timestamp  uint32
    (0,   "i", 0x85),   # position_lat sint32
    (1,   "i", 0x85),   # position_long sint32
    (2,   "H", 0x84),   # altitude  uint16
    (3,   "B", 0x02),   # heart_rate uint8
    (4,   "B", 0x02),   # cadence uint8
    (5,   "I", 0x86),   # distance  uint32
    (6,   "H", 0x84),   # speed  uint16
    (13,  "b", 0x01),   # temperature sint8
    (78,  "I", 0x86),   # enhanced_altitude uint32
]

_FMT_SIZE = {"B": 1, "b": 1, "H": 2, "h": 2, "I": 4, "i": 4}


def _build_definition_record() -> bytes:
    """Build a FIT definition record for local message type 0 = global message 20 (record)."""
    # Record header: bit7=0, bit6=1 (definition), bits5-0=local mesg type 0
    header = 0x40
    reserved = 0
    architecture = 0  # little-endian
    global_mesg_num = 20  # record
    n_fields = len(FIELD_DEFS)

    msg = struct.pack("<BBBHB", header, reserved, architecture, global_mesg_num, n_fields)
    for def_num, fmt, base_type in FIELD_DEFS:
        size = _FMT_SIZE[fmt]
        msg += struct.pack("BBB", def_num, size, base_type)
    return msg


def _encode_row(
    timestamp_unix: float,
    lat_deg: float,
    lon_deg: float,
    altitude_m: float,
    heart_rate_bpm: float,
    cadence_spm_per_foot: float,
    distance_m: float,
    speed_ms: float,
    temperature_c: float,
) -> bytes:
    """Encode one data record."""
    ts = int(timestamp_unix - FIT_EPOCH_OFFSET)

    lat = int(lat_deg * SEMICIRCLES_PER_DEGREE)
    lon = int(lon_deg * SEMICIRCLES_PER_DEGREE)

    alt_raw = int((altitude_m + 500) * 5)
    alt_raw = max(0, min(0xFFFE, alt_raw))

    hr = int(np.clip(heart_rate_bpm, 0, 254))
    cad = int(np.clip(cadence_spm_per_foot, 0, 254))

    dist_raw = int(distance_m * 100)
    speed_raw = int(np.clip(speed_ms * 1000, 0, 0xFFFE))
    temp = int(np.clip(temperature_c, -127, 126))

    enh_alt_raw = int((altitude_m + 500) * 5)
    enh_alt_raw = max(0, min(0xFFFFFFFE, enh_alt_raw))

    # Data record header: bit7=0, bit6=0, bits5-0=local mesg type 0
    header = 0x00
    payload = struct.pack(
        "<B" + "".join(f for _, f, _ in FIELD_DEFS),
        header,
        ts, lat, lon, alt_raw, hr, cad, dist_raw, speed_raw, temp, enh_alt_raw,
    )
    return payload


def _build_fit_bytes(data_bytes: bytes) -> bytes:
    """Wrap data records in a valid FIT file (header + data + CRC)."""
    data_size = len(data_bytes)

    # File header: 14 bytes
    # headerSize(1) protocolVer(1) profileVer(2) dataSize(4) ".FIT"(4) crc(2)
    header_without_crc = struct.pack("<BBHI4s", 14, 0x10, 2132, data_size, b".FIT")
    header_crc = _crc16(header_without_crc)
    header = header_without_crc + struct.pack("<H", header_crc)

    # File CRC over data bytes
    file_crc = _crc16(data_bytes)
    return header + data_bytes + struct.pack("<H", file_crc)


def make_fit_bytes(
    n_seconds: int = 3600,
    start_time: datetime | None = None,
    base_lat: float = 51.5,
    base_lon: float = -0.1,
    base_alt_m: float = 50.0,
    base_speed_ms: float = 3.3,         # ~5:03/km easy pace
    base_hr_bpm: float = 140.0,
    base_cadence: float = 88,            # per foot; ×2 = 176 spm total
    temperature_c: float = 15.0,
    pace_variation: float = 0.15,        # fraction for random speed variation
    hr_drift: float = 0.005,             # bpm/s drift to simulate cardiac drift
    interval_profile: list[tuple[int, float, float]] | None = None,
    seed: int = 42,
) -> bytes:
    """Generate raw FIT file bytes for a synthetic run.

    Parameters
    ----------
    n_seconds:          total run duration in seconds
    start_time:         UTC datetime for first record (default: 2025-08-01 07:00)
    base_speed_ms:      average speed in m/s
    interval_profile:   list of (start_s, end_s, speed_multiplier) tuples
                        to overlay interval efforts on the base pace.
    Returns raw bytes of a valid FIT file.
    """
    if start_time is None:
        start_time = datetime(2025, 8, 1, 7, 0, 0, tzinfo=timezone.utc)

    start_unix = start_time.timestamp()
    rng = np.random.default_rng(seed)

    # Build interval lookup
    interval_zones: dict[int, float] = {}
    if interval_profile:
        for start_s, end_s, mult in interval_profile:
            for s in range(int(start_s), int(end_s)):
                interval_zones[s] = mult

    def_record = _build_definition_record()
    data_records = bytearray()
    data_records += def_record

    cumulative_dist = 0.0
    for s in range(n_seconds):
        ts = start_unix + s
        mult = interval_zones.get(s, 1.0)
        speed = base_speed_ms * mult * (1.0 + rng.normal(0, pace_variation * 0.1))
        speed = max(0.5, speed)

        hr_base = base_hr_bpm + s * hr_drift
        if mult > 1.2:
            hr_base += (mult - 1.0) * 30
        hr = hr_base + rng.normal(0, 1.5)
        hr = np.clip(hr, 60, 200)

        cumulative_dist += speed  # 1 second per sample
        alt = base_alt_m + 0.5 * np.sin(s / 600) + rng.normal(0, 0.02)
        lat = base_lat + cumulative_dist * 1e-5 * np.cos(rng.uniform(0, 2 * np.pi))
        lon = base_lon + cumulative_dist * 1e-5 * np.sin(rng.uniform(0, 2 * np.pi))

        cad = base_cadence * (0.9 + 0.2 * mult) + rng.normal(0, 1)
        cad = int(np.clip(cad, 40, 120))

        data_records += _encode_row(ts, lat, lon, alt, hr, cad, cumulative_dist, speed, temperature_c)

    return _build_fit_bytes(bytes(data_records))


def make_fit_gz(
    output_path: str | None = None,
    **kwargs,
) -> bytes:
    """Return gzip-compressed FIT bytes (and optionally write to a file)."""
    raw = make_fit_bytes(**kwargs)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(raw)
    gz_bytes = buf.getvalue()
    if output_path:
        with open(output_path, "wb") as f:
            f.write(gz_bytes)
    return gz_bytes


def make_interval_run_fit_gz(
    n_warmup_s: int = 600,
    n_intervals: int = 5,
    interval_duration_s: int = 300,
    recovery_duration_s: int = 180,
    n_cooldown_s: int = 600,
    speed_easy_ms: float = 3.0,
    speed_hard_ms: float = 4.5,
    **kwargs,
) -> bytes:
    """Generate a synthetic interval session FIT file (gzip'd)."""
    profile = []
    t = n_warmup_s
    for _ in range(n_intervals):
        profile.append((t, t + interval_duration_s, speed_hard_ms / speed_easy_ms))
        t += interval_duration_s + recovery_duration_s

    total = t + n_cooldown_s
    return make_fit_gz(
        n_seconds=total,
        base_speed_ms=speed_easy_ms,
        interval_profile=profile,
        **kwargs,
    )


def make_long_run_fit_gz(
    distance_km: float = 30.0,
    pace_min_per_km: float = 4.5,
    **kwargs,
) -> bytes:
    """Generate a synthetic long run FIT file (gzip'd)."""
    speed_ms = 1000.0 / (pace_min_per_km * 60.0)
    n_seconds = int(distance_km * 1000 / speed_ms)
    return make_fit_gz(n_seconds=n_seconds, base_speed_ms=speed_ms, **kwargs)
