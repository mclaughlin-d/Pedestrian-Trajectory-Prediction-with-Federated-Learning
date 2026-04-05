#!/usr/bin/env python3
"""
Compute per-row pedestrian velocity from trajectory .txt files.

Input format (tab separated):
    frame_id  ped_id  x  y

Velocity is the Euclidean distance between a pedestrian's current and previous
position divided by the frame difference.  Rows that are the first observation
of a pedestrian receive velocity = NaN.

Output file: <input_stem>_wv.txt  (same directory as input)
Extra column appended: velocity
"""

import sys
import os
import math
import statistics


def compute_velocities(input_path: str) -> None:
    # ── read ────────────────────────────────────────────────────────────────
    rows = []
    with open(input_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            frame_id = int(float(parts[0]))
            ped_id = float(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            rows.append((frame_id, ped_id, x, y))

    # ── compute velocity row by row ─────────────────────────────────────────
    # last_pos[ped_id] = (frame_id, x, y)
    last_pos: dict[float, tuple[int, float, float]] = {}
    output_rows = []

    for frame_id, ped_id, x, y in rows:
        if ped_id in last_pos:
            prev_frame, px, py = last_pos[ped_id]
            dt = frame_id - prev_frame
            if dt > 0:
                vel = math.sqrt((x - px) ** 2 + (y - py) ** 2) / dt
            else:
                vel = float("nan")
        else:
            vel = float("nan")

        last_pos[ped_id] = (frame_id, x, y)
        output_rows.append((frame_id, ped_id, x, y, vel))

    # ── write output file ───────────────────────────────────────────────────
    stem, _ = os.path.splitext(input_path)
    output_path = stem + "_wv.txt"

    with open(output_path, "w") as fh:
        for frame_id, ped_id, x, y, vel in output_rows:
            vel_str = f"{vel:.6f}" if not math.isnan(vel) else "nan"
            fh.write(f"{frame_id}\t{ped_id:.1f}\t{x}\t{y}\t{vel_str}\n")

    print(f"Wrote {output_path}")

    # ── per-frame statistics ────────────────────────────────────────────────
    frame_vels: dict[int, list[float]] = {}
    for frame_id, _, _, _, vel in output_rows:
        if not math.isnan(vel):
            frame_vels.setdefault(frame_id, []).append(vel)

    print(f"\n{'Frame':>8}  {'Avg Vel':>10}  {'Max Vel':>10}  {'Median Vel':>12}  {'N':>4}")
    print("-" * 54)
    for frame_id in sorted(frame_vels):
        vels = frame_vels[frame_id]
        avg = statistics.mean(vels)
        mx = max(vels)
        med = statistics.median(vels)
        print(f"{frame_id:>8}  {avg:>10.4f}  {mx:>10.4f}  {med:>12.4f}  {len(vels):>4}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    compute_velocities(input_path)
