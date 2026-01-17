#!/usr/bin/env python3

import os
import re

def extract_chip_area(filepath: str) -> float | None:
    """Extract chip area from Yosys area report file."""
    expanded_path = os.path.expandvars(filepath)
    if not os.path.exists(expanded_path):
        print(f"Warning: File not found: {expanded_path}")
        return None

    pattern = re.compile(r"Chip area for top module '\\(\w+)':\s+([\d.]+)")

    with open(expanded_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                module_name = match.group(1)
                area = float(match.group(2))
                return area
    return None


def format_area(area: float | None) -> str:
    """Format area value with thousands separator."""
    if area is None:
        return "N/A"
    return f"{area:.2f}"


if __name__ == '__main__':
    # Check for APS_VLSI env
    if 'APS_VLSI' not in os.environ:
        print("APS environment variable is not set. Please run `pixi shell` first.")
        exit(1)

    YOSYS_REPORT_PATH = '$APS_VLSI/yosys/default/reports/RocketTile_area.rpt'
    YOSYS_REPORT_PATH_ROCC = '$APS_VLSI/yosys/default/reports/RocketTile_area_rocc.rpt'

    # Extract chip area from both files
    area_all = extract_chip_area(YOSYS_REPORT_PATH)
    area_rocc = extract_chip_area(YOSYS_REPORT_PATH_ROCC)

    area_base_fmt = format_area(area_all - area_rocc) if area_all is not None and area_rocc is not None else None
    area_rocc_fmt = format_area(area_rocc) if area_rocc is not None else None
    if area_base_fmt is None or area_rocc_fmt is None:
        print("Synthesis failed, could not extract area information from reports.")
        exit(1)

    overhead = area_rocc / (area_all - area_rocc) * 100 if area_all is not None and area_rocc is not None else None

    print("Synthesis report\n=============================")
    print(f"Original processor’s area: {area_base_fmt} um²")
    print(f"Your design’s area: {area_rocc_fmt} um² ({overhead:+.2f}% overhead)")
    print("=============================")