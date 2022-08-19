#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Built-in imports
import argparse
import csv
import os

# 3rd party imports
from bbfsdfs.load import load_fpi
from bbfsdfs.utils import find_bbfs
from pyrfu import pyrf

__author__ = "Louis Richard"
__email__ = "louisr@irfu.se"
__copyright__ = "Copyright 2022"
__license__ = "Apache 2.0"


def main(args):
    tint = [f"{args.day}T00:00:00.000", f"{args.day}T23:59:59.000"]

    path = os.path.join(os.path.pardir, "database", *args.day.split("-"))
    os.makedirs(path, exist_ok=True)
    _, v_gsm_i, _, _ = load_fpi(tint)

    times, tints = find_bbfs(v_gsm_i, args.threshold, args.direction)

    if not tints:
        tints = [[None, None]]

    with open(os.path.join(path, f"{pyrf.date_str(tint, 3)}_earthward.csv"), "w") as f:
        write = csv.writer(f)
        write.writerows(tints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "direction",
        help="Direction of the flow to search",
        choices=["earthward", "tailward"],
        type=str,
    )
    parser.add_argument("day", help="Day to look for BBFs", type=str)
    parser.add_argument(
        "--threshold", "-t", help="Velocity threshold", type=float, default=300.0
    )

    main(parser.parse_args())
