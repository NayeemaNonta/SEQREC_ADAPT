#!/usr/bin/env python3
"""
prepare_30music_csv.py

Convert data/ThirtyMusic/relations/events.idomaar to data_csv/30M.csv
in the format expected by the Does-It-Look-Sequential repo:

    user,item,timestamp

No filtering is applied here — the paper's preprocessing.py handles
5-core filtering, deduplication, and ID encoding.
"""

import json
import os

EVENTS_FILE = "data/ThirtyMusic/relations/events.idomaar"
OUT_DIR     = "data/data_csv"
OUT_FILE    = os.path.join(OUT_DIR, "30M.csv")
MIN_PLAYTIME = 1

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Parsing {EVENTS_FILE} ...")
written = 0
skipped = 0

with open(EVENTS_FILE, "r", encoding="utf-8") as src, \
     open(OUT_FILE, "w", encoding="utf-8") as dst:

    dst.write("user,item,timestamp\n")

    for lineno, line in enumerate(src, 1):
        if lineno % 5_000_000 == 0:
            print(f"  {lineno:,} lines  |  {written:,} written  |  {skipped:,} skipped")

        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            skipped += 1
            continue
        try:
            playtime = json.loads(parts[3]).get("playtime", -1)
            if playtime is None or playtime < MIN_PLAYTIME:
                skipped += 1
                continue

            ts   = parts[2].strip()
            inter = json.loads(parts[4])
            uid  = inter["subjects"][0]["id"]
            iid  = inter["objects"][0]["id"]

            dst.write(f"{uid},{iid},{ts}\n")
            written += 1
        except (json.JSONDecodeError, KeyError, IndexError, ValueError):
            skipped += 1

print(f"\nDone. {written:,} rows written to {OUT_FILE}  ({skipped:,} skipped)")
