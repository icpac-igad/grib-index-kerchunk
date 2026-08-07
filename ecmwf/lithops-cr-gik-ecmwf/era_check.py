#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["google-cloud-storage"]
# ///
"""
era_check.py -- era/cycle reference and pre-flight checker for the ECMWF GIK catalog
====================================================================================

Companion to run_lithops_ecmwf.py (which is NOT modified by this file). It answers
the questions that are easy to get wrong and expensive to get wrong silently:

    which era does this date+cycle belong to?
    do my two era switches agree?
    where exactly does the output go, and under which identity?
    is this date range actually complete, member by member?

    uv run era_check.py guide                       # the A-Z operator guide
    uv run era_check.py paths                       # identity + bucket + layout
    uv run era_check.py era   --date 20240228 --run 06
    uv run era_check.py preflight --run 06          # do switch 1 and switch 2 agree?
    uv run era_check.py verify --run 06 --start 20230118 --end 20260702


================================================================================
                            A-Z OPERATOR GUIDE
================================================================================

A. IDENTITY / CREDENTIALS
   Key (relative to this file, mode 600):
       ./service_account/ecmwf-lithops-deployer-key.json
   Full path on the deployer host:
       /data/08-2023/working_notes_jupyter/ignore_nka_gitrepos/cno-e4drr/
           devops/lithops_cr_ecmwf_gik/service_account/ecmwf-lithops-deployer-key.json
   Identity : ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com
   Project  : e4drr-crafd        Region: europe-west3 (nearest to ECMWF S3)
   Wired in via lithops_config.yaml:29 `credentials_path:`.
   ONE key covers every era and cycle -- era is a property of the image and the
   env, never of the identity.

       gcloud auth activate-service-account \
           --key-file=service_account/ecmwf-lithops-deployer-key.json \
           --project=e4drr-crafd

B. WHERE OUTPUT GOES
   Bucket : gs://gik-ecmwf-aws-tf              (GCS_BUCKET)
   Prefix : v20260623_run_par_ecmwf            (GCS_PARQUET_PREFIX)
   Layout : {prefix}/{YYYY}/{MM}/{YYYYMMDD}/{HH}z/{YYYYMMDD}{HH}z-{member}.parquet
   51 members/date: control + ens_01..ens_50.
   Example:
       gs://gik-ecmwf-aws-tf/v20260623_run_par_ecmwf/2026/06/20260621/06z/
           2026062106z-control.parquet

   !! gs://gik-ecmwf-aws-tf/run_par_ecmwf/ (no v2 prefix) is the LEGACY catalog.
      Superseded -- never write there. Always export GCS_PARQUET_PREFIX; older
      checkouts of run_lithops_ecmwf.py default to the legacy name.

C. THE FOUR CYCLES
   --run picks the cycle. 00z/12z are full length; 06z/18z are SHORT
   (S3-verified: +150h and beyond are 404 for those cycles).

       cycle  reach    steps
       00z    0-360h     85
       06z    0-144h     49
       12z    0-360h     85
       18z    0-144h     49

   run_lithops_ecmwf.py:forecast_hours_for_run() derives the hour list from
   --run, so this is automatic. Requesting the 36 absent steps would cost
   36 x 51 = 1836 futile ranged GETs per date.

D. THE THREE ERAS -- and the TWO SWITCHES
   Era selection is TWO independent switches. NOTHING links them, and a
   mismatch is SILENT -- it writes 0 files (or 50), never an error.

     switch 1 -- exported ENV        : which S3 bytes are READ
     switch 2 -- lithops_config.yaml : which image/template DECODES them
                 line 32, `runtime:` tag

   era   image   ECMWF_RESOLUTION  ECMWF_CONTROL_STREAM  ECMWF_REFERENCE_DATE
   0p4   :0p4    0p4               enfo                  20230601
   49r1  :49r1   0p25              enfo                  20250515
   50r1  :50r1   0p25              oper                  20260513

   S3 source (ecmwf_index_url()):
       0.4 deg : s3://ecmwf-forecasts/{date}/{HH}z/0p4-beta/{stream}/...
       0.25deg : s3://ecmwf-forecasts/{date}/{HH}z/ifs/0p25/{stream}/...
       suffix  : -enfo-ef (perturbed + bundled control) | -oper-fc (50r1 control)

   Templates (HF E4DRR/grib-index-kerchunk-templates):
       0p4  gik-fmrc-v2ecmwf_fmrc-0p4-beta.tar.gz       ( 9 levels)
       49r1 gik-fmrc-v2ecmwf_fmrc-49r1-perlevel.tar.gz  (13 levels)
       50r1 gik-fmrc-v2ecmwf_fmrc-50r1.tar.gz           (14 levels)
   The image's BAKED template wins: ensure_template() returns
   ECMWF_TEMPLATE_PATH before TEMPLATE_URL is read, so an exported
   TEMPLATE_URL is dead code on Cloud Run and cannot override the era.

       grep -E 'runtime: gcr' lithops_config.yaml     # check switch 2, always
   Flip it only BETWEEN waves, never during one.

E. ERA WINDOWS -- cutovers are CYCLE-granular
   era    00z                       06z / 12z / 18z
   0p4    20230118 - 20240228       20230118 - 20240227
   49r1   20240229 - 20260512       20240228 - 20260511
   50r1   20260513 - ongoing        20260512 - ongoing

   The switch happens partway through the day, so every cycle after 00z moves
   a day earlier:
       20240228  0p4-beta 404 but ifs/0p25 200   -> already 49r1
       20260512  control absent from enfo, in oper -> already 50r1
   Wrong side => 0 files (20240228) or 50/51 (20260512).
       bash fix_cycle_boundary_dates.sh <06|12|18>

F. DATES THAT CAN NEVER BE FILLED
   ECMWF never published these at 0.4 deg (404 on S3), also cycle-dependent:
       00z, 06z : 20230427-20230502  (6)  -> corpus tops out at 1256
       12z, 18z : 20230426-20230502  (7)  -> corpus tops out at 1255
   A wave covering them reports PARTIAL. Correct, not a failure.

G. REQUIRED ENVIRONMENT
       export UV_PYTHON=3.12          # host Python must match the runtime
       export GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf
       export AWS_NO_SIGN_REQUEST=YES
       # + the three ECMWF_* vars for the era (section D)

H. RUN IT
   Single date, current era (50r1):
       export UV_PYTHON=3.12 GCS_PARQUET_PREFIX=v20260623_run_par_ecmwf \
              AWS_NO_SIGN_REQUEST=YES
       export ECMWF_REFERENCE_DATE=20260513 ECMWF_RESOLUTION=0p25 \
              ECMWF_CONTROL_STREAM=oper
       uv run era_check.py preflight --run 00        # <- switches agree?
       timeout 1500 uv run run_lithops_ecmwf.py \
           --start-date $D --end-date $D --run 00 --max-workers 4 --yes

   Whole cycle, all three eras (handles tag flips and era windows):
       bash run_cycle_waves.sh <06|12|18>

I. VERIFY -- never by exit code
   The driver finishes the work, prints its full output, then HANGS at
   interpreter exit on a lingering lithops thread. Under `timeout` that is
   exit 124 == SUCCESS. Truth is the GCS member set:
       uv run era_check.py verify --run 06 --start $D --end $D

J. VALIDATE THE CONTENTS
       bash run_cycle_herbie_gate.sh <00|06|12|18>
   Pass: 49r1/50r1 r >= 0.9999 ; 0p4 r >= 0.9997 (grid-reindex residual).
   Check a LATE step as well as T+0 -- a T+0-only pass cannot detect a wrong
   time axis.

K. KNOWN GOTCHAS
   1. exit 124 == success. Verify via GCS, never the exit code.
   2. Log silence != dead driver -- a wave can run 40 min without writing.
      Check `ps`, don't just tail the log.
   3. Wrong active gcloud account -> 403 on registry / cloudbuild bucket.
   4. `uv run --with lithops` misses GCP deps: add httplib2, google-auth,
      google-api-python-client, google-cloud-storage.
   5. Network loss: Cloud Run workers finish regardless (they write GCS
      themselves), but the local monitor goes blind and mislabels good waves
      PARTIAL(0/n). run_cycle_waves.sh gates on require_net().
   6. ONE activation per DATE, not per member -- all 51 members are built
      inside a single worker.

See also:
    ECMWF_00Z_BACKFILL_SUMMARY.md    how the 00z corpus was built (secs 4, 11)
    cycles/README.md                 06z/12z/18z plan; sec 1b = cutover finding
    cycles/{06z,12z,18z}/RESULTS.md  per-cycle outcomes
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- constants --
SA_KEY = HERE / "service_account" / "ecmwf-lithops-deployer-key.json"
SA_EMAIL = "ecmwf-lithops-deployer@e4drr-crafd.iam.gserviceaccount.com"
PROJECT = "e4drr-crafd"
REGION = "europe-west3"
BUCKET = "gik-ecmwf-aws-tf"
PREFIX = os.environ.get("GCS_PARQUET_PREFIX", "v20260623_run_par_ecmwf")
LEGACY_PREFIX = "run_par_ecmwf"
MEMBERS = ["control"] + [f"ens_{i:02d}" for i in range(1, 51)]

CYCLES = {"00": (360, 85), "06": (144, 49), "12": (360, 85), "18": (144, 49)}

ERA_ENV = {
    "0p4":  dict(ECMWF_REFERENCE_DATE="20230601", ECMWF_RESOLUTION="0p4",
                 ECMWF_CONTROL_STREAM="enfo"),
    "49r1": dict(ECMWF_REFERENCE_DATE="20250515", ECMWF_RESOLUTION="0p25",
                 ECMWF_CONTROL_STREAM="enfo"),
    "50r1": dict(ECMWF_REFERENCE_DATE="20260513", ECMWF_RESOLUTION="0p25",
                 ECMWF_CONTROL_STREAM="oper"),
}

# Cutovers are CYCLE-granular: every cycle after 00z switches a day earlier.
ERA_WINDOWS = {
    "00": [("0p4", "20230118", "20240228"), ("49r1", "20240229", "20260512"),
           ("50r1", "20260513", "20991231")],
    "06": [("0p4", "20230118", "20240227"), ("49r1", "20240228", "20260511"),
           ("50r1", "20260512", "20991231")],
}
ERA_WINDOWS["12"] = ERA_WINDOWS["18"] = ERA_WINDOWS["06"]

# Never published by ECMWF at 0.4 deg -- 404 on S3, not fixable.
ABSENT = {
    "00": {"20230427", "20230428", "20230429", "20230430", "20230501", "20230502"},
    "06": {"20230427", "20230428", "20230429", "20230430", "20230501", "20230502"},
    "12": {"20230426", "20230427", "20230428", "20230429", "20230430", "20230501", "20230502"},
}
ABSENT["18"] = ABSENT["12"]


def era_for(date: str, run: str) -> str:
    for era, a, b in ERA_WINDOWS[run]:
        if a <= date <= b:
            return era
    raise SystemExit(f"{date} is before the catalog start (20230118)")


def daterange(a: str, b: str):
    d = dt.date(int(a[:4]), int(a[4:6]), int(a[6:]))
    e = dt.date(int(b[:4]), int(b[4:6]), int(b[6:]))
    while d <= e:
        yield f"{d:%Y%m%d}"
        d += dt.timedelta(days=1)


def config_tag(cfg: Path) -> str | None:
    """switch 2 -- the runtime tag currently in lithops_config.yaml."""
    try:
        m = re.search(r"ecmwf-lithops-runtime:([A-Za-z0-9._-]+)", cfg.read_text())
        return m.group(1) if m else None
    except OSError:
        return None


# ------------------------------------------------------------------ commands --
def cmd_guide(_):
    print(__doc__)


def cmd_paths(_):
    print(f"""
identity
  service account : {SA_EMAIL}
  key file        : {SA_KEY}
  key present     : {'yes' if SA_KEY.exists() else 'NO -- runs will fail'}
  project         : {PROJECT}
  region          : {REGION}

storage
  bucket          : gs://{BUCKET}
  prefix          : {PREFIX}
  layout          : gs://{BUCKET}/{PREFIX}/{{YYYY}}/{{MM}}/{{YYYYMMDD}}/{{HH}}z/
                        {{YYYYMMDD}}{{HH}}z-{{member}}.parquet
  members/date    : 51  (control + ens_01..ens_50)
  LEGACY (do not use): gs://{BUCKET}/{LEGACY_PREFIX}/

activate
  gcloud auth activate-service-account \\
      --key-file={SA_KEY} --project={PROJECT}
""".rstrip())


def cmd_era(args):
    era = era_for(args.date, args.run)
    env = ERA_ENV[era]
    reach, steps = CYCLES[args.run]
    y, m = args.date[:4], args.date[4:6]
    print(f"""
{args.date}  {args.run}z  ->  era {era}

  runtime tag        : gcr.io/e4drr-crafd/ecmwf-lithops-runtime:{era}
  ECMWF_RESOLUTION   : {env['ECMWF_RESOLUTION']}
  ECMWF_CONTROL_STREAM: {env['ECMWF_CONTROL_STREAM']}
  ECMWF_REFERENCE_DATE: {env['ECMWF_REFERENCE_DATE']}
  cycle reach        : 0-{reach}h  ({steps} steps)
  output             : gs://{BUCKET}/{PREFIX}/{y}/{m}/{args.date}/{args.run}z/
  S3 source          : s3://ecmwf-forecasts/{args.date}/{args.run}z/\
{'0p4-beta' if env['ECMWF_RESOLUTION'] == '0p4' else 'ifs/0p25'}/{env['ECMWF_CONTROL_STREAM']}/

  export ECMWF_REFERENCE_DATE={env['ECMWF_REFERENCE_DATE']} \
ECMWF_RESOLUTION={env['ECMWF_RESOLUTION']} \
ECMWF_CONTROL_STREAM={env['ECMWF_CONTROL_STREAM']}
""".rstrip())
    if args.date in ABSENT[args.run]:
        print(f"\n  !! {args.date} is NOT published at {args.run}z (404 on S3). Nothing to fetch.")


def cmd_preflight(args):
    """Do switch 1 (env) and switch 2 (config tag) agree?"""
    cfg = HERE / "lithops_config.yaml"
    tag = config_tag(cfg)
    env = {k: os.environ.get(k) for k in
           ("ECMWF_RESOLUTION", "ECMWF_CONTROL_STREAM", "ECMWF_REFERENCE_DATE")}
    print(f"  switch 2  lithops_config.yaml runtime tag : {tag or 'NOT FOUND'}")
    print(f"  switch 1  ECMWF_RESOLUTION               : {env['ECMWF_RESOLUTION']}")
    print(f"            ECMWF_CONTROL_STREAM           : {env['ECMWF_CONTROL_STREAM']}")
    print(f"            ECMWF_REFERENCE_DATE           : {env['ECMWF_REFERENCE_DATE']}")
    if not SA_KEY.exists():
        print(f"\n  FAIL: service-account key missing at {SA_KEY}")
        return 2
    if tag not in ERA_ENV:
        print(f"\n  FAIL: runtime tag '{tag}' is not one of {list(ERA_ENV)}")
        return 2
    want = ERA_ENV[tag]
    bad = [k for k, v in want.items() if env.get(k) != v]
    if bad:
        print(f"\n  FAIL: env does not match the deployed :{tag} image -> {bad}")
        print("        A mismatch is SILENT -- it writes 0 files, not an error.")
        print("        export " + " ".join(f"{k}={v}" for k, v in want.items()))
        return 2
    if args.date:
        expect = era_for(args.date, args.run)
        if expect != tag:
            print(f"\n  FAIL: {args.date} {args.run}z belongs to era {expect}, "
                  f"but the config is on :{tag}")
            return 2
    print(f"\n  OK: both switches agree on era {tag}")
    return 0


def cmd_verify(args):
    """Member-set completeness over a date range -- catches a duplicate masking a gap."""
    try:
        from google.cloud import storage
    except ImportError:
        sys.exit("google-cloud-storage not available; run via `uv run era_check.py`")
    if SA_KEY.exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(SA_KEY))
    client = storage.Client(project=PROJECT)
    bkt = client.bucket(BUCKET)

    want = set(MEMBERS)
    found: dict[str, set] = {}
    months = sorted({d[:6] for d in daterange(args.start, args.end)})
    for i, ym in enumerate(months, 1):
        pre = f"{PREFIX}/{ym[:4]}/{ym[4:]}/"
        for b in client.list_blobs(bkt, prefix=pre):
            parts = b.name.split("/")
            if len(parts) < 6 or not parts[-1].endswith(".parquet"):
                continue
            if parts[-2] != f"{args.run}z":
                continue
            found.setdefault(parts[-3], set()).add(
                re.sub(r"^\d{10}z-", "", parts[-1]).replace(".parquet", ""))
        print(f"\r  scanned {i}/{len(months)} months", end="", file=sys.stderr)
    print(file=sys.stderr)

    absent = ABSENT[args.run]
    target = [d for d in daterange(args.start, args.end) if d not in absent]
    full = [d for d in target if found.get(d) == want]
    partial = {d: sorted(want - found[d]) for d in target if d in found and found[d] != want}
    missing = [d for d in target if d not in found]
    skipped = [d for d in daterange(args.start, args.end) if d in absent]

    print(f"\n{args.run}z  {args.start} -> {args.end}")
    print(f"  target dates (excl. {len(skipped)} never-published) : {len(target)}")
    print(f"  complete 51/51                                 : {len(full)}")
    print(f"  partial                                        : {len(partial)}")
    print(f"  missing                                        : {len(missing)}")
    per = {}
    for d in full:
        per[era_for(d, args.run)] = per.get(era_for(d, args.run), 0) + 1
    if per:
        print("  by era: " + "  ".join(f"{e}={n} ({n*51} parquets)" for e, n in sorted(per.items())))
    for d, miss in list(partial.items())[:20]:
        print(f"    PARTIAL {d}: missing {miss}")
    if missing:
        print(f"    MISSING: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        for d in missing[:5]:
            print(f"      hint: {d} {args.run}z belongs to era {era_for(d, args.run)} "
                  f"-- check it was not run under the neighbouring era (guide, §E)")
    return 0 if not partial and not missing else 1


def main():
    ap = argparse.ArgumentParser(description="ECMWF GIK era/cycle reference and checker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("guide", help="print the A-Z operator guide").set_defaults(fn=cmd_guide)
    sub.add_parser("paths", help="identity, bucket, layout").set_defaults(fn=cmd_paths)

    p = sub.add_parser("era", help="which era does a date+cycle belong to")
    p.add_argument("--date", required=True)
    p.add_argument("--run", default="00", choices=list(CYCLES))
    p.set_defaults(fn=cmd_era)

    p = sub.add_parser("preflight", help="check the two era switches agree")
    p.add_argument("--run", default="00", choices=list(CYCLES))
    p.add_argument("--date", help="also check this date belongs to the configured era")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("verify", help="member-set completeness over a range")
    p.add_argument("--run", default="00", choices=list(CYCLES))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
