#!/usr/bin/env python
"""Is a store's data actually WHERE its longitude coordinate says it is?

Every other check in this repo is self-referential: it compares the store to
itself, or to expectations derived from the same assumption that built it. A
uniform longitude shift is invisible to all of them, because every variable is
shifted by the same amount and nothing inside the store disagrees.

This one uses an external reference the store cannot influence: **the Sun**.

    At 00 UTC the sun is overhead at 180 deg E.
    Local solar time  LST = (UTC + lon/15) mod 24.

Skin temperature over land has a large diurnal cycle -- peak near 13-14 LST,
minimum near 05-06 LST. So the SIGN of the step-0 diurnal anomaly over land is
fixed by longitude and nothing else:

    land at   0- 90 E  ->  LST 00-06  ->  pre-dawn   -> anomaly NEGATIVE
    land at 180-270 E  ->  LST 12-18  ->  afternoon  -> anomaly POSITIVE

Land is identified from the diurnal amplitude itself (land swings many K, ocean
under 1 K), so this needs no land-sea mask -- which matters, because the older
30-channel stores do not carry one.

    $P check_geolocation.py --source --era 49r1 --date 2026-03-01
    $P check_geolocation.py --prefix ea-swio/v1-49r1-mar-may2026 --env .env
    $P check_geolocation.py --prefix ea-cgan/v3-june2026 --env .env
"""
import argparse
import sys

import numpy as np

STEPS = [0, 3, 6, 9, 12, 15, 18, 21]      # one full diurnal cycle
AMP_MIN = 8.0                              # K; land swings well past this


def solar_verdict(lon_mean):
    """Expected sign of the 00 UTC skin-temperature anomaly at this longitude."""
    lst = (0.0 + lon_mean / 15.0) % 24.0
    if 10.0 <= lst <= 19.0:
        return lst, "POSITIVE", "afternoon"
    if lst >= 21.0 or lst <= 8.0:
        return lst, "NEGATIVE", "night / pre-dawn"
    return lst, "AMBIGUOUS", "dawn or dusk transition"


def report(name, lon, skt, amp_min=AMP_MIN):
    """skt: (nstep, ny, nx) at STEPS. lon: matching longitude coordinate."""
    amp = skt.max(axis=0) - skt.min(axis=0)
    land = amp > amp_min
    n_land = int(land.sum())
    print(f"\n=== {name} ===")
    print(f"  longitude coordinate : {lon.min():.2f} .. {lon.max():.2f}")
    print(f"  land-like cells      : {n_land:,} of {land.size:,} "
          f"({100*land.mean():.1f}%)  [diurnal amplitude > {amp_min} K]")
    print(f"  diurnal amplitude    : median {np.median(amp):.2f} K, "
          f"p99 {np.percentile(amp, 99):.2f} K")
    if n_land < 50:
        print(f"  TOO FEW cells above {amp_min} K -- rerun with a lower "
              f"--amp-min. NOTE: a box genuinely over East Africa would have "
              f"~35-45% of cells swinging >8 K; near zero is itself evidence "
              f"the box is over open ocean.")
        return None

    anom = skt[0] - skt.mean(axis=0)
    got = float(anom[land].mean())

    # Predict from the longitudes the store CLAIMS, weighted to land cells.
    # Longitude is circular, so a plain mean is only meaningful for a REGIONAL
    # box. Averaging it over a global field is nonsense -- it drifts to
    # whatever the land happens to be distributed around and lands in the
    # dawn/dusk band. Global fields are judged in main() on the position of
    # the zonal diurnal maximum instead, which is a real physical landmark.
    span = float(lon.max() - lon.min())
    if span > 180.0:
        print(f"  global field (span {span:.0f} deg) -- a mean longitude is "
              f"not meaningful here; see the zonal-maximum test below")
        return None
    lon2d = np.broadcast_to(lon[None, :], land.shape)
    lon_land = float(np.mean(lon2d[land] % 360.0))
    lst, expect, phase = solar_verdict(lon_land)

    print(f"  mean land longitude  : {lon_land:.1f} E (as labelled)")
    print(f"  -> local solar time at 00 UTC : {lst:04.1f} h  ({phase})")
    print(f"  -> anomaly should be : {expect}")
    print(f"  measured step-0 land anomaly  : {got:+.2f} K")

    if expect == "AMBIGUOUS":
        print("  VERDICT: inconclusive at this longitude")
        return None

    # A sign is only meaningful if the signal clears the noise. Over open
    # ocean the diurnal swing is a few tenths of a K, and the mean anomaly
    # lands within rounding of zero -- which once produced OPPOSITE verdicts
    # for two stores built by the same code from the same source. Demand the
    # anomaly be a real fraction of the observed swing before ruling.
    floor = max(0.5, 0.25 * float(np.median(amp[land])))
    if abs(got) < floor:
        print(f"  |anomaly| {abs(got):.2f} K < {floor:.2f} K noise floor "
              f"(0.25 x median swing, min 0.5 K)")
        print("  VERDICT: INCONCLUSIVE -- too little diurnal signal here. "
              "Raise --amp-min to select genuine land, or test a box that "
              "contains some.")
        return None
    ok = (got > 0) if expect == "POSITIVE" else (got < 0)
    print(f"  VERDICT: {'CONSISTENT — coordinate is trustworthy' if ok else
                        'CONTRADICTION — data is not where the label says'}")
    if not ok:
        shifted, exp2, ph2 = solar_verdict((lon_land + 180.0) % 360.0)
        print(f"           at lon+180 the LST is {shifted:04.1f} h ({ph2}), "
              f"which predicts {exp2} — matching the measurement")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="store_true",
                   help="test the published virtual store (global)")
    p.add_argument("--era", default="49r1")
    p.add_argument("--date", default="2026-03-01")
    p.add_argument("--prefix", default=None, help="a realized store instead")
    p.add_argument("--env", default=".env")
    p.add_argument("--member", type=int, default=1)
    p.add_argument("--amp-min", type=float, default=AMP_MIN,
                   help="K; cells whose diurnal swing exceeds this are used. "
                        "8 isolates land; ~0.3 also admits ocean, whose "
                        "diurnal cycle is weak but still solar-phased")
    args = p.parse_args()

    if args.source:
        import frisky_daily_dag as dag
        ds = dag._open_era(args.era)
        lon = ds.longitude.values
        skt = np.stack([
            ds["skt"].sel(time=np.datetime64(args.date), number=args.member,
                          step=np.timedelta64(h, "h")).values for h in STEPS])
        ok = report(f"SOURCE {args.era}/00z  {args.date}", lon, skt, args.amp_min)

        # Global stores can say more: where IS the diurnal peak?
        amp = skt.max(axis=0) - skt.min(axis=0)
        land = amp > args.amp_min
        anom = np.where(land, skt[0] - skt.mean(axis=0), np.nan)
        prof = np.nanmean(anom, axis=0)
        j = int(np.nanargmax(prof))
        peak = float(lon[j]) % 360.0
        # Sun over 180 E at 00 UTC; skin temperature lags ~1-3 h -> 195-225 E.
        off = min(abs(peak - 210.0), 360.0 - abs(peak - 210.0))
        print(f"\n  zonal diurnal maximum at {lon[j]:.1f} (= {peak:.1f} E)")
        print(f"  expected near 210 E (sun over 180 E at 00 UTC, plus thermal "
              f"lag); offset {off:.0f} deg")
        ok = off <= 35.0
        print(f"  VERDICT: {'CONSISTENT — the longitude axis is trustworthy'
                            if ok else
                            'CONTRADICTION — diurnal peak in the wrong place'}")
        if not ok:
            print(f"           a 180 deg shift would put it at "
                  f"{(peak + 180) % 360:.1f} E")
    else:
        if not args.prefix:
            sys.exit("give --source or --prefix")
        from inspect_store import open_store
        _, ds = open_store(args.prefix, args.env)
        if "skt" not in ds.data_vars:
            sys.exit(f"{args.prefix} has no `skt` channel; cannot run this test")
        t0 = ds.time.values[0]
        have = set(int(x / np.timedelta64(1, "h")) for x in ds.step.values)
        steps = [h for h in STEPS if h in have]
        if len(steps) < 6:
            sys.exit(f"need >=6 of {STEPS}; store has {sorted(have)[:10]}")
        skt = np.stack([
            ds["skt"].sel(time=t0, number=args.member,
                          step=np.timedelta64(h, "h")).values for h in steps])
        ok = report(f"REALIZED {args.prefix}  "
                    f"{np.datetime_as_string(t0, 'D')}",
                    ds.longitude.values, skt, args.amp_min)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
