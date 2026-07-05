# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///
import argparse
import json, re, base64
import pandas as pd
from collections import Counter

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)


def decode_val(v):
    """Return python value from the `value` cell (may be bytes/str)."""
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode('utf-8', errors='replace')
        except Exception:
            return v
    return v


def try_json(s):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def classify(key, raw):
    s = decode_val(raw)
    parsed = try_json(s) if isinstance(s, str) else None
    # chunk ref: list len 3 starting with string url
    if isinstance(parsed, list) and len(parsed) == 3 and isinstance(parsed[0], str):
        return 'chunk_ref', parsed
    if isinstance(key, str) and key.endswith('.zarray'):
        return 'zarray', parsed
    if isinstance(key, str) and key.endswith('.zattrs'):
        return 'zattrs', parsed
    if isinstance(key, str) and key.endswith('.zgroup'):
        return 'zgroup', parsed
    # inline / base64 data
    if isinstance(s, str) and s.startswith('base64:'):
        return 'inline_base64', s
    if isinstance(raw, (bytes, bytearray)) and parsed is None:
        return 'inline_bytes', raw
    return 'other', s


def audit(path, full=True, label=""):
    out = []
    def p(*a):
        out.append(' '.join(str(x) for x in a))
    df = pd.read_parquet(path)
    p(f"===== {label} : {path} =====")
    p("shape:", df.shape)
    p("columns:", df.columns.tolist())
    p("dtypes:\n" + str(df.dtypes))
    if full:
        p("\n--- first 10 rows (untruncated) ---")
        p(df.head(10).to_string())

    kcol = 'key' if 'key' in df.columns else df.columns[0]
    vcol = 'value' if 'value' in df.columns else df.columns[1]

    cats = Counter()
    examples = {}
    d1 = Counter()
    d2 = Counter()
    chunk_urls = []
    chunk_offsets = []
    chunk_lengths = []
    zarray_vars = []

    for _, row in df.iterrows():
        key = row[kcol]
        raw = row[vcol]
        cat, parsed = classify(key, raw)
        cats[cat] += 1
        # key prefixes
        if isinstance(key, str):
            parts = key.split('/')
            d1[parts[0]] += 1
            if len(parts) >= 2:
                d2['/'.join(parts[:2])] += 1
            else:
                d2[parts[0]] += 1
        if cat == 'chunk_ref':
            chunk_urls.append(parsed[0])
            try:
                chunk_offsets.append(int(parsed[1]))
                chunk_lengths.append(int(parsed[2]))
            except Exception:
                pass
        if cat == 'zarray':
            zarray_vars.append(key)
        # collect examples
        examples.setdefault(cat, []).append((key, parsed if parsed is not None else decode_val(raw)))

    p("\n--- category counts ---")
    for c, n in cats.most_common():
        p(f"  {c}: {n}")

    p("\n--- distinct depth-1 key prefixes (row counts) ---")
    for k, n in sorted(d1.items(), key=lambda x: -x[1]):
        p(f"  {k}: {n}")

    p("\n--- distinct depth-2 key prefixes (row counts) ---")
    for k, n in sorted(d2.items(), key=lambda x: -x[1]):
        p(f"  {k}: {n}")

    if full:
        p("\n--- example rows per category ---")
        for cat in ['zarray', 'zattrs', 'zgroup', 'chunk_ref', 'inline_base64', 'inline_bytes', 'other']:
            if cat not in examples:
                continue
            p(f"\n=== {cat} examples (up to 3) ===")
            for key, val in examples[cat][:3]:
                if cat in ('inline_base64', 'inline_bytes', 'other'):
                    sval = str(val)
                    if len(sval) > 200:
                        sval = sval[:200] + '...[TRUNC]'
                    p(f"KEY: {key}")
                    p(f"VAL: {sval}")
                else:
                    p(f"KEY: {key}")
                    p(f"VAL: {json.dumps(val)}")

    # URL analysis
    if chunk_urls:
        p("\n--- chunk-ref URL analysis ---")
        patt = Counter(re.sub(r'\d', '#', u) for u in chunk_urls)
        p("distinct URL patterns (digits->#):", len(patt))
        for pat, n in patt.most_common(5):
            p(f"  {pat}  (x{n})")
        hosts = set()
        for u in chunk_urls:
            m = re.match(r'^[a-z0-9]+://([^/]+)/', u)
            if m:
                hosts.add(m.group(1))
            else:
                hosts.add(u.split('/')[0])
        p("distinct hostnames/buckets:", sorted(hosts))
        p("offset min/max:", min(chunk_offsets), max(chunk_offsets))
        p("length min/max:", min(chunk_lengths), max(chunk_lengths))
        p("num chunk refs:", len(chunk_urls))

    p("\n--- distinct zarr variables (.zarray count):", len(zarray_vars))

    if full and zarray_vars:
        # parse one data-variable zarray (multi-dim, not a coord)
        chosen = None
        for _, row in df.iterrows():
            key = row[kcol]
            if isinstance(key, str) and key.endswith('.zarray'):
                parsed = try_json(decode_val(row[vcol]))
                if parsed and len(parsed.get('shape', [])) >= 3:
                    chosen = (key, parsed)
                    break
        if chosen is None:
            key = zarray_vars[0]
            r = df[df[kcol] == key].iloc[0]
            chosen = (key, try_json(decode_val(r[vcol])))
        p("\n--- one data-variable .zarray verbatim ---")
        p("KEY:", chosen[0])
        z = chosen[1]
        for fld in ['shape', 'chunks', 'dtype', 'compressor', 'filters', 'order', 'fill_value', 'zarr_format']:
            p(f"  {fld}: {json.dumps(z.get(fld))}")
        p("FULL_ZARRAY_JSON:", json.dumps(z))
        # a zattrs verbatim
        for _, row in df.iterrows():
            key = row[kcol]
            if isinstance(key, str) and key.endswith('.zattrs'):
                parsed = try_json(decode_val(row[vcol]))
                if parsed:
                    p("\n--- one .zattrs verbatim ---")
                    p("KEY:", key)
                    p("FULL_ZATTRS_JSON:", json.dumps(parsed))
                    break

    return '\n'.join(out), dict(cats), df.shape, len(zarray_vars)


def main():
    ap = argparse.ArgumentParser(description="Audit an ECMWF GIK parquet reference file.")
    ap.add_argument('--par', required=True, help="Path to a control or member parquet (audited in full).")
    ap.add_argument('--compare', default=None,
                    help="Optional second parquet to diff category counts against.")
    args = ap.parse_args()

    par_txt, par_cats, par_shape, par_vars = audit(args.par, full=True, label="PAR")
    print(par_txt)

    if args.compare:
        cmp_txt, cmp_cats, cmp_shape, cmp_vars = audit(args.compare, full=False, label="COMPARE")
        diff = []
        diff.append("\n\n===== PAR vs COMPARE structural comparison =====")
        diff.append(f"par shape: {par_shape}   compare shape: {cmp_shape}")
        diff.append(f"par cats:  {par_cats}")
        diff.append(f"compare cats:      {cmp_cats}")
        diff.append(f"par #vars: {par_vars}   compare #vars: {cmp_vars}")
        print("\n\n" + cmp_txt + "\n" + '\n'.join(diff))


if __name__ == '__main__':
    main()
