# source.coop credentials from a headless gateway

How to get a **12-hour** source.coop STS credential on a machine with no
browser, and the one step that does not work out of the box.

**No key, token or secret appears in this file, and none should be added to
it.** Everything below uses placeholders.

---

## 0. First: do you actually need credentials?

Probably not. **Reading a public source.coop repository is anonymous.**

```python
st = icechunk.s3_storage(bucket="e4drr-project", prefix="forecasts/...",
                         endpoint_url="https://data.source.coop",
                         region="us-east-1",
                         anonymous=True, from_env=False, force_path_style=True)
```

That is how `frisky_daily_dag.py` reads the published ECMWF store, and how
every verification of the mirrored realized stores was done. Credentials are
needed **only to write** — i.e. to run `mirror_ceph_to_source_coop.py`.

Worth stating because it was got wrong once: a verification script was written
against the STS token purely out of habit, and failed when the token expired,
for a read that never needed it.

## 1. Why the CLI

The STS credentials handed out by the web console last **~1 hour**. A 3.5 TB
mirror takes ~8 h, so an hourly refresh means babysitting. The CLI's `login`
takes `--duration`, and a 12 h credential covers a whole transfer unattended.

The 1 h limit was never a cap on *volume* — `mirror_ceph_to_source_coop.py`
resumes on size comparison, so a refresh costs a re-run, not a re-transfer. The
12 h credential buys **unattendedness**, not capability.

## 2. Install, with no toolchain

`cargo`, `rustc` and `brew` are all absent on the gateway, so build-from-source
and the package managers are out. Use the prebuilt release binary.

```bash
URL=$(curl -sS https://api.github.com/repos/source-cooperative/source-coop-cli/releases/latest \
      | grep -oE '"browser_download_url": "[^"]*x86_64-unknown-linux-gnu.tar.xz"' \
      | cut -d'"' -f4)
curl -sSL -o sc.tar.xz "$URL"
```

**`xz` is not installed either**, so `tar xf` fails with `xz: Cannot exec`.
Extract with Python, which has `lzma` in the standard library:

```python
import lzma, tarfile, io, os
tf = tarfile.open(fileobj=io.BytesIO(lzma.open("sc.tar.xz").read()))
for m in tf.getmembers():
    if m.isfile() and os.path.basename(m.name) == "source-coop":
        dst = os.path.expanduser("~/.local/bin/source-coop")
        open(dst, "wb").write(tf.extractfile(m).read())
        os.chmod(dst, 0o755)
```

`~/.local/bin` is already on PATH. Confirm with `source-coop --version`.

## 3. Login — and the part that does not work

```bash
source-coop login --duration 12h
```

`--duration` accepts `3600`, `90s`, `5m`, `12h`, `1d`. It lives on **`login`**,
not on `creds`; it is far enough down `--help` to be missed if you only read
the first screen.

The CLI opens a **random local port** and waits for an OAuth2 callback. On a
headless gateway that is the whole problem: the listener is on the *gateway's*
`127.0.0.1`, and your browser is on your laptop, so the redirect lands on

```
http://127.0.0.1:39427/callback?code=...&state=...
```

which your browser cannot reach. The flow hangs and eventually times out.

### The fix: deliver the callback locally

The gateway *can* reach that address — it is the one listening. So:

1. Run `source-coop login --duration 12h` and open the URL it prints.
2. Complete the sign-in. Your browser lands on the unreachable
   `127.0.0.1:<port>/callback?...` page. **Copy that whole URL.**
3. On the gateway, while the login process is still waiting:

```bash
curl "http://127.0.0.1:<port>/callback?code=<CODE>&state=<STATE>"
```

It returns `Authentication Successful` and `login` exits normally.

Two conditions: the `login` process must **still be running** (check with
`pgrep -af source-coop`), and OAuth codes are **single-use and short-lived** —
if it has been minutes, start again. Treat a pasted callback URL as spent.

## 4. Store the credentials — including the two the CLI omits

```bash
source-coop creds --format env      # AWS_ACCESS_KEY_ID / SECRET / SESSION_TOKEN
```

**It emits only those three.** `AWS_ENDPOINT_URL` and `AWS_DEFAULT_REGION` are
*not* included and the mirror needs both. Writing `creds --format env` straight
over an existing `.env` will silently drop them — that happened once and the
next run failed on a missing endpoint.

The `.env` used here holds Ceph and source.coop credentials side by side:

```
AK=<ewc ceph access key>
SK=<ewc ceph secret key>
export AWS_ACCESS_KEY_ID=<from source-coop creds>
export AWS_SECRET_ACCESS_KEY=<from source-coop creds>
export AWS_SESSION_TOKEN=<from source-coop creds>
export AWS_ENDPOINT_URL=https://data.source.coop
export AWS_DEFAULT_REGION=us-east-1
```

`chmod 600` it — and check, don't assume: this file was found at **644** on
2026-08-26, world-readable on a shared gateway. It is gitignored by `**/.env` in
`medium-range-forecast/.gitignore`, along with `*.json` and `*credentials*` —
that rule is broad on purpose, because a key named
`coiled-data-e4drr_202505.json` once slipped past a narrower one.

The merge that keeps the two omitted variables:

```bash
umask 077
source-coop creds --format env > /tmp/creds.env
{ grep -vE '^(export )?(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)=' .env
  sed -E 's/^(export )?/export /' /tmp/creds.env ; } > .env.new
mv .env.new .env && chmod 600 .env && shred -u /tmp/creds.env
cut -d= -f1 .env | sed 's/^export //'      # expect all FIVE names back
```

> **A backup leaves the ignore rule behind.** Taking a `.env.bak-<stamp>` before
> that merge produced a file `**/.env` does **not** match — it appeared untracked
> in `git status`, one `git add -A` from a commit. Same lesson as the coiled key,
> one suffix down. `.gitignore` now also carries `.env.*` / `**/.env.*`, but the
> better habit is to keep no backup at all: its credentials are expired by
> definition and the only lines worth saving are the endpoint and region, which
> the merge preserves anyway. Verify with `git check-ignore -v <file>`, not by eye.

Then `source` it before any mirror command.

## 5. Check what you were actually granted

`--duration 12h` is a **request**. STS caps a session at the role's
`MaxSessionDuration`, so a role provisioned at 1 h will silently return 3600 s.
Verify rather than assume:

```bash
source-coop creds --format credential-process \
  | python -c "import sys,json,datetime; d=json.load(sys.stdin); \
      e=datetime.datetime.fromisoformat(d['Expiration'].replace('Z','+00:00')); \
      print(e, (e-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/3600, 'h')"
```

Measured 2026-08-19: **11.99 h granted**, so the 12 h request is honoured here.

## 6. When it expires mid-transfer

Nothing is lost. The mirror skips objects already present at the same size, so:

```bash
source-coop login --duration 12h        # + the callback trick above
source <the .env>
# re-run the same mirror command -- it resumes
```

`mirror_all_to_source_coop.sh` checks remaining credential life against the
measured 426 GB/h before starting each store, and refuses one it cannot
finish, rather than dying half-way unattended.

## 7. What not to do

- **Do not paste a key, secret, session token or callback code into a
  document, commit message, or log.** Scan before committing:
  `grep -c "$(grep '^AK=' .env | cut -d= -f2)" <file>` should be 0.
- **Do not write credentials to the worker VMs.** The mirror passes them as
  function arguments and subprocess environment; only the shard list touches
  VM disk, and it is deleted in a `finally`.
- **Do not reach for credentials to read.** See §0.

---

| | |
|---|---|
| `mirror_ceph_to_source_coop.py` | the only thing needing write credentials |
| `mirror_all_to_source_coop.sh` | checks credential life before each store |
| `MIRROR_RUN_ARCHITECTURE.md` | how credentials reach the VMs without touching disk |
