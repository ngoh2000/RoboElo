#!/usr/bin/env python3
"""Download public Hugging Face model repositories with curl, continuing partly downloaded files.

huggingface_hub 1.x deletes a partly downloaded file when an attempt fails. On this machine's slow link,
which also drops out for minutes at a time, files of several GB then never finish. Here every file is
fetched with `curl -C -` into <file>.part, an interrupted transfer continues where it stopped, and the
finished file is checked against the size and SHA-256 that the Hub lists before it is renamed.

  download_hf_resume.py <repo>=<folder under ~/roboelo/checkpoints>=<marker>[=<path prefix,path prefix,...>] ...

Repositories are downloaded one after another. state/model_<marker>.done is written when one is complete;
a repository whose marker exists is skipped. Only the standard library and curl are used.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.expanduser("~/roboelo")


def say(msg):
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def get_json(url):
    """The Hub's answer to an API call; waits out network outages."""
    while True:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r), r.headers.get("Link", "")
        except Exception as exc:
            say(f"cannot reach the Hub ({type(exc).__name__}: {exc}); trying again in 30 s")
            time.sleep(30)


def list_files(repo):
    files, url = [], f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    while url:
        page, link = get_json(url)
        files += [f for f in page if f["type"] == "file"]
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return files


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo, entry, folder):
    path = entry["path"]
    size = entry.get("lfs", {}).get("size", entry["size"])
    dest = os.path.join(folder, path)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(path)}"
    say(f"{repo}: {path} ({size / 1e6:.1f} MB)")
    while True:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            continue
        if have < size:
            # give up on a transfer that stays below 20 kB/s for a minute, then continue it
            rc = subprocess.call(["curl", "-L", "-sS", "--fail", "-C", "-", "--connect-timeout", "30",
                                  "--speed-limit", "20000", "--speed-time", "60", "-o", part, url])
            now = os.path.getsize(part) if os.path.exists(part) else 0
            if now < size:
                if rc == 33 and os.path.exists(part):  # the server refused to continue: start this file again
                    os.remove(part)
                say(f"{repo}: {path} interrupted at {now / 1e6:.0f} of {size / 1e6:.0f} MB (curl exit {rc}); continuing in 15 s")
                time.sleep(15)
            continue
        if "lfs" in entry and sha256_of(part) != entry["lfs"]["oid"]:
            say(f"{repo}: {path} has the wrong SHA-256; downloading it again")
            os.remove(part)
            continue
        os.replace(part, dest)
        return


def main():
    for spec in sys.argv[1:]:
        repo, folder, marker, *rest = spec.split("=")
        prefixes = [p for p in rest[0].split(",") if p] if rest else []
        done_file = os.path.join(ROOT, "state", f"model_{marker}.done")
        if os.path.exists(done_file):
            say(f"{repo}: already complete")
            continue
        folder = os.path.join(ROOT, "checkpoints", folder)
        files = [f for f in list_files(repo) if not prefixes or any(f["path"].startswith(p) for p in prefixes)]
        total = sum(f.get("lfs", {}).get("size", f["size"]) for f in files)
        say(f"{repo}: {len(files)} files, {total / 1e9:.2f} GB -> {folder}")
        for entry in sorted(files, key=lambda f: f.get("lfs", {}).get("size", f["size"])):
            fetch(repo, entry, folder)
        with open(done_file, "w") as f:
            f.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")
        say(f"{repo}: complete")
    say("ALL_DOWNLOADS_END")


if __name__ == "__main__":
    main()
