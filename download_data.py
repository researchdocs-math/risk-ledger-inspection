"""Download and checksum the original manufacturing datasets."""
from pathlib import Path
import argparse, hashlib, json, urllib.request, zipfile
ROOT = Path(__file__).resolve().parent

def fetch(name, spec):
    archive = ROOT / "data" / "downloads" / (name + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with urllib.request.urlopen(spec["url"], timeout=120) as src, partial.open("wb") as dst:
            while True:
                block = src.read(1024 * 1024)
                if not block: break
                dst.write(block)
        partial.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""): digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {archive}; inspect the downloaded file before retrying")
    destination = ROOT / "data" / name
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe archive member")
        z.extractall(destination)
    print(name, "verified SHA256", digest.hexdigest(), "extracted to", destination)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["secom", "ksdd2", "all"], nargs="?", default="all")
    args = ap.parse_args()
    manifest = json.loads((ROOT / "data" / "downloads.json").read_text())
    for name in ([args.dataset] if args.dataset != "all" else ["secom", "ksdd2"]):
        fetch(name, manifest[name])
