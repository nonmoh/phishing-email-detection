from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVES = ROOT / "data" / "source_archives"
RAW.mkdir(parents=True, exist_ok=True)
ARCHIVES.mkdir(parents=True, exist_ok=True)

KAGGLE_LARGE = "naserabdullahalam/phishing-email-dataset"
KAGGLE_GRECO = "francescogreco97/human-llm-generated-phishing-legitimate-emails"
ZENODO_RECORD = "20250116"
ZENODO_FILENAME = "cross-model-phishing.zip"
ZENODO_MD5 = "7bd6097768e0f44e433a153a77b374f5"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        if p.is_file():
            target = dst / p.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def download_kaggle(slug: str, dest: Path, force: bool = False) -> dict:
    data_exts = {".csv", ".parquet", ".json", ".jsonl"}
    has_data = dest.exists() and any(
        p.is_file() and p.suffix.lower() in data_exts for p in dest.rglob("*")
    )
    if has_data and not force:
        print(f"[SKIP] Kaggle data already present: {dest}")
        return {"source": slug, "status": "already_present", "path": str(dest)}

    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError("kagglehub is not installed. Run the bootstrap installer first.") from exc

    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[DOWNLOAD] Kaggle: {slug}")
    try:
        cache_path = Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle download failed for {slug}. Public datasets normally download without a token, "
            "but if Kaggle requests authentication, sign in once or place kaggle.json in "
            "%USERPROFILE%\\.kaggle\\kaggle.json, then rerun. Original error: {exc!r}"
        ) from exc

    print("[COPY]", cache_path, "->", dest)
    copy_tree(cache_path, dest)
    return {
        "source": slug,
        "status": "downloaded",
        "cache_path": str(cache_path),
        "path": str(dest),
        "files": sum(1 for p in dest.rglob("*") if p.is_file()),
    }


def download_zenodo(dest_zip: Path, extract_to: Path, force: bool = False) -> dict:
    if extract_to.exists() and (extract_to / "data" / "corpus_features.csv").exists() and not force:
        print(f"[SKIP] Zenodo cross-model data already present: {extract_to}")
        return {"source": f"zenodo:{ZENODO_RECORD}", "status": "already_present", "path": str(extract_to)}

    import requests

    if force:
        if dest_zip.exists():
            dest_zip.unlink()
        if extract_to.exists():
            shutil.rmtree(extract_to)

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"[LOOKUP] Zenodo record {ZENODO_RECORD}")
    meta_url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    r = requests.get(meta_url, timeout=60)
    r.raise_for_status()
    record = r.json()

    chosen = None
    for item in record.get("files", []):
        key = item.get("key") or item.get("filename")
        if key == ZENODO_FILENAME:
            chosen = item
            break
    if not chosen:
        raise RuntimeError(f"{ZENODO_FILENAME} was not found in Zenodo record {ZENODO_RECORD}")

    url = (chosen.get("links") or {}).get("self") or (chosen.get("links") or {}).get("content")
    if not url:
        raise RuntimeError("Zenodo file download URL is missing from API metadata")

    print(f"[DOWNLOAD] Zenodo: {ZENODO_FILENAME}")
    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        got = 0
        with dest_zip.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"  {got/1024/1024:.1f}/{total/1024/1024:.1f} MB", end="\r")
    print()

    actual = md5(dest_zip)
    if actual.lower() != ZENODO_MD5.lower():
        raise RuntimeError(f"Zenodo archive checksum mismatch: expected {ZENODO_MD5}, got {actual}")

    print("[EXTRACT]", dest_zip, "->", extract_to)
    with zipfile.ZipFile(dest_zip, "r") as z:
        z.extractall(extract_to)

    return {
        "source": f"zenodo:{ZENODO_RECORD}",
        "status": "downloaded",
        "archive": str(dest_zip),
        "path": str(extract_to),
        "md5": actual,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download all public data used by the project.")
    ap.add_argument("--force", action="store_true", help="Redownload even if data already exists.")
    args = ap.parse_args()

    manifest = []
    manifest.append(download_kaggle(KAGGLE_LARGE, RAW / "kaggle_large", args.force))
    manifest.append(download_kaggle(KAGGLE_GRECO, RAW / "four_class_web", args.force))
    manifest.append(
        download_zenodo(
            ARCHIVES / ZENODO_FILENAME,
            RAW / "cross_model_web",
            args.force,
        )
    )

    out = ROOT / "data" / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAll project data sources are ready.")
    print("Manifest:", out)


if __name__ == "__main__":
    main()
