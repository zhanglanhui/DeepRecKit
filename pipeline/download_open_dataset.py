import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve


DATASET_CONFIG = {
    "display_name": "Criteo Sample",
    "url": "https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip",
    "archive_name": "Criteo_x1.zip",
    "train_file": "train.csv",
    "valid_file": "valid.csv",
    "test_file": "test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the public Criteo dataset package and optionally build a local sample file."
    )
    parser.add_argument(
        "--output-dir",
        default="data/open_data/criteo_sample",
        help="Directory used to store the archive, extracted files, and generated sample CSV.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100000,
        help="Number of rows to keep in the generated sample CSV from train.csv.",
    )
    parser.add_argument(
        "--skip-sample",
        action="store_true",
        help="Only download and extract the dataset package without generating sample CSV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the archive and overwrite extracted outputs.",
    )
    return parser.parse_args()


def download_archive(dataset_dir: Path, force: bool) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / DATASET_CONFIG["archive_name"]
    if archive_path.exists() and not force:
        print(f"Reuse existing archive: {archive_path}")
        return archive_path

    print(f"Downloading {DATASET_CONFIG['display_name']} from {DATASET_CONFIG['url']}")
    urlretrieve(DATASET_CONFIG["url"], archive_path)
    print(f"Archive saved to: {archive_path}")
    return archive_path


def extract_archive(archive_path: Path, dataset_dir: Path, force: bool) -> Path:
    extract_root = dataset_dir / "raw"
    default_extracted_dir = extract_root / archive_path.stem

    if extract_root.exists() and force:
        shutil.rmtree(extract_root)

    existing_split_path = find_split_parent(extract_root, DATASET_CONFIG["train_file"])
    if existing_split_path is not None:
        print(f"Reuse existing extracted data: {existing_split_path}")
        return existing_split_path

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extract_root)

    extracted_dir = find_split_parent(extract_root, DATASET_CONFIG["train_file"])
    if extracted_dir is None:
        print(f"Archive extracted under: {default_extracted_dir}")
        return default_extracted_dir

    print(f"Archive extracted to: {extracted_dir}")
    return extracted_dir


def find_split_parent(search_root: Path, split_name: str) -> Optional[Path]:
    if not search_root.exists():
        return None

    direct_path = search_root / split_name
    if direct_path.exists():
        return search_root

    matches = list(search_root.rglob(split_name))
    if matches:
        return matches[0].parent

    return None


def resolve_split_path(extracted_dir: Path, split_name: str) -> Path:
    direct_path = extracted_dir / split_name
    if direct_path.exists():
        return direct_path

    nested_matches = list(extracted_dir.rglob(split_name))
    if nested_matches:
        return nested_matches[0]

    raise FileNotFoundError(f"Unable to find split file {split_name} under {extracted_dir}")


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        next(reader, None)
        return sum(1 for _ in reader)


def build_sample_file(source_path: Path, target_path: Path, sample_rows: int) -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8", newline="") as src, target_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty CSV file: {source_path}")

        writer.writerow(header)
        written_rows = 0
        for row in reader:
            if written_rows >= sample_rows:
                break
            writer.writerow(row)
            written_rows += 1

    return written_rows


def write_metadata(
    dataset_dir: Path,
    extracted_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    sample_path: Optional[Path],
    sample_rows: Optional[int],
) -> dict:
    metadata = {
        "dataset_name": DATASET_CONFIG["display_name"],
        "source_url": DATASET_CONFIG["url"],
        "raw_extract_dir": str(extracted_dir),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "test_path": str(test_path),
        "train_rows": count_rows(train_path),
        "valid_rows": count_rows(valid_path),
        "test_rows": count_rows(test_path),
        "sample_path": str(sample_path) if sample_path else None,
        "sample_rows": sample_rows,
    }
    metadata_path = dataset_dir / "dataset_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.output_dir)

    archive_path = download_archive(dataset_dir, args.force)
    extracted_dir = extract_archive(archive_path, dataset_dir, args.force)

    train_path = resolve_split_path(extracted_dir, DATASET_CONFIG["train_file"])
    valid_path = resolve_split_path(extracted_dir, DATASET_CONFIG["valid_file"])
    test_path = resolve_split_path(extracted_dir, DATASET_CONFIG["test_file"])

    sample_path = None
    sample_rows = None
    if not args.skip_sample:
        sample_path = dataset_dir / "normalized" / "criteo_sample.csv"
        sample_rows = build_sample_file(train_path, sample_path, args.sample_rows)

    metadata = write_metadata(
        dataset_dir=dataset_dir,
        extracted_dir=extracted_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        sample_path=sample_path,
        sample_rows=sample_rows,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
