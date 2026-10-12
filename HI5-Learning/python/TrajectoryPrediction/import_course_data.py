"""Import the preprocessed Lisam ZIP without changing existing data."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "datasets"
    with zipfile.ZipFile(args.archive) as archive:
        entries = []
        for member in archive.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if "sinD" not in parts or member.is_dir():
                continue
            relative = Path(*parts[parts.index("sinD"):])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Invalid archive path: {member.filename}")
            entries.append((member, target))
        if not any(p.parent.name == "test" and p.parent.parent.name == "sinD" for _, p in entries):
            raise ValueError("ZIP does not contain sinD/test. Use the preprocessed course ZIP from Lisam.")
        for member, target in entries:
            if target.exists():
                raise FileExistsError(f"Data already exists: {target}. No files were overwritten.")
        for member, target in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"Imported {len(entries)} files into {destination}")


if __name__ == "__main__":
    main()
