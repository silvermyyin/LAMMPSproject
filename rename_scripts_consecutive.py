import os
import re
import sys

SCRIPT_RE = re.compile(r"script_(\d{5})\.in$")


def main(directory: str):
    files = [f for f in os.listdir(directory) if SCRIPT_RE.match(f)]
    # Extract numeric value and sort
    files.sort(key=lambda x: int(SCRIPT_RE.match(x).group(1)))

    total = len(files)
    width = 5  # zero padding width

    renamed = []
    for idx, fname in enumerate(files, start=1):
        new_name = f"script_{idx:0{width}d}.in"
        if fname != new_name:
            src = os.path.join(directory, fname)
            dst = os.path.join(directory, new_name)
            # Ensure destination not already exists (shouldn't if mapping is correct)
            if os.path.exists(dst):
                raise RuntimeError(f"Destination {dst} already exists; aborting to avoid overwrite.")
            os.rename(src, dst)
            renamed.append((fname, new_name))
    print(f"Processed {total} files. Renamed {len(renamed)} files.")
    if renamed:
        print("Sample of changes (first 20):")
        for old, new in renamed[:20]:
            print(f"  {old} -> {new}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_scripts_consecutive.py <cleaned_dir>")
        sys.exit(1)
    main(sys.argv[1]) 