import os
import re
import json
import csv
import sys
from collections import Counter, defaultdict

STYLE_COMMANDS = {
    # command : index of style token after splitting
    "pair_style": 1,
    "bond_style": 1,
    "angle_style": 1,
    "dihedral_style": 1,
    "improper_style": 1,
    "kspace_style": 1,
    "run_style": 1,
    "pair_modify": 1,
    # commands where style comes after ID and group
    "fix": 3,       # fix id group style args
    "compute": 3,   # compute id group style args
}

STOPWORDS = {"all", "none", "yes", "no", "on", "off"}
ALPHA_RE = re.compile(r"[a-zA-Z]")


def clean_line(line: str) -> str:
    """Remove inline comments and strip."""
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def join_continuations(lines):
    """Merge lines ending with '\\' or '&' with the next line."""
    merged = []
    buffer = ""
    for line in lines:
        stripped = line.rstrip()
        cont = None
        if stripped.endswith("\\"):
            cont = "\\"
        elif stripped.endswith("&"):
            cont = "&"
        if cont:
            buffer += stripped[:-1] + " "
            continue
        else:
            buffer += stripped
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def extract_keywords_from_file(path: str):
    with open(path, "r", errors="ignore") as f:
        raw_lines = f.readlines()

    lines = join_continuations(raw_lines)
    keywords = []

    for line in lines:
        cline = clean_line(line)
        if not cline:
            continue
        tokens = cline.lower().split()
        if not tokens:
            continue
        cmd = tokens[0]
        # skip obvious non-command lines (e.g., numbers)
        if not ALPHA_RE.search(cmd):
            continue
        keywords.append(cmd)

        if cmd in STYLE_COMMANDS:
            idx = STYLE_COMMANDS[cmd]
            if len(tokens) > idx:
                style_tok = tokens[idx]
                # ensure alphabetic content and not stopword
                if ALPHA_RE.search(style_tok) and style_tok not in STOPWORDS:
                    keywords.append(style_tok)

    # remove stopwords and purely numeric tokens
    filtered = [kw for kw in keywords if kw not in STOPWORDS and ALPHA_RE.search(kw)]
    return sorted(set(filtered))


def main(directory: str):
    per_script = {}
    freq = Counter()

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".in"):
            continue
        path = os.path.join(directory, fname)
        kws = extract_keywords_from_file(path)
        per_script[fname] = kws
        freq.update(kws)

    # output files
    out_json = os.path.join(directory, "keywords_per_script.json")
    out_csv = os.path.join(directory, "global_keyword_freq.csv")

    with open(out_json, "w") as jf:
        json.dump(per_script, jf, indent=2)
    print(f"Wrote per-script keywords to {out_json}")

    with open(out_csv, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["keyword", "count"])
        for kw, cnt in freq.most_common():
            writer.writerow([kw, cnt])
    print(f"Wrote global frequency table to {out_csv}")

    print("Top 20 keywords:")
    for kw, cnt in freq.most_common(20):
        print(f"  {kw:20s} {cnt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_keywords.py <cleaned_dir>")
        sys.exit(1)
    main(sys.argv[1]) 