#!/usr/bin/env python3
"""Test the proposal parser to debug why metadata isn't being extracted."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROPOSALS_DIR = PROJECT_ROOT / "proposals"

# Test on one proposal
test_file = PROPOSALS_DIR / "039-warp-specialized-pingpong-chunkwise-linear-rnn.md"
content = test_file.read_text()
lines = content.split('\n')

print(f"Testing: {test_file.name}")
print(f"First line: {repr(lines[0])}")
print(f"First line stripped: {repr(lines[0].strip())}")
print(f"Equals '---': {lines[0].strip() == '---'}")
print()

metadata = {
    "id": test_file.stem,
    "filename": test_file.name,
}

# Try parsing YAML frontmatter
if lines and lines[0].strip() == '---':
    print("✓ Found frontmatter start")
    in_frontmatter = True
    frontmatter_end = None

    for i, line in enumerate(lines[1:], start=1):
        print(f"  Line {i}: {repr(line)}")
        if line.strip() == '---':
            print(f"  ✓ Found frontmatter end at line {i}")
            frontmatter_end = i
            break
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            print(f"    Extracted: {key} = {value}")
            metadata[key] = value

    print()
    print(f"Frontmatter end: {frontmatter_end}")
    print(f"Metadata extracted: {metadata}")

    # Extract title from content after frontmatter
    if frontmatter_end:
        for line in lines[frontmatter_end + 1:]:
            if line.startswith('# '):
                metadata['title'] = line.lstrip('# ').strip()
                print(f"Title: {metadata['title']}")
                break

print()
print("Final metadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")
