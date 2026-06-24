import re

with open("ui/app.py", "r", encoding="utf-8") as f:
    content = f.read()

old = (
    '    # Check if this image matches an existing selected image by perceptual hash\n'
    '    matched_stem = None\n'
    '    for p in sorted(SELECTED_DIR.iterdir(), key=lambda x: x.name):\n'
    '        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:\n'
    '            if have_same_image(temp_path, p):\n'
    '                matched_stem = p.stem\n'
    '                break\n'
    '\n'
    '    if matched_stem:\n'
    '        image_target = SELECTED_DIR / f"{matched_stem}{temp_path.suffix.lower()}"\n'
    '    else:\n'
    '        existing = count_files(SELECTED_DIR, IMAGE_EXTS)\n'
    '        stem = f"keyboard_{existing + 1:04d}"\n'
    '        image_target = next_available_path(SELECTED_DIR, stem, temp_path.suffix.lower())'
)

new = (
    '    # Always save with a new unique sequential filename\n'
    '    existing = count_files(SELECTED_DIR, IMAGE_EXTS)\n'
    '    stem = f"keyboard_{existing + 1:04d}"\n'
    '    image_target = next_available_path(SELECTED_DIR, stem, temp_path.suffix.lower())'
)

if old in content:
    content = content.replace(old, new)
    with open("ui/app.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed successfully!")
else:
    print("Pattern not found. Searching...")
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "have_same_image" in line:
            start = max(0, i - 3)
            end = min(len(lines), i + 5)
            for j in range(start, end):
                print(f"{j+1}: {lines[j]}")
            print("---")
