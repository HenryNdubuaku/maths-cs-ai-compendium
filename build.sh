#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Clean previous build artifacts
rm -rf src book

mkdir -p src

# Copy README as introduction
# Wrap ```math blocks in <div>$$...$$</div> so markdown parser leaves them alone
awk '
/^```math$/ { in_math=1; print "<div>\n$$"; next }
in_math && /^```$/ { in_math=0; print "$$\n</div>"; next }
{ print }
' README.md > src/README.md

# Symlink images so ../images/ from chapter dirs resolves correctly
ln -s ../images src/images

# Start SUMMARY.md
{
  echo "# Summary"
  echo ""
  echo "[Introduction](README.md)"
  echo ""
  echo "---"
} > src/SUMMARY.md

# Process each chapter directory
for chapter_dir in "chapter "*/; do
  # Strip trailing slash
  chapter_dir="${chapter_dir%/}"

  # Extract chapter number and name
  # "chapter 01: vectors" -> num="01", name="vectors"
  num="${chapter_dir#chapter }"
  num="${num%%:*}"
  name="${chapter_dir#*: }"

  # Clean directory name: lowercase, spaces to hyphens
  clean_name="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"
  dest_dir="src/ch${num}-${clean_name}"
  mkdir -p "$dest_dir"

  # Part heading in SUMMARY.md (title-cased chapter name)
  echo "" >> src/SUMMARY.md
  echo "# ${name^}" >> src/SUMMARY.md
  echo "" >> src/SUMMARY.md

  # Process each markdown file in the chapter
  for md_file in "$chapter_dir"/*.md; do
    [ -f "$md_file" ] || continue

    basename="$(basename "$md_file")"

    # Clean filename: "01. vector spaces.md" -> "01-vector-spaces.md"
    clean_basename="$(echo "$basename" | sed 's/\. /-/' | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"

    # Wrap ```math blocks in <div>$$...$$</div> so markdown parser leaves them alone
    awk '
/^```math$/ { in_math=1; print "<div>\n$$"; next }
in_math && /^```$/ { in_math=0; print "$$\n</div>"; next }
{ print }
' "$md_file" > "$dest_dir/$clean_basename"

    # Extract title from first H1, or derive from filename
    title="$(head -1 "$dest_dir/$clean_basename" | sed 's/^#* *//')"
    if [ -z "$title" ]; then
      # Derive from filename: "01-vector-spaces.md" -> "Vector Spaces"
      title="$(echo "$clean_basename" | sed 's/\.md$//' | sed 's/^[0-9]*-//' | tr '-' ' ')"
    fi

    # Add entry to SUMMARY.md
    echo "- [${title}](ch${num}-${clean_name}/${clean_basename})" >> src/SUMMARY.md
  done
done

# Rewrite README chapter links from GitHub directory URLs to mdBook paths
# e.g. "chapter%2001%3A%20vectors" -> "ch01-vectors/01-vector-spaces.html"
for chapter_dir in "chapter "*/; do
  chapter_dir="${chapter_dir%/}"
  num="${chapter_dir#chapter }"
  num="${num%%:*}"
  name="${chapter_dir#*: }"
  clean_name="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')"

  # URL-encoded form of the directory name (spaces=%20, colon=%3A)
  encoded="$(echo "$chapter_dir" | sed 's/ /%20/g; s/:/%3A/g')"

  # Find the first .md file in the chapter
  first_file="$(ls "src/ch${num}-${clean_name}/"*.md 2>/dev/null | head -1)"
  if [ -n "$first_file" ]; then
    first_basename="$(basename "$first_file" .md)"
    sed "s|${encoded}|ch${num}-${clean_name}/${first_basename}.html|g" src/README.md > src/README.md.tmp
    mv src/README.md.tmp src/README.md
  fi
done

echo "src/ prepared successfully."
echo ""

# Build with mdBook
mdbook build
echo ""
echo "Done. Output in book/"
