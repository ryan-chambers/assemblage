#!/usr/bin/env zsh

DATA_DIR="../wine_notes/rag_source_data"
IMPORT_SCRIPT="import_wine.py"

for file in "$DATA_DIR"/*.txt; do
  [[ -e "$file" ]] || continue

  # skip files whose basename ends with wip.txt
  if [[ "${file##*/}" == *wip.txt ]]; then
    echo "Skipping WIP file: \"$file\""
    continue
  fi

  echo "Running: Importing \"$file\""
  python3 "$IMPORT_SCRIPT" "$file"
done

echo "All done."
