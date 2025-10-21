#!/bin/zsh

echo "🔄 Updating 'last_modified_at' fields before commit..."

# Prüft alle Markdown- und HTML-Dateien im Repo
#for file in $(git ls-files | grep -E '\.(md|html)$'); do
for file in $(git diff --cached --name-only | grep -E '\.(md|html)$'); do
  # Letztes Änderungsdatum aus Git holen
  last_modified=$(git log -1 --format="%cs" -- "$file" 2>/dev/null)
  [[ -z "$last_modified" ]] && continue  # Überspringen, wenn kein Datum vorhanden

  # Prüfen, ob Datei Front Matter enthält
  if grep -q "^---" "$file"; then
    if grep -q "^last_modified_at:" "$file"; then
      # Feld ersetzen
      sed -i.bak "s/^last_modified_at:.*/last_modified_at: $last_modified/" "$file"
    else
      # Feld nach der ersten '---' Zeile einfügen
      awk -v date="$last_modified" '
        /^---$/ && ++count==2 { print "last_modified_at: " date }
        { print }
      ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
    rm -f "$file.bak"
    git add "$file"
  fi
done

echo "✅ All 'last_modified_at' fields updated."