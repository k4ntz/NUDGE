#!/bin/bash
# find_and_replace.sh

echo "Find and replace in current directory!"
echo "File pattern to look for? (eg '*.txt')"
read filepattern
echo "Existing string?"
read existing
echo "Replacement string?"
read replacement
echo "Replacing all occurences of $existing with $replacement in files matching $filepattern"

find . -name $filepattern -print0 | xargs -0 sed -i '' -e "s/$existing/$replacement/g"