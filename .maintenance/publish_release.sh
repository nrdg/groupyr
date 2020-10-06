#!/bin/bash

# Check for the version number arg
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version_number>"
    exit 2
fi

# Store version number and upstream repo name
VERSION=$1
REPO=$(git remote -v | grep github.com:richford/groupyr.git | head -n 1 | cut -f 1)
BRANCH=main

/bin/bash .maintenance/update_changes.sh "${VERSION}"

echo "CHANGES.rst has been automatically updated to record the PRs since the previous tag."
while true; do
    read -p "Would you like to add your own edits to CHANGES.rst? [y/n]" yn
    case $yn in
        [Yy]* ) ${EDITOR} CHANGES.rst; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

while true; do
    read -p "Would you like to add your own edits to the notes for this release? [y/n]" yn
    case $yn in
        [Yy]* ) ${EDITOR} RELEASE.rst; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

/bin/bash .maintenance/update_zenodo.py

echo ".zenodo.json has been automatically updated to record the line editors since the previous tag."
while true; do
    read -p "Would you like to review/edit .zenodo.json for this release? [y/n]" yn
    case $yn in
        [Yy]* ) ${EDITOR} .zenodo.json; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done

git add CHANGES.rst RELEASE.rst .zenodo.json
git commit -m "Update CHANGES.rst, RELEASE.rst, and .zenodo.json"
git push "${REPO}" "${BRANCH}"
git tag "${VERSION}" -F RELEASE.rst
git push "${REPO}" "${VERSION}"
