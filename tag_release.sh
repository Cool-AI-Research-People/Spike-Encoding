#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.0"
    exit 1
fi

VERSION=$1

# Validate version format (simple check for x.y.z)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format x.y.z (e.g., 0.2.0)"
    exit 1
fi

echo "Setting version to $VERSION"

# Update version in pyproject.toml
sed -i "s/^version = .*/version = \"$VERSION\"/" pyproject.toml

# Check if sed worked
if ! grep -q "version = \"$VERSION\"" pyproject.toml; then
    echo "Error: Failed to update version in pyproject.toml"
    exit 1
fi

echo "Updated pyproject.toml with version $VERSION"

# Create and push tag
git add pyproject.toml
git commit -m "bump version to $VERSION"
git tag "v$VERSION"
git push origin main
git push origin "v$VERSION"

echo "Tagged and pushed v$VERSION"