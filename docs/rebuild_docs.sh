#!/bin/bash
# Script to rebuild STARLING documentation from scratch

set -e  # Exit on error

echo "========================================="
echo "Rebuilding STARLING Documentation"
echo "========================================="
echo

# Navigate to docs directory
cd "$(dirname "$0")"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
make clean
echo "✓ Cleaned"
echo

# Remove autosummary cache (forces regeneration)
echo "🗑️  Removing autosummary cache..."
rm -rf autosummary/_autosummary
rm -rf _build/doctrees
echo "✓ Cache removed"
echo

# Build HTML documentation
echo "📚 Building HTML documentation..."
make html
echo "✓ Build complete"
echo

# Check for warnings
if grep -q "WARNING" _build/html/*.html 2>/dev/null; then
    echo "⚠️  Some warnings detected in build output"
else
    echo "✓ No warnings detected"
fi
echo

echo "========================================="
echo "Documentation built successfully!"
echo "========================================="
echo
echo "View docs at: _build/html/index.html"
echo "Or run: open _build/html/index.html"
echo
