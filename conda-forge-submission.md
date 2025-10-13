# Conda-Forge Submission Guide for groupby-lib

## Prerequisites (Action Required)

### 1. Add LICENSE File
You need to add a LICENSE file to your repository root. Since your setup.py specifies MIT license:

```bash
# Create MIT LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Eoin Condron

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

### 2. Publish to PyPI (Recommended)
While not strictly required, having your package on PyPI makes the conda-forge submission easier:

```bash
# Build and upload to PyPI
python -m build
twine upload dist/*
```

## Submission Process

### Step 1: Fork staged-recipes
1. Go to https://github.com/conda-forge/staged-recipes
2. Click "Fork" to create your fork

### Step 2: Create Recipe Branch
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes

# Create new branch
git checkout -b groupby-lib-recipe

# Create recipe directory
mkdir recipes/groupby-lib
cp /path/to/groupby-lib/conda-forge-recipe/meta.yaml recipes/groupby-lib/
```

### Step 3: Test Recipe Locally (Optional but Recommended)
```bash
# Install conda-smithy
conda install conda-smithy

# Lint the recipe
cd recipes/groupby-lib
conda smithy recipe-lint .

# Build locally (if you have conda-build)
conda build .
```

### Step 4: Submit Pull Request
1. Commit and push your changes:
```bash
git add recipes/groupby-lib/
git commit -m "Add groupby-lib recipe"
git push origin groupby-lib-recipe
```

2. Go to your fork on GitHub and create a Pull Request
3. In the PR description, mention: `@conda-forge/help-python` to request review from Python team

### Step 5: Address Review Feedback
The conda-forge team will review and may request changes. Common requests:
- Update dependencies versions
- Fix recipe formatting
- Add more comprehensive tests
- Update metadata

## Recipe Details

The prepared recipe (`conda-forge-recipe/meta.yaml`) includes:
- ✅ Source from GitHub v0.2.2 release with SHA256 hash
- ✅ Python >=3.10 requirement
- ✅ All runtime dependencies (numpy, pandas, numba, polars)
- ✅ Proper test imports and commands
- ✅ Complete metadata (license, homepage, summary)
- ✅ Entry point for CLI command

## Next Steps After Acceptance

Once your recipe is accepted:
1. A new repository `groupby-lib-feedstock` will be created
2. You'll be added as a maintainer
3. Future updates can be automated via bots or manual PRs to the feedstock
4. Your package will be available via: `conda install -c conda-forge groupby-lib`

## Resources
- [Conda-forge documentation](https://conda-forge.org/docs/)
- [Staged-recipes repository](https://github.com/conda-forge/staged-recipes)
- [Example recipes](https://github.com/conda-forge/staged-recipes/tree/main/recipes)