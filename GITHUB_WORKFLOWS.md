# GitHub Workflows Documentation

This document describes the CI/CD workflows for groupby-lib.

## Workflows Overview

### 1. CI Pipeline (`ci.yml`)
**Triggers:** Push to main/develop, Pull requests to main

**What it does:**
- Runs tests on multiple Python versions (3.10, 3.12)
- Runs on Ubuntu and Windows
- Uses conda/mamba for dependency management
- Runs flake8 linting
- Generates code coverage reports
- Uploads coverage to Codecov

**Dependencies:** Installed via mamba from conda-forge, then package installed with pip

### 2. Documentation (`docs.yml`)
**Triggers:** Push to main (docs/**), Pull requests with doc changes

**Jobs:**

#### check-docs
- Validates docstrings
- Checks README links
- Checks code examples in README
- Builds Sphinx documentation
- Uploads documentation artifacts

#### deploy-docs (main branch only)
- Builds documentation
- Deploys to GitHub Pages (gh-pages branch)
- Available at: https://eoincondron.github.io/groupby-lib/

#### update-changelog (main branch only)
- Generates changelog from git commits
- Uploads as artifact

**Dependencies:** Uses conda/mamba for core dependencies + Sphinx tools

### 3. Conda Build (`conda-build.yml`)
**Triggers:** Push to main, Pull requests, Manual dispatch

**What it does:**
- Builds conda package
- Tests conda package installation
- Validates conda recipe
- Uploads package artifacts

### 4. Performance Benchmarks (`performance-benchmarks.yml`)
**Triggers:** Push to main, Pull requests, Manual dispatch

**What it does:**
- Runs performance benchmarks
- Compares with baseline
- Uploads benchmark results

**Note:** Currently uses pip - could be optimized with conda/mamba

### 5. Release (`release.yml`)
**Triggers:** Manual workflow dispatch with version input

**What it does:**
- Creates git tags
- Builds distribution packages
- Publishes to PyPI
- Creates GitHub release

**Note:** Currently uses pip - build tools don't need conda

### 6. Security and Dependencies (`security-and-deps.yml`)
**Triggers:** Weekly schedule, Manual dispatch

**What it does:**
- Checks for security vulnerabilities (bandit, safety)
- Scans dependencies
- Generates security report

## Dependency Management Strategy

### Why Conda/Mamba + Pip?

We use a **hybrid approach**:

1. **Conda/Mamba for heavy dependencies:**
   - numpy, pandas, numba, polars, pyarrow
   - These benefit from pre-built conda binaries
   - Faster installation, better compatibility

2. **Pip for package installation:**
   - `pip install -e . --no-deps`
   - Installs package without reinstalling dependencies
   - Version constraints come from pyproject.toml

### Version Management

- **Package versions:** Defined in `pyproject.toml`
- **Workflow dependencies:** Installed without version pins via mamba
- **Pip validates:** When installing package, pip checks pyproject.toml constraints

This means:
- ✅ No version duplication between files
- ✅ Fast conda binary installs
- ✅ Version constraints enforced by pip
- ✅ pyproject.toml is single source of truth

## Adding New Dependencies

### Runtime Dependency

1. Add to `pyproject.toml` under `dependencies`:
   ```toml
   dependencies = [
       "numpy>=1.19.0",
       "your-new-dep>=1.0.0",
   ]
   ```

2. Add to workflow install steps:
   ```yaml
   mamba install -y \
     numpy \
     your-new-dep \
     ...
   ```

### Development Dependency

1. Add to `pyproject.toml` under `optional-dependencies.dev`:
   ```toml
   [project.optional-dependencies]
   dev = [
       "pytest>=6.0",
       "your-new-dev-tool",
   ]
   ```

2. Add to workflow install steps if needed in CI

## Workflow Maintenance

### Testing Workflow Changes

1. **Use workflow_dispatch:** Add manual trigger for testing
   ```yaml
   on:
     workflow_dispatch:  # Manual trigger
     push:
       branches: [main]
   ```

2. **Test in PR:** Push to feature branch and create PR

3. **Check workflow syntax:**
   ```bash
   # Use act to test locally (optional)
   brew install act
   act -l  # List workflows
   ```

### Common Issues

#### Conda environment not activating
- **Symptom:** "command not found" errors
- **Solution:** Ensure using `shell: bash -l {0}` on all run steps

#### Dependencies not found
- **Symptom:** Import errors during tests
- **Solution:** Check mamba install step includes all required packages

#### Slow CI
- **Symptom:** Workflow takes >10 minutes
- **Solution:** Consider using mamba (faster than conda)

#### GitHub Pages not updating
- **Symptom:** Old docs showing
- **Solution:**
  1. Check workflow succeeded
  2. Verify Pages settings: Settings → Pages → Source: gh-pages branch
  3. Clear browser cache

## Secrets and Tokens

Required secrets (set in repository settings):

- `CODECOV_TOKEN`: For uploading coverage reports (optional, can use GITHUB_TOKEN)
- `PYPI_API_TOKEN`: For publishing to PyPI
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## Best Practices

1. **Always test locally first:** Build and test before pushing
2. **Use matrix strategy:** Test multiple Python versions/OS
3. **Cache dependencies:** Conda/pip caches speed up workflows
4. **Fail fast:** Set `fail-fast: false` to see all failures
5. **Use artifacts:** Upload build outputs for debugging
6. **Add timeouts:** Prevent hung workflows
7. **Document changes:** Update this file when modifying workflows

## Monitoring

- **Workflow runs:** https://github.com/eoincondron/groupby-lib/actions
- **Coverage:** https://codecov.io/gh/eoincondron/groupby-lib
- **Documentation:** https://eoincondron.github.io/groupby-lib/
