# Documentation Guide

This directory contains the Sphinx documentation for groupby-lib.

## Building Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
conda activate groupby-lib-dev
mamba install sphinx sphinx-rtd-theme myst-parser
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view.

### Clean Build

To start fresh:

```bash
cd docs
make clean
make html
```

## Documentation Structure

```
docs/
├── source/              # Source files
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main documentation page
│   ├── api_reference.rst   # API documentation
│   ├── getting_started.rst  # Getting started guide
│   ├── examples.rst    # Usage examples
│   └── contributing.rst     # Contributing guide
├── build/              # Built documentation (generated)
├── images/             # Images used in documentation
├── Makefile           # Build commands for Unix
└── make.bat           # Build commands for Windows
```

## Updating Documentation

### Adding New Pages

1. Create a new `.rst` file in `docs/source/`
2. Add it to the `toctree` in `index.rst`
3. Build and verify

### Documenting New Code

The API documentation is generated automatically from docstrings using Sphinx autodoc. When you add new public functions or classes:

1. Write docstrings following NumPy/Google style
2. The docstrings will automatically appear in the API reference
3. No manual updates needed!

Example docstring format:

```python
def my_function(param1: int, param2: str) -> float:
    """
    Short description of the function.

    Longer description providing more details about what the function does,
    its behavior, and any important notes.

    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2

    Returns
    -------
    float
        Description of return value

    Examples
    --------
    >>> result = my_function(42, "hello")
    >>> print(result)
    3.14
    """
    return 3.14
```

## GitHub Pages Deployment

Documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch.

### Workflow

1. Push changes to `main` branch
2. GitHub Actions builds documentation
3. Deploys to `gh-pages` branch
4. Available at: `https://<username>.github.io/groupby-lib/`

### Setup GitHub Pages (First Time)

1. Go to repository Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select branch: `gh-pages`, folder: `/ (root)`
4. Click Save

The documentation will be available at the URL shown in the Pages settings.

## Sphinx Configuration

Key configuration in `docs/source/conf.py`:

- **Theme**: `sphinx_rtd_theme` (Read the Docs theme)
- **Extensions**:
  - `sphinx.ext.autodoc` - Automatic API documentation
  - `sphinx.ext.napoleon` - NumPy/Google docstring support
  - `sphinx.ext.viewcode` - Source code links
  - `sphinx.ext.intersphinx` - Links to other documentation
  - `myst_parser` - Markdown support

- **Version**: Automatically extracted from package using `importlib.metadata`

## Intersphinx Links

The documentation can link to external documentation:

- Python: https://docs.python.org/3/
- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- Numba: https://numba.readthedocs.io/en/stable/

Use `:func:`numpy.array`` or `:class:`pandas.DataFrame`` to create links.

## Tips

### Preview Changes Quickly

Use Sphinx autobuild for live preview:

```bash
pip install sphinx-autobuild
sphinx-autobuild docs/source docs/build/html
```

Opens browser with live reload on changes.

### Check for Warnings

Sphinx warnings often indicate documentation issues:

```bash
cd docs
make html 2>&1 | grep WARNING
```

### Test Documentation Locally

Before pushing, always:

1. Build documentation locally
2. Check for warnings
3. Review in browser
4. Verify all links work

## Troubleshooting

### "Module not found" errors

Make sure the package is installed in your environment:

```bash
pip install -e .
```

### Version shows as "0.0.0+unknown"

The package needs to be installed for version detection to work:

```bash
pip install -e .
```

### Import errors in autodoc

Ensure all dependencies are installed in the documentation build environment.
