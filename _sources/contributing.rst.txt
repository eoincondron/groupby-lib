Contributing
============

We welcome contributions to groupby-lib! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone https://github.com/YOUR_USERNAME/groupby-lib.git
    cd groupby-lib

3. Create a development environment::

    conda env create -f environment.yml
    conda activate groupby-lib-env

4. Install the package in development mode::

    pip install -e .[dev,test]

Running Tests
-------------

Run the full test suite::

    python -m pytest tests/ -v

Run tests with coverage::

    python -m pytest tests/ --cov=groupby_lib --cov-report=term-missing

Run specific tests::

    python -m pytest tests/test_groupby.py::TestGroupBy::test_sum -v

Code Style
----------

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Run all style checks::

    black groupby_lib tests
    isort groupby_lib tests
    flake8 groupby_lib tests
    mypy groupby_lib

Guidelines
----------

**Code Style**

* Use Python type hints for all public functions
* Follow PEP 8 naming conventions (snake_case for functions/variables)
* Add docstrings to all public functions and classes
* Use descriptive variable names
* Keep functions focused and small

**Testing**

* Write unit tests for all new functionality
* Use parametrized tests where applicable
* Test edge cases and error conditions
* Maintain high test coverage (>90%)

**Performance**

* Use NumPy arrays for numeric computations
* Leverage Numba JIT compilation where beneficial
* Profile code to identify bottlenecks
* Include benchmarks for performance-critical changes

**Documentation**

* Update docstrings for any API changes
* Add examples to docstrings where helpful
* Update the changelog for user-facing changes

Submitting Changes
------------------

1. Create a new branch for your feature::

    git checkout -b feature-name

2. Make your changes and commit them::

    git add .
    git commit -m "Add feature description"

3. Push to your fork::

    git push origin feature-name

4. Create a pull request on GitHub

Pull Request Guidelines
-----------------------

* Include a clear description of the changes
* Reference any related issues
* Ensure all tests pass
* Add tests for new functionality
* Update documentation as needed
* Keep changes focused and atomic

Code Review Process
-------------------

1. Automated checks must pass (tests, linting, type checking)
2. At least one maintainer review is required
3. Address any feedback from reviewers
4. Maintainer will merge once approved

Building Documentation
----------------------

Build the documentation locally::

    cd docs
    make html
    open build/html/index.html  # On macOS
    # Or navigate to docs/build/html/index.html

The documentation is automatically built and deployed on successful merges to main.

Releasing
---------

Releases are handled by maintainers:

1. Update version numbers
2. Update changelog
3. Create GitHub release
4. Automatic deployment to PyPI and conda-forge

Getting Help
------------

* **GitHub Issues**: Bug reports and feature requests
* **GitHub Discussions**: General questions and usage help
* **Email**: Contact maintainers directly for sensitive issues

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.