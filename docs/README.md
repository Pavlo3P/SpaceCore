# Build the documentation

From the repository root:

```bash
pip install -e ".[docs]"
python scripts/check_tutorials.py
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Edit pages in `docs/source`. Open `docs/build/html/index.html` to check the result.
The build treats warnings as errors, matching CI.

## Deployment

[The docs workflow](../.github/workflows/docs.yml) builds pull requests and deploys
pushes to `main` or `master` to GitHub Pages. It also supports manual runs from
GitHub Actions. Set the repository's Pages source to **GitHub Actions** before
the first deployment.

Tutorials live directly in `docs/source/tutorials/*.rst`; there are no source
notebooks to regenerate. The check executes each page in its own namespace,
including its numerical assertions, using a non-interactive plotting backend.
