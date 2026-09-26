# Build the documentation

From the repository root:

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Edit pages in `docs/source`. Open `docs/build/html/index.html` to check the result.
The build treats warnings as errors, matching CI.

## Deployment

[The docs workflow](../.github/workflows/docs.yml) builds pull requests and deploys
pushes to `main` or `master` to GitHub Pages. It also supports manual runs from
GitHub Actions. Set the repository's Pages source to **GitHub Actions** before
the first deployment.
