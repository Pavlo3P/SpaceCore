# SpaceCore Documentation

SpaceCore uses Sphinx to build the documentation website from `docs/source`.
The generated HTML is written to `docs/build/html`.

## Local Build

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Build the HTML site:

```bash
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser to inspect the result.

## GitHub Pages Deployment

Documentation deployment is handled by `.github/workflows/docs.yml`.

The workflow:

1. installs the project with the `docs` extra;
2. builds the Sphinx HTML site from `docs/source` into `docs/build/html`;
3. uploads `docs/build/html` as a GitHub Pages artifact;
4. deploys the artifact with GitHub Actions Pages deployment.

Deployment runs automatically on pushes to `main` or `master`, and can also be
started manually from the GitHub Actions tab with `workflow_dispatch`.

Pull requests build the documentation but do not deploy it.

Before the first deployment, configure the repository's GitHub Pages source to
use **GitHub Actions**.
