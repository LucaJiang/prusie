# Build and preview these pages

The site is static HTML in `docs/`, with local CSS and optional JavaScript.
Its Markdown sources are maintained alongside the generated pages. Navigation,
tables, code and reports remain readable without JavaScript. No remote fonts,
CDN or network connection is needed to read it.

## Regenerate

From the repository root, use the pinned lightweight renderer:

```sh
python -m pip install -r tools/docs-requirements.txt
python tools/build_docs.py
python tools/build_docs.py --check
python tools/check_docs.py
python tools/check_checkout.py --output-dir .ci-check
```

`--check` verifies generated HTML without writing. `check_docs.py` validates
relative links, anchors, local assets, image alternative text and version
consistency. It does not claim to check external links or replace a browser
review. Identical Markdown, configuration and renderer produce identical HTML;
the build adds no timestamp or network-dependent content.

## Preview offline

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000/`. Opening `docs/index.html` directly also works;
clipboard buttons depend on browser permissions, while commands remain
selectable text. The site uses relative paths and also works below a repository
subpath. Press Tab to reach navigation, code and scrollable tables.

## Configure GitHub Pages

For [prusie](https://github.com/LucaJiang/prusie), open **Settings → Pages → Build and deployment**.
Select **Source: Deploy from a branch**, **Branch: main**, and **Folder: /docs**,
then click **Save**. The committed `docs/.nojekyll` serves the prebuilt static files.
The expected address after a successful deployment is `https://LucaJiang.github.io/prusie/`.

Repository upload and Pages deployment are separate steps. This source delivery
prepares the site but does not claim an enabled Pages setting or a successful live
build. Check the repository's Pages status after saving the setting. SSH Git access
does not grant access to the Pages administration API.
