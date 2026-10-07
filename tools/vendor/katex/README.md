# KaTeX documentation renderer

`katex.js` is the unmodified CommonJS distribution from KaTeX 0.16.22.
The matching stylesheet and fonts are in `docs/assets/katex`; both copies
retain the upstream MIT license. `tools/math-assets.lock.json` records the npm
archive integrity and every vendored file's SHA256. These files are documentation
build resources, not Python inference dependencies.

The renderer uses the official [renderToString API](https://katex.org/docs/api)
with HTML and MathML output, strict errors and untrusted TeX. The maintained
[Arithmatex parser](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/)
protects math before rendering; code is handled by Markdown and Pygments.
Node.js is needed to rebuild pages. Reading the generated pages requires no
Node.js, network connection or JavaScript execution.

To update KaTeX, obtain the chosen official npm tarball, replace `dist/katex.js`,
`dist/katex.min.css`, `dist/fonts` and the accompanying license together, then
record the archive integrity and file hashes in the lock. Run the math tests,
rebuild the pages, and inspect both desktop and mobile rendering before review.
