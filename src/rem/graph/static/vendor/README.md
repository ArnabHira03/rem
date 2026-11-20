# Vendored: cytoscape.js 3.34.4

Inlined into the graph window page at launch (`rem.graph.window.render_html`). Nothing is fetched at runtime.

- Version: 3.34.4
- Sources:
  - https://cdn.jsdelivr.net/npm/cytoscape@3.34.4/dist/cytoscape.min.js
  - https://cdn.jsdelivr.net/npm/cytoscape@3.34.4/LICENSE
- License: MIT (the standard MIT text, copyright The Cytoscape Consortium; see `LICENSE`)

`sha256sum` at vendoring time (2026-11-17):

```
081591ab735b2b6a3b31a3c60f8f3a3e2fee8c92939e0c7820ad1208571b6cd8  cytoscape.min.js
eb319c6e6f233607f71e8e2f450391751883cfc0eeb3ca7ef574c13d1d9c2203  LICENSE
```

# Vendored: three.js 0.186.1

Embedded into the graph window page at launch (`rem.graph.window.render_html`) as `data:` URLs in an import map,
for the Sphere tab. Nothing is fetched at runtime. These are the exact, unminified files from the npm package; the
only change, made when the page is assembled and never to the files here, rewrites the two `'./three.core.js'`
imports in `three.module.js` to the import-map name `'three/core'`.

- Version: 0.186.1
- Sources (the npm package `three@0.186.1`):
  - https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js
  - https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.core.js
  - https://cdn.jsdelivr.net/npm/three@0.186.1/LICENSE
- License: MIT (copyright three.js authors; see `three/LICENSE`)

`sha256sum` at vendoring time (2026-11-20):

```
9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea  three/three.module.js
9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6  three/three.core.js
8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc  three/LICENSE
```
