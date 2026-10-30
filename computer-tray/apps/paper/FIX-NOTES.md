# Paper sovereign — fix notes

Plain English. No LLM. No Trident. No API keys.

## What was broken

1. **Uploads looked dead.** Files only got listed. The receipt said `compiled-without-execution` because every upload called the kernel with `execute:false`. Nothing useful happened with the file.
2. **HTML was labeled `text`.** The classifier did not know `.html` / `text/html`, so Pages never treated uploads as real pages.
3. **HTML / Markdown / text uploads did not become site pages.** Images went into assets but felt unused (no Gallery unless you built one by hand).
4. **The receipt was opaque developer JSON** with the message “compiled and retained,” which looked like failure.
5. **Generate pages was anemic.** “A small garden site” basically made Home and little else, with empty bodies.

## What we fixed

### Classification (`automata/artifact-normalizer.mjs`)
- Detects **html**, **markdown**, **css**, **image** from extension / mime (and HTML sniffing).
- Keeps zip / javascript / json / wasm / text / binary as before.

### Kernel helpers
- **primitive-reducer** understands the new kinds (titles from `<title>` / `# heading`).
- **browser-execution-surface** can “execute” html/md/css/text/image/zip locally (ingest / parse), not only json/js/wasm.

### Upload path (`app.js`)
- **JSON / HTML / MD / TXT / CSS / images:** run through the kernel with execution on.
- **JavaScript:** still compile-only on upload; **Run** still executes in the sandbox.
- **HTML / MD / TXT:** become real **pages** (title from filename or `<title>` / `# heading`; HTML tags stripped for the body).
- **Images:** go into **assets** and create/update a **Gallery** page if missing.
- Banner messages are human: e.g. `Imported x as a page`, `Imported image into assets`, `JS ready — press Run`.
- Receipt shows a short summary first; full JSON is under a collapsed “Technical receipt”.

### Generate pages
- Local parser now builds sensible page sets for garden, portfolio, shop, blog, docs, restaurant, music, events, courses, etc.
- Each page gets a short stub body so Preview is not empty.
- Cardstock / growth suggestions also fill stub bodies.

### Other
- Added missing `manifest.webmanifest` and `icon.svg` (were 404).
- Bumped service worker cache to `paper-offline-v3`.
- Gallery / Work / Portfolio / Garden pages show uploaded images in the static preview/export.

## How to verify

1. Open **http://127.0.0.1:8765/** (hard refresh once so the new SW/cache loads).
2. **Paper** tab → name `Garden`, description `A small garden site` → **Generate pages**.
   - Expect pages like Home, Garden, Visit, About and a Preview with those sections.
3. **Files** tab → upload an `.html` file.
   - Expect a green notice like “Imported … as a page”, Sources kind `html`, and that page listed under Paper.
4. Upload a `.js` file → notice “JS ready — press Run” → press **Run**.
5. Upload an image → assets grow, Gallery page appears, Preview shows the image on Gallery.
6. Click every sidebar tab (Seed, Loom, Cardstock, Papermate, Notebook, Files, Preview). None should error.
7. **Export site** downloads a ZIP with `index.html` + `pages/`.

### Automated checks

```bash
cd /workspace/paper-sovereign
node smoke-check.mjs
```

## Remaining gaps (honest)

- CSS uploads are kept in Sources but are **not** yet injected into the exported site stylesheet.
- ZIP uploads are inspected by the kernel; entries are **not** auto-exploded into pages/assets in the UI yet.
- JS with `import`/`export` still cannot Run until dependencies are linked (same as before — sandbox says so).
- This folder is the **browser Paper studio**, not a chat persona named Synthia/Cynthia.
- Phone builds that still ask for Trident are a **different install**; this sovereign tree does not use Trident.

## Files touched

- `app.js`, `style.css`, `index.html`, `sw.js`
- `automata/artifact-normalizer.mjs`
- `automata/primitive-reducer.mjs`
- `automata/browser-execution-surface.mjs`
- `manifest.webmanifest`, `icon.svg` (new)
- `smoke-check.mjs`, `browser-smoke.html` (test helpers)
- `FIX-NOTES.md` (this file)

Originals under `/workspace/zips/` were not modified. Backups of edited sources sit next to them as `*.bak`.

## TinyLlama Neural (2026-10-26)

- Added `neural.mjs`: WebLLM TinyLlama in-browser (legacy `createEngine` → `CreateMLCEngine` + `TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC`).
- First-class **Neural** sidebar tab: Load / progress / prompt / Run / output; optional project context.
- Paper tab: **Draft with TinyLlama** uses the same engine (`max_tokens: 256`).
- No chat API keys, no backend for the model.
