# Paper — browser runtime

Paper is a local JavaScript application with a multi page site builder, Seed tree, Cardstock templates, Loom project composition, Papermate inherited agent traits, Notebook version lineage, and ZIP/text/code ingestion. It exports a deployable static site ZIP and complete history JSON. The app is installable as a PWA after its first online visit and works without an application backend.

## Source integration

- `dist/ato/` is an unmodified copy of ATO Core Klein Browser v0.4.0 source. Paper generation invokes `ATOEngine.compile(...).execute(...)` and stores its five-stage trace. The original `bootstrapATO` tools and `MorphingWorkspace` are mounted in the interface.
- `dist/automata/` is an unmodified copy of the browser-safe automata ingestion/execution kernel from the Synthia v0.5.1 patch. Uploaded artifacts pass through its compile/observe/address cycle. JavaScript execution is explicit and occurs in its isolated iframe.
- The Synthia patch's `IntegratedSynthiaSystem` and `ExecutionPipeline` require a separate PRE-DNA-WIRING checkpoint absent from the supplied patch. They are not presented as active here.
- The large Adaya organism archive includes local server, MCP and optional Supabase paths. Its full engine cannot be substituted into a zero-backend Android WebView without changing its execution contract. It remains a source reference rather than a claimed runtime integration.

## Run

Serve `dist/` from any static host or local HTTP server. Open `index.html` through that host. No API keys, Node server, or remote model is used at runtime. Exported sites contain index.html, distinct page HTML files, and a project/history JSON copy.

## Android packaging

The companion `Paper-Mobile-Termux-Build.zip` adapts the supplied Synthia Compiler Rescue toolchain for an on-device Termux build. This workspace lacks Android SDK build tools (`android.jar`, `aapt2`, Gradle and `apksigner`), so a signed APK cannot be produced here. `dist/` is the ready web payload for a trusted Android WebView packaging environment; the installable PWA is immediately usable on Android without a backend. Do not rename the ZIP as an APK.
