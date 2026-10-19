# Synthia / YOU-N-I-VERSE — master reassembly build

**Started:** 2026-09-29  
**Build identity:** one continuing source tree and one continuing Android package identity.  
**Current gate (2026-09-29):** continuing Android build on `reassembly/computer-2026-09-29`; first debug APK compiled in GitHub Actions and is available in the checkpoint ZIP linked below. Source census and Mac control integration remain open.

## Locked instructions

1. Inspect the actual user-supplied archives, files, and repository revisions before selecting the assembly base. An old report is a lead to verify, not a substitute for source inspection.
2. Preserve originals. Copy the selected source into the build, retain its origin and hash, and connect or repair it there. Do not reduce a supplied implementation to a description or reference when its executable source can be reused.
3. Continue the same build through checkpoints. Do not replace a working feature with a different implementation or strand it behind a new screen. Record migrations and regressions.
4. This is an Android-hosted computer with an operative local runtime and visible mesh, not a generic chat template. The system's existing semantic laws are to be recovered, connected, and exposed, not casually rewritten.
5. Each reasonable checkpoint produces an installable APK and a short receipt: source changes, live entry path, visible controls, test result, known gaps, package/version/hash, and what the human tester should try. A source ZIP or CI recipe alone does not count as an APK.
6. A feature counts as **present** when its source exists, **wired** when a path from the real app entry reaches it, and **verified** when that path runs and its result is observed. An isolated smoke test does not certify phone UI wiring.
7. If a required piece is missing or ambiguous, mark that exact gap and keep the requirement open. Never silently substitute a stub, placeholder, or invented component.

## Source inventory started

| Piece located | What is known now | Audit status |
| --- | --- | --- |
| `SYNTHIA_AUDIT_2026-09-20.md` and its status matrix | Earlier read-only audit of five archives; concrete file comparisons and tests. | Read; its claims require recheck against chosen source copies. |
| `SynthAI-FULL-SUITE-r21.22-closure.zip` | Earlier audit reported a verified core chassis. A saved 32.7 MB archive was located. | Source bytes pending inspection in this build. |
| `Synthia-Organism-OS-r22.0-MERGED.zip` | A saved 105 MB archive was located. Earlier audit used a similarly named `...MERGED-1.zip`; equality is unproven. | Compare exact files and hashes before applying audit conclusions. |
| `Synthia-Reassembly-v0.7.6.zip` and numbered copies | Saved 34 MB reassembly checkpoints were located. The unnumbered ZIP has SHA-256 `bcda343f28931fb0f4b0e35decf44b957d05030aa35dde149b9032b8365ba367` and 2,237 ZIP entries. | Initial read of the unnumbered copy completed; further source and build verification pending. |
| `Kimi_Agent_代码结构分析与建议(2)(1).zip` | Two similarly named saved copies were located, each 221,805 bytes. | Compare hashes and inspect source; no integration claim yet. |
| `Pure-Synthia-Prime-v0.4.1-UNIVERSAL-ENTRY.zip`, `Synthia-Computer-Stellar-Lab.zip`, `Synthia-Coherent-Autonomy-v0.1.0.zip` | Additional saved computer/runtime candidates located. | Pending census. |
| Earlier Hybrid, `src(1)`, and YOU-N-I-VERSE automaton archives | Earlier audit identified unique code and data, including 80 `src(1)` files without same-basename counterparts in r22. | Locate and recheck source archives; preserve unique parts. |
| Supplied Drive links and GitHub repositories | Prior conversations contain exact links. | Resolve each relevant item and record its revision and hash; no URL alone proves contents. |

The saved inventory is larger than this first pass. This table is deliberately an **open census**, not a claim that every file has been examined.

## Prior audit findings to verify first

- The ordinary r22 phone path was traced as `index.html → synthia-runtime.html → app.mjs → SynthiaUnit`. `UnifiedSynthiaOS` ran in a dedicated test but was bypassed by that user path.
- Hybrid reportedly created separate server and browser `SynthiaUnit` instances without a demonstrated state bridge. Its heartbeat receiver lacked a normal sender, and the server path lacked the daemon's file-backed storage shim.
- `src(1)` held unique triad, ingestion, Human Design, PHS, sentence, and research material. The PDF ingestor reportedly had a syntax error and the triad had data-shape errors.
- The YOU-N-I-VERSE automaton reportedly executed, while the full app failed typecheck/build and used memory-only runtime storage.

These are findings from the dated audit, not current integration results. The repair path must be traced through the actual APK entry and exercised on the target Android version.

## New direct archive observation (2026-09-29)

The unnumbered `Synthia-Reassembly-v0.7.6.zip` contains a root `README.md`, `package.json`, `app/`, `functional/`, `system/`, `vendor/ato-core/`, `recovered/`, and an optional self-hosted server tree. Its root package reports `0.7.5-reassembly` despite the ZIP name `v0.7.6`, while `app/package.json` reports `0.8.0` and `app/README.md` calls itself `v0.9.2`. These are internal version labels, not proof of release lineage. No `AndroidManifest.xml`, Gradle build file, or settings file was found in this archive's ZIP entry list. The direct archive check establishes contents and a packaging gap; it does not establish whether an external Android wrapper exists elsewhere. The root readme describes `app/index.html` as Cynthia's room and `index.mjs` as the JS engine entry. Verify the actual execution path before selecting it as the chassis.

## Seven Drive pieces supplied 2026-09-29

All seven were downloaded into separate audit copies; original Drive files were not changed. ZIP counts below exclude directory entries. Source status here is **inspected**, not **wired into the continuing APK**.

| Supplied piece | SHA-256 / contents | Direct finding |
| --- | --- | --- |
| `Synthia-browser-outcome-v0.6.10-1.zip` | `94ad2ffc9aebd54d2b1e964036e425444051e038da8c4766346115201322b37a`; 464 files | Contains `synthia.mjs`, browser executor/continuation/outcome code, resolver variants, proofs and tests. `npm run smoke:outcome` exited 0, but reported Chromium unavailable and exercised verifier logic only. Real browser action remains unverified. |
| `Synthia-Autopoietic-Assembly-v0.1.0-FULL-R21.22-MORPH-GAME-1.zip` | `5c5c5ca5ac9ac1538b0a6cd90ef55e88c890a07088e05c65d12c3d656ff95b0e`; 581 files | Source tree contains the r21 organism, morph game, hand pose, service worker and tests. Its WebView static boot graph test passed (178 reachable modules, zero Node-only imports). Morph game wiring smoke passed with gate 25, open hand, Space target, score 8, persistence and renderer input. This does not constitute an installed Android test. |
| `Synthia-v0.5.1-EXECUTION-PIPELINE-PATCH-1.zip` | `71af421b0e8ed9c6ed135b7b244c31bbe6c69b45b83a1b9a0724c87fb5df9365`; 19 files | Real 11-stage execution pipeline source and mobile bridge changes, expressly based on `Synthia-Unified-Capacitor-v0.5.1.zip`. Alone it fails due to missing `pure-synthia`. The exact 517-file baseline was found, and an additive audit copy passed `verify:execution` and `test:all` (see below). |
| `hd_cyber_morph_vqvae_ato_v1-2.zip` | `42d9dbb26997c8782f73fc43482cf73d00a5ff71e456d2385ab9384df0a9d7ab`; 6 files | TypeScript morph engine, HTML widget, Python VQ-VAE/training files and design docs. No model weights, Android project, or tests in this ZIP; inspect implementation and runtime dependencies before wiring image generation. |
| `Adaya-Automaton-ORGANISM-v0_11_15-r21_15-ATO-FACTORY-ARMS-SYNTHIA.zip (1)` | `7188eab42410ce5083fb921bd09a8dd1deba73319313f4bef12847bbd8ccf573`; valid ZIP with 515 files | Compared path and bytes against the assembly ZIP after stripping the latter's root folder: 496 common files, 460 identical, 36 changed, 85 assembly-only, and 19 Adaya-only files. The 19 Adaya-only paths are historical `.synthia/inbox` records and must still be preserved as provenance/state. |
| `ATO-Core-Klein-Browser-v0.4.0-gate-content-3.zip` | `3b1c54191a36de6cb7fd0e76236e497c423c2733cca3f25883b5d902fe163968`; 56 files | Dependency-free addressed Automaton engine, Klein operators, gate content, browser form/research, typed-port mesh and tests. Its full `node --test test/*.test.mjs` run exited 0. Integration into the chosen Synthia entry remains to be traced. |
| `Synthia-FULL-Executable-v4-Android.html` | `69bdf0d3ed754c0bdd0cfd665c79dc2cc65cd4d4b8393fbb9c12c5f0c1ccfc84`; 2,505,792 bytes | Single HTML page titled `Synthia · Full Executable v4 · Android`; it contains Autoling and ATO references plus local storage and IndexedDB code. It was not exercised in Android WebView in this pass; assess live behavior and overlap before copying. |

**New constraint:** None of these seven is an APK. The assembly ZIP has a viable static WebView graph test but no Android Gradle project in its entry list. The execution pipeline patch explicitly requires its source baseline. Do not present a patched source ZIP or browser smoke result as a phone-ready build.

### Exact Capacitor baseline and additive patch trial

The saved `Synthia-Unified-Capacitor-v0.5.1.zip` has 517 files and SHA-256 `3ec10122d0e35aa9022aaceb4db4f6df795f5ae256d16a84922c05c75f049f82`, exactly matching the baseline hash stated in the patch change record. In a separate audit copy, the 19-file patch was overlaid without altering either original archive. `npm run verify:execution` passed all 11 stages and observed execution cache reuse on a second run. `npm run test:all` exited 0, including the pipeline, persistence, and mobile source tests. This is a working source integration trial, **not the chosen final chassis or APK**.

The baseline includes `apps/mobile/capacitor.config.ts` with package ID `com.synthia.phone` and an `apk:debug` script. Its ZIP does not contain a generated Android Gradle directory. The patch's own build evidence says Vite was unavailable and its mobile build stopped before bundling. An actual Android build and phone test remain open.

## Assembly sequence and APK gates

| Gate | Work | APK acceptance |
| --- | --- | --- |
| 0. Census | Hash archives, enumerate files, identify duplicate/unique variants, source ownership, license, dependencies, build manifests, and Android constraints. Choose the strongest existing computer chassis by demonstrated runtime reachability. | No APK promised at this gate; deliver the source map and selected continuing tree. |
| 1. Existing computer boots | Copy the selected chassis, build its existing Android shell, preserve package identity, and start its real local runtime. | APK installs and opens on the agreed minimum Android version; browser/UI and runtime report the same identity and state after restart. |
| 2. Foreground and execution | Connect the existing mesh/UI to the actual resident runtime, tools, state-space/execution engine, and visible status. Instrument errors instead of swallowing them. | APK demonstrates a user action through UI → execution → visible result and reports the tool/path that ran. |
| 3. Recovered semantic pieces | Copy in and connect Autoling, Disseminer, the addressing/sentence substrate, dimension engines, ATO/semantic transformations, and unique datasets in dependency order. | APK exposes and exercises each added piece through the real UI with a trace and persistent output. |
| 4. Computer capabilities | Integrate the user's Kimi/client tools, local files, browser, builder, execution state, and eligible Linux/container pieces where supported by actual source and Android version. | APK exercises real file and tool operations, reboot persistence, and offline behavior; failures are visible. |
| 5. Full system regression | Check identity, sovereignty/export, recovery, performance, permissions, and all previously verified flows. | Signed/testable APK with a feature ledger showing every source piece as present, wired, verified, or blocked with evidence. |

Gate numbering is a test cadence, not permission to discard work between gates. Each APK grows from the same prior checkpoint. The exact gate contents may change when the source census shows actual dependencies.

## Permanent evidence per component

For every copied piece record: original archive/URL and revision, original path and SHA-256, destination path, modifications as a diff, dependencies, Android compatibility, UI entry, execution trace, persistence behavior, and gate where it was first verified. A copied file without a reachable user path remains **present** only.

## Next actions

1. Materialize and hash the located reassembly, r21/r22, Kimi, and other likely chassis archives; inspect manifests and entry points without altering the originals.
2. Reconcile the earlier five-archive audit against those exact bytes and extend the census to the later supplied files and links.
3. Name the continuing source tree only after this comparison; then make Gate 1's APK from that tree and supply it for hands-on testing.

## Current checkpoint receipt

**2026-09-29:** Located the dated audit and matrix, read their findings, and identified several saved candidate archives. Inspected the unnumbered reassembly ZIP without modifying it and recorded its hash, entry count, version label mismatch, and lack of an Android project within that ZIP. Read and compared the seven newly supplied pieces; ran focused existing tests in extracted audit copies. The workspace had no existing git repository or Android project at the start of this pass. No APK was built yet. The broad source audit remains in progress.

## Continuing computer checkpoint, 2026-09-29

The live continuing source is the user's `justappgrabbin/SynthAi` repository branch [`reassembly/computer-2026-09-29`](https://github.com/justappgrabbin/SynthAi/tree/reassembly/computer-2026-09-29), commit `a5b1157f570cf3192af534de47d4185864405bce`, tree `4d0e842d01ca91d0ad4164647c3370b4f5efdfdb`. Its parent is main commit `d7b355707ab7d2ea72c7e72a1ad5e546d1b6e522`. The source lives under `computer/`; do not start a replacement tree. Local commit `445be4a32ef8acdd8e1a7d1a7a30e6d4943850cb` has the identical tree. `computer/REASSEMBLY_CHECKPOINT.md` has the execution receipt and provenance.

The original Kimi Web Linux desktop and exact 0.5.1 execution pipeline baseline plus patch were copied. The entire Kimi ZIP is preserved under `computer/donors/source-archives/`; the original operator/state-space/automata modules remain in `computer/engines/unified-execution-spine/`. The active desktop ships five reachable apps: Files, Import, Synthia chat, Code Editor, and Text Editor. Unshipped Kimi demo apps are preserved in the ZIP, outside the active build. No Paper product poster board or fake deployment buttons were shipped.

The Computer's import endpoint preserves exact file bytes and the source archive, derives an execution address, registers a node in the execution bridge capability mesh and automata knowledge projection, registers parsed code in the Code DNA learner, and binds an ATO recall automaton to a VQ-VAE cue. Clicking the file in Import or Files resolves and fires the ATO address, checks the SHA-256, and opens its content. Kimi desktop file state persists through the Computer state store. Chat calls the original MorphChat/AutoLing/DISEMINER bridge and the state/execution context and returns readable text; a file inventory question is answered from actual imported records. Code Editor Run calls the same execution route.

`npm run build` passed. `npm run test:reassembly` passed import, restart, addressed recall, and execution. Focused upstream tests passed 10/10. Wider tests passed 22/24, with two penta ephemeris tests requiring unavailable `pyephem`. Live HTTP checks returned desktop 200, ZIP ingestion of three addressed records 200, exact ATO activation 200, 11-stage execution 200, filesystem 200, and grounded readable chat 200. This proves the desktop computer interface, chat, and ingester service path. A graphical browser click test was unavailable because the local Playwright Chromium executable was absent. These facts do not certify an Android install.

**Outstanding:** Build a native Android service host and an actual APK from this same branch, then verify phone UI clicks, restart persistence, and checkpoints. Restore full primitive-level dynamic operator resolution: the current five-field allocation/source-order primitive graph is narrower than the historical claimed capability. The direct runtime runs plain JS but reports exported ES modules as unsatisfied; add a genuine multi-file project adapter before claiming that feature. The broader supplied-source census remains open. DNA/Paper Seed multi-page growth is retained as future integration work, not claimed as a live feature. The user will specify preinstalled programs after the complete computer interface/chat/ingester milestone; do not guess that list.

## Android and Mac checkpoint, 2026-09-29

The Android wrapper in the same `computer/` tree starts the existing Node execution service in process and opens the Kimi interface in WebView. It uses Android's file chooser, versioned asset extraction, persistent app data, and mobile panel layouts. The ZIP reader now uses Node's built-in zlib instead of a shell unzip dependency. The build workflow pinned its Node mobile distribution hash and built a debug APK successfully on GitHub Actions run [36620586938](https://github.com/justappgrabbin/SynthAi/actions/runs/36620586938), commit `468ac7c286aca505812ca95ed5860b50ddad46e3`. The artifact is [checkpoint 1 ZIP containing the APK](https://drive.google.com/file/d/14KSgbfDl3OPA1YAmkG_sQ1JkA6kp1vBr/view?usp=drivesdk), 39,540,956 bytes as the Actions artifact. The debug APK itself and its install behavior have not been inspected on a physical phone yet. The earlier "no APK" statements above record older gates and are superseded by this compile result.

The next continuing commit is `d33b2b88d81396fdf55cf115c5c171363e127285`. It adds exact source and original ZIP for `StellarVM-Mac-v0.1.0` (ZIP SHA-256 `597279660cd4182bf025a2392d36005ea8ea0ae3034c111804022a6ae8a8af43`), excludes the Mac source from the Android runtime asset payload, and routes every `/api/execute` file through the existing addressed ATO activation and hash check. The local reassembly test and Kimi build passed; its successor APK build should be checked separately. StellarVM is a macOS Virtualization.framework Linux host. It has no guest agent or authenticated phone control endpoint in the supplied v0.1.0 source, so the Android client does not yet control the Mac. This must be connected and verified on a Mac; a Mac-like screen would not satisfy that gate.

Remaining acceptance: install and operate the APK on a device, verify import/chat/file recall/run and reboot persistence, then connect the Mac guest control seam to the same canonical address and execution receipt contract. The self-evolving DNA, Paper Seed, and preinstalled program inventory are future gates; do not label them live prematurely.

### Verified Android APK handoff

The continuing branch head is [`22efb2676ab197ce6e91987d245ccade8e9f4efa`](https://github.com/justappgrabbin/SynthAi/commit/22efb2676ab197ce6e91987d245ccade8e9f4efa). [Actions run 36622323102](https://github.com/justappgrabbin/SynthAi/actions/runs/36622323102) passed Kimi build, reassembly test, APK assembly, and ZIP integrity check. The reconstructed APK is **112,010,294 bytes**, SHA-256 **`9fc1983e3afeb1f10663231d349c74ba57fe525d9ec032baaa7683085ae77581`**, with Android manifest, two `libnode.so` ABIs, and bundled computer assets. Its ZIP archive validated locally; this is a build validation, not a phone install test. [Download the checkpoint ZIP containing `app-debug.apk`](https://drive.google.com/file/d/1P2rJyHTYHkfKxvse33uSSixvvoeu3ael/view?usp=drivesdk). The same [GitHub Actions artifact](https://github.com/justappgrabbin/SynthAi/actions/runs/36622323102) remains a second handoff. The single APK exceeded the connected Drive 100 MiB upload limit, so the 39,541,988-byte ZIP carries the installable file. The package ID is `org.synthai.computer`; it installs separately from the older `com.synthia.phone` identifier and does not migrate that app's data.

The newer Android runtime extraction uses package update time and a completion marker, keeping persistent computer data in a separate directory. The file admission limit now matches the six-megabyte upload boundary; a 600 KB binary file passed write, restart, exact addressed ATO recall, and byte comparison. Execution from `/api/execute` now requires that addressed activation and hash check before dispatch. These are additive changes around the user's original Kimi and execution modules.
