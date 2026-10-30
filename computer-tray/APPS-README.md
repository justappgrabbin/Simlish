# Apps tray: how to use it and how to change it

**Opening it:** go to the **▦ Apps** screen in the left rail. Tap a tile to open that app full screen, and tap ✕ Close to go back.

**Address gate:** before any app opens, and before any file is saved or imported, its address (gate.line.color.tone.base, for example `18.5.3.4.2`) has to go through the address engine in `address-engine/resolver.mjs`. That engine was built from your State Space doc. If the address doesn't resolve, the app doesn't open and the file isn't saved. Each app asks for its address the first time you open it, and the computer remembers it after that. Every check is written to the gate ledger at the bottom of the Apps screen.

**Changing an app from inside the computer:** tap ✎ on a tile. Pick a file, edit it, and press **Save**. Your edits are kept in that browser and used the next time you open the app. **Export edits** saves them to a .json file, and **Import file** brings in a single file or a saved `*-edits.json`.

**Changing apps from your own computer (the permanent way):** on GitHub, open the branch `computer-apps-tray` of Simlish and edit the files under `computer-tray/`:
- `computer-tray/apps/apps.src.json` is the app list. Each entry has an id, name, icon, folder, entry and status.
- `computer-tray/apps/<app>/` holds each app's files. Add a folder with an `index.html` and give it an entry with `"status":"ready"`.
- Then run `node computer-tray/scripts-build-apps.mjs`, or let the APK build run it for you.

Each change you push to that branch makes a new test APK. You can also ask ChatGPT or Claude to make the edit for you through the plug-in in `mcp/`.

**Running it on a computer without the APK:** first `cp -r you-n-i-verse computer-tray/apps/youniverse` (and change its `computer/index.html` link to `../../index.html`), then run `cd computer-tray && python3 -m http.server 8770`, then open http://localhost:8770/index.html
