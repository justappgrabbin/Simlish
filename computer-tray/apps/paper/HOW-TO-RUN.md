# Run Paper (sovereign)

No backend. No Supabase. No Trident. No API keys for chat.
ATO / Klein stay local. TinyLlama is optional and runs **in the browser** via WebLLM (WebGPU).

```bash
cd /workspace/paper-sovereign
python3 -m http.server 8765 --bind 127.0.0.1
```

Open: http://127.0.0.1:8765/

Hard-refresh once after updates (service worker cache).

## TinyLlama (Neural tab)

1. Open the **Neural** sidebar tab.
2. Click **Load TinyLlama** once — WebLLM downloads `TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC` into browser cache (tries legacy `createEngine("TinyLlama-1.1B-Chat-v1.0-q4f16_1")` first).
3. Type a prompt, optionally include project pages / uploaded source, click **Run** (`max_tokens: 256`).
4. On the **Paper** tab, **Draft with TinyLlama** fills Home page body from your description once the model is loaded.

All inference stays on-device. Needs a WebGPU-capable browser.

See `FIX-NOTES.md` for earlier studio fixes.
