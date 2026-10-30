# Swarm Computer plug-in (MCP server)

A small plug-in that lets ChatGPT, Claude, and other AI apps see and edit the apps inside your Swarm Computer.
It is plain Node.js with no extra packages. You need Node 18 or newer.

## What it can do
| Tool | What it does |
|---|---|
| `list_apps` | Lists the apps in the tray |
| `resolve_address` | Runs an address like `18.5.3.4.2` through your address engine and returns the state and the B‑H‑A‑N‑T‑U‑G‑H layers |
| `list_app_files` / `read_app_file` | Lists and reads the files in one app's folder |
| `write_app_file` | Writes a file into an app's folder. **You have to give an address.** The address engine checks it first, and if it fails, nothing gets written. Every write is logged in `apps/address-ledger.jsonl` |
| `search` / `fetch` | The two tools ChatGPT's connectors look for |

## Running it
From the folder that contains `mcp/`:
- **Local mode (Claude Desktop, Cursor, Claude Code):** `node mcp/server.mjs`
- **Web mode (ChatGPT, anything that takes a URL):** `node mcp/server.mjs --http --port 8787`
  - New-style address: `http://localhost:8787/mcp`
  - Old-style SSE address: `http://localhost:8787/sse`

## Claude Desktop
Open Settings → Developer → Edit Config. Add the following, using your own full path:
```json
{ "mcpServers": { "swarm-computer": { "command": "node", "args": ["/FULL/PATH/computer/mcp/server.mjs"] } } }
```
Then restart Claude. The tools appear under the 🔌 icon.
For Claude Code, run: `claude mcp add swarm-computer -- node /FULL/PATH/computer/mcp/server.mjs`

## ChatGPT
ChatGPT can only reach a public **https** address, so it can't see a server on your own computer directly.
1. Start web mode: `node mcp/server.mjs --http --port 8787`
2. Get a free temporary https link with `npx cloudflared tunnel --url http://localhost:8787`. It prints a `https://….trycloudflare.com` address. No account or paid hosting is needed.
3. In ChatGPT, open Settings → Apps & Connectors → Advanced, turn on **Developer mode**, then choose **Create**. Paste `https://….trycloudflare.com/mcp` and set Authentication to "No authentication".
4. In a chat, turn the connector on and ask it to list the apps.

Warning: anyone who has that tunnel link can read and write your app files while it's running. Close the tunnel when you're done.

## Manifest
ChatGPT used to require an `ai-plugin.json` file, but the old plugin system is retired. Today's connectors only need the URL above.
`server.json` uses the standard MCP registry format, for tools that read one.

## Testing
`node mcp/test-client.mjs` runs every tool over stdio, HTTP and SSE.
