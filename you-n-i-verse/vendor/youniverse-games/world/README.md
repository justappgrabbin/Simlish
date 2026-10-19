# World Module Interface

Future worlds (Consciousness Realm, SimWorldForge, ...) plug in here.

A world module exports a factory (`create()` or a bare function) returning an
instance implementing:

- `enter(ctx)` — called when the player enters. `ctx = {seed, gameState,
  adapter, morphEngine, renderer}`. Must return the entry scene/state object.
- `handleChoice(choice)` — apply a player choice; return the new scene/state.
  Consequences (world behavior, morphs) must be observable in `getState()`.
- `getState()` — current serializable world/game state.
- `save()` — persist via `ctx.gameState` + the GameStateStore; returns path.
- `load()` — restore from the GameStateStore.

Rules:
- Never copy birth data; read the chart from `ctx.seed` (hydrated from the
  canonical BirthDataStore).
- Never mutate canonical identity; morph through `morphEngine.morph()`.
- Use the shared `iching/ichingResolver.js` for I Ching logic — no per-game
  hardcoded hexagram rules.

Register with `adapter.registerWorld(name, module)`, then
`enterWorld(playerId, name)` / `handleChoice(playerId, c)` / `exitWorld(playerId)`.
