# NexTone engine ↔ UI protocol (v1)

The engine runs on its own and the UI (and the CLI) connect to it over a local WebSocket.

## Discovery
On start the engine writes `engine.json`:

| Platform | Location |
|---|---|
| Windows | `%LOCALAPPDATA%\NexTone\engine.json` |
| Linux | `$XDG_RUNTIME_DIR/nextone/engine.json` (fallback `~/.local/state/nextone/engine.json`) |
| macOS | `~/Library/Application Support/NexTone/engine.json` |

```json
{ "port": 49873, "token": "hex32", "pid": 1234, "version": "0.1.0", "protocol": 1 }
```

Connect to `ws://127.0.0.1:<port>/?token=<token>`. The engine listens on loopback only, and a wrong token is rejected with HTTP 403.

## Text frames: JSON-RPC-like
- Request: `{"id": 7, "method": "route.connect", "params": {...}}`
- Response: `{"id": 7, "result": ...}` or `{"id": 7, "error": {"code": -32602, "message": "..."}}`
- Event: `{"event": "state", "data": EngineState}`. A full snapshot is sent on connect and after every change, coalesced to at most 20 per second.
- Event: `{"event": "log", "data": {"level": "info", "message": "..."}}`

## Binary frames: meters (≈30 Hz)
All fields are little-endian:

```
u32 magic = 0x314D544E ("NTM1")
u32 layoutVersion        // must equal EngineState.layoutVersion; otherwise ignore frame
u32 floatCount
f32 values[floatCount]
```

Each node has `meter: {offset, channels}`. Values at `offset + 2*c` are peak and `offset + 2*c + 1` are RMS, linear 0..∞ (1.0 = 0 dBFS), for channel `c`.

## Methods
See `ui/src/shared/protocol.ts` (the source of truth for types) and `engine/src/ipc/rpc_methods.cpp`.

| Method | Params | Result |
|---|---|---|
| `engine.getState` | – | `EngineState` |
| `engine.setSettings` | `Partial<EngineSettings>` | `EngineSettings` |
| `engine.restart` | – | – |
| `engine.shutdown` | – | – |
| `devices.refresh` | – | – |
| `node.update` | `{id, gainDb?, muted?, solo?, enabled?, name?, settings?}` | – |
| `node.openControlPanel` | `{id}` | – |
| `route.connect` | `{from, to, mode?: "auto" \| "one-to-one" \| "mono-sum" \| "spread"}` | `RouteInfo` |
| `route.disconnect` | `{id}` | – |
| `route.update` | `{id, gainDb?, muted?, enabled?}` | – |
| `route.setCrosspoint` | `{id, fromCh, toCh, gainDb: number \| null}` | – |
| `route.applyPreset` | `{id, mode: "one-to-one" \| "mono-sum" \| "spread" \| "clear"}` | – |
| `bus.create` | `{name, channels}` | `NodeInfo` |
| `bus.delete` | `{id}` | – |
| `fx.add` | `{nodeId, type, pluginId?, index?}` (`type: "vst3"` needs a `pluginId` from `EngineState.plugins`) | `FxInfo` |
| `fx.remove` | `{nodeId, fxId}` | – |
| `fx.update` | `{nodeId, fxId, bypass?, params?: {[k]: number}}` | – |
| `fx.move` | `{nodeId, fxId, index}` | – |
| `fx.openEditor` | `{nodeId, fxId}`: opens (or raises) the plugin's own window, owned by the engine | – |
| `plugins.scan` | – (starts a background scan; `EngineState.pluginsScanning` is true until it ends) | – |
| `app.setMode` | `{exe, mode: "off" \| "copy" \| "isolate"}` | – |
| `app.setRule` | `{exe, inputFrom?: nodeId \| null, autoCapture?: boolean}` | – |
| `vban.add` | `VbanStreamConfig` | `NodeInfo` |
| `vban.remove` | `{id}` | – |
| `scene.save` | `{name, id?}` | `SceneInfo` |
| `scene.load` | `{id}` | – |
| `scene.delete` | `{id}` | – |
| `recorder.start` | `{nodeIds, folder?, format: "wav" \| "flac"}` | – |
| `recorder.stop` | – | `{files: string[]}` |
| `recorder.saveReplay` | `{seconds}` | `{file}` |
| `tone.set` | `{enabled, freq, levelDb, target?: nodeId}` | – |
| `latency.measure` | `{outId, inId}` | `{ms}` |

## Node id scheme (stable across restarts)

| Kind | Id |
|---|---|
| WASAPI endpoint | `wasapi:in:<endpointId>` / `wasapi:out:<endpointId>` |
| ASIO device | `asio:in:<driverName>` / `asio:out:<driverName>` |
| NexTone virtual device | `vad:in:<n>` / `vad:out:<n>` |
| Compatibility cable | `compat:in:<endpointId>` / `compat:out:<endpointId>` |
| Application | `app:out:<exe>` / `app:in:<exe>` |
| NexTone ASIO client | `asioclient:out:<exe>` / `asioclient:in:<exe>` |
| Bus | `bus:<uuid>` |
| VBAN | `vban:in:<uuid>` / `vban:out:<uuid>` |
| Built-in | `tone` |

## VST3 plugins

- **Scanning.** Plugins come from the standard VST3 folders plus `EngineSettings.pluginFolders`
  (and the `NEXTONE_VST3_PATH` environment variable). Each module is loaded by
  `nextone-plugscan` in its own process, and the results are cached in `plugins.json` (data
  directory), keyed by modification time.
- **Plugin effects.** `FxInfo.type` is `"vst3"` and `FxInfo.pluginId` is the VST3 class id.
  Plugin parameters are edited in the plugin window, so `params` is empty.
- **Saved state.** It is stored with the configuration and scenes (`state`, base64).
- **Problems.** `FxInfo.error` is set when the plugin is not installed on this PC (settings are
  kept) or stopped after an internal error (the effect is switched off, so the engine keeps running).