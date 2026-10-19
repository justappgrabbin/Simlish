// Browser entry point: bundles the real, verified ato-core engine + shell-bridge
// into one global for a classic <script> to consume. Matches the already-proven
// "single self-contained HTML file, esbuild IIFE, no sibling files" delivery
// pattern (see project notes on the prior browser-entry.mjs/ato-bundle.js fix --
// a multi-file delivery previously failed silently on mobile file sharing).
import { bootstrapATO } from './index.mjs';
import { ShellBridge, createShellBridge, matchGates } from './shell-bridge.mjs';

export { bootstrapATO, ShellBridge, createShellBridge, matchGates };
