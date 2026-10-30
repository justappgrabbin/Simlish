# SWARM OS Computer tray icons

Source: https://github.com/homarr-labs/dashboard-icons (svg/), copied into `icons/dashboard/`. Bundled in the app, no CDN. Each app keeps its emoji `icon` as fallback if the file fails to load.

License: Apache-2.0 (copy in `icons/dashboard/LICENSE-dashboard-icons.txt`). Their README: product names and logos belong to their owners, used for identification only; the license does not waive trademark/brand-guideline restrictions.

| App | Icon file | Note |
|---|---|---|
| paper (Paper) | papermark.svg | document/site studio, closest match |
| youniverse (YOU-N-I-VERSE) | google-earth.svg | world; Google brand logo |
| synthia-lab (Synthia Lab) | ollama.svg | local AI lab; Ollama brand logo |
| synthia (Synthia v14) | sparkleshare.svg | sparkle stand-in |
| human-design (Human Design) | qnap-qvr-human.svg | human figure; QNAP product logo |
| synthworld (SYNTHWORLD) | godot.svg | real match: it is a Godot game |
| prism-keys (Prism Keys) | monkeytype.svg | keyboard/typing |
| synthia-resonance (Synthia Resonance) | music.svg | music |
| trident (Trident) | android.svg | no trident icon; generic Android |
| stellar (Stellar Proximology) | starlight.svg | star |
| any added web app | generic-web.svg (box.svg) | fallback |
| any added native APK | generic-native.svg (android.svg) | fallback |

Flag: only Godot is a true match; the rest are other products' logos used as stand-ins. Easy to swap: change `iconFile` in apps/apps.src.json.
