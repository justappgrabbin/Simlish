# primitive-functions.build.mjs line 195 regex fix (2026-10-19 PT)

Bug: `f.origin.replace(/\\|/g,'/')` — the pattern means 'a backslash OR nothing', so it matched the empty string between every character and wrote '/' between all letters of the origin column in PRIMITIVE-FUNCTIONS.md (e.g. `/r/e/p/o/`). Intended: escape '|' for the markdown table, i.e. `/\|/g`.

Regenerated with `node integrated/primitive-functions.build.mjs`:
- primitive-functions.json, primitive-attributes.json, primitive-sources.json: byte-identical to before (cmp).
- PRIMITIVE-FUNCTIONS.md: one hunk, lines 104-162 (the 59 rows of the Function registry 'origin' column), nothing else.

```diff
104,162c104,162
< | gateToMacro | EXISTS | state-space.mjs:34 | /r/e/p/o/ /(/b/u/i/l/t/ /f/r/o/m/ /h/e/r/ /N/S/N/F/ /s/p/e/c/ /§/2/./1/;/ /K/i/n/g/ /W/e/n/</-/>/F/u/ /X/i/ /d/a/t/a/ /v/e/n/d/o/r///k/i/n/g/w/e/n/./m/j/s/ /f/r/o/m/ /h/e/r/ /z/i/p/s/)/ |
< | macroToGate | EXISTS | state-space.mjs:35 | /r/e/p/o/ |
< | bitsOf | EXISTS | state-space.mjs:22 | /r/e/p/o/ |
< | lineOf | EXISTS | state-space.mjs:25 | /r/e/p/o/ |
< | flipLines | EXISTS | state-space.mjs:27 | /r/e/p/o/ |
...
```
