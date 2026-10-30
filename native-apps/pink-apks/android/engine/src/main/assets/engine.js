var process=globalThis.process||{argv:[],env:{}};if(typeof URL==="undefined"){globalThis.URL=class{constructor(p,b){this.href=String(b||"")+"#"+p}toString(){return this.href}}}
var Engine = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __esm = (fn, res) => function __init() {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  };
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod2) => __copyProps(__defProp({}, "__esModule", { value: true }), mod2);

  // shim/fs.mjs
  function readFileSync() {
    return globalThis.__STATE_SPACE_JSON;
  }
  function existsSync() {
    return false;
  }
  function writeFileSync() {
    throw new Error("no fs on device");
  }
  var fs_default;
  var init_fs = __esm({
    "shim/fs.mjs"() {
      fs_default = { readFileSync, existsSync, writeFileSync };
    }
  });

  // entry.mjs
  var entry_exports = {};
  __export(entry_exports, {
    FIELDS: () => FIELDS,
    canWrite: () => canWrite,
    format13: () => format13,
    gate13: () => gate13,
    intake: () => intake,
    meshCycle: () => meshCycle,
    parse13: () => parse13,
    resetSession: () => resetSession,
    safe: () => safe
  });

  // ../../drive-builds/mesh-automaton-extracted/address/resolver.mjs
  init_fs();

  // ../../drive-builds/mesh-automaton-extracted/address/vendor/vendor/kingwen.mjs
  var TABLE = Object.freeze([
    63,
    0,
    17,
    34,
    23,
    58,
    2,
    16,
    55,
    59,
    7,
    56,
    61,
    47,
    4,
    8,
    25,
    38,
    3,
    48,
    41,
    37,
    32,
    1,
    57,
    39,
    33,
    30,
    18,
    45,
    28,
    14,
    60,
    15,
    40,
    5,
    53,
    43,
    20,
    10,
    35,
    49,
    31,
    62,
    24,
    6,
    26,
    22,
    29,
    46,
    9,
    36,
    52,
    11,
    13,
    44,
    54,
    27,
    50,
    19,
    51,
    12,
    21,
    42
  ]);
  var KING_WEN_TO_FUXI_DECIMAL = Object.freeze(
    Object.fromEntries(TABLE.map((decimal, index) => [index + 1, decimal]))
  );
  var FUXI_DECIMAL_TO_KING_WEN = Object.freeze(
    Object.fromEntries(TABLE.map((decimal, index) => [decimal, index + 1]))
  );
  var KING_WEN_HEXAGRAM_NAMES = Object.freeze([
    "The Creative",
    "The Receptive",
    "Difficulty at the Beginning",
    "Youthful Folly",
    "Waiting (Nourishment)",
    "Conflict",
    "The Army",
    "Holding Together (Union)",
    "The Taming Power of the Small",
    "Treading (Conduct)",
    "Peace",
    "Standstill (Stagnation)",
    "Fellowship with Men",
    "Possession in Great Measure",
    "Modesty",
    "Enthusiasm",
    "Following",
    "Work on What Has Been Spoiled (Decay)",
    "Approach",
    "Contemplation (View)",
    "Biting Through",
    "Grace",
    "Splitting Apart",
    "Return (The Turning Point)",
    "Innocence (The Unexpected)",
    "The Taming Power of the Great",
    "The Corners of the Mouth (Providing Nourishment)",
    "Preponderance of the Great",
    "The Abysmal (Water)",
    "The Clinging (Fire)",
    "Influence (Wooing)",
    "Duration",
    "Retreat",
    "The Power of the Great",
    "Progress",
    "Darkening of the Light",
    "The Family (The Clan)",
    "Opposition",
    "Obstruction",
    "Deliverance",
    "Decrease",
    "Increase",
    "Break-through (Resoluteness)",
    "Coming to Meet",
    "Gathering Together (Massing)",
    "Pushing Upward",
    "Oppression (Exhaustion)",
    "The Well",
    "Revolution (Molting)",
    "The Caldron",
    "The Arousing (Shock, Thunder)",
    "Keeping Still (Mountain)",
    "Development (Gradual Progress)",
    "The Marrying Maiden",
    "Abundance (Fullness)",
    "The Wanderer",
    "The Gentle (The Penetrating, Wind)",
    "The Joyous (Lake)",
    "Dispersion (Dissolution)",
    "Limitation",
    "Inner Truth",
    "Preponderance of the Small",
    "After Completion",
    "Before Completion"
  ]);
  function gateToFuXiDecimal(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) {
      throw new RangeError("gate must be 1..64.");
    }
    return TABLE[gate2 - 1];
  }
  function fuXiDecimalToGate(decimal) {
    const gate2 = FUXI_DECIMAL_TO_KING_WEN[decimal];
    if (!gate2) {
      throw new RangeError(`No King Wen gate for Fu-Xi decimal ${decimal}.`);
    }
    return gate2;
  }
  function hexagramName(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) {
      throw new RangeError("gate must be 1..64.");
    }
    return KING_WEN_HEXAGRAM_NAMES[gate2 - 1];
  }
  function gatePattern(gate2) {
    const decimal = gateToFuXiDecimal(gate2);
    return Object.freeze(Array.from({ length: 6 }, (_, i) => decimal >> i & 1));
  }
  var kingwen_default = Object.freeze({
    KING_WEN_TO_FUXI_DECIMAL,
    FUXI_DECIMAL_TO_KING_WEN,
    KING_WEN_HEXAGRAM_NAMES,
    gateToFuXiDecimal,
    fuXiDecimalToGate,
    hexagramName,
    gatePattern
  });

  // ../../drive-builds/mesh-automaton-extracted/address/vendor/state-space.mjs
  var INVESTIGATING = Object.freeze([
    "split factor: Alexis said 12 and also six; both are accepted, 12 is the default only because she said it first (Spec \xA7IV.D flag).",
    'number of nested levels: Spec \xA76 says "five levels" are tiers whose semantics are not fixed; depth 5 is configurable and is not tied to the five dimensions or centers.',
    "Space readout: no emergence rule is given (Spec \xA770 says the order of differentiation and integration MUST NOT be silently decided). Default is no value, reported as investigating; a rule can be injected.",
    "center_binding is null: nine centers must emerge (Spec \xA719); no gate-to-center table is stored and no modulo-nine mapping is used.",
    'role of the 64-bit mask (Spec \xA73 "pending design decision"): here it only marks which level-1 slots are active in the current event.',
    "gate numbering: the repo kernel treats address.gate as King Wen numbering; whether HD gate numbers are identical is not settled (Spec \xA736)."
  ]);
  var DIMENSIONS = Object.freeze(["Movement", "Evolution", "Being", "Design", "Space"]);
  var PRIMITIVE_DIMENSIONS = Object.freeze(["Movement", "Evolution", "Being", "Design"]);
  var SPLIT_CHOICES = Object.freeze([12, 6]);
  function checkCode(c) {
    if (!Number.isInteger(c) || c < 0 || c > 63) throw new RangeError("Hexagram code is 0..63");
    return c;
  }
  var complement = (code2) => checkCode(code2) ^ 63;
  var reverseLines = (code2) => {
    checkCode(code2);
    let r = 0;
    for (let i = 0; i < 6; i++) if (code2 >> i & 1) r |= 1 << 5 - i;
    return r;
  };
  var gateToMacro = (gate2) => {
    const m = KING_WEN_TO_FUXI_DECIMAL[gate2];
    if (m === void 0) throw new RangeError("Gate is 1..64");
    return m;
  };

  // ../../drive-builds/mesh-automaton-extracted/address/vendor/dodeca-face-axes.mjs
  var P = (1 + Math.sqrt(5)) / 2;
  var raw = [];
  for (const s1 of [1, -1]) for (const s2 of [1, -1]) {
    raw.push([0, s1, s2 * P], [s1, s2 * P, 0], [s2 * P, 0, s1]);
  }
  var dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  var FACES = raw;
  var AXES = (() => {
    const used = /* @__PURE__ */ new Set(), ax = [];
    FACES.forEach((f, i) => {
      if (used.has(i)) return;
      const j = FACES.findIndex((g) => g.every((x, k) => Math.abs(x + f[k]) < 1e-9));
      used.add(i);
      used.add(j);
      ax.push([i, j]);
    });
    return ax;
  })();
  var n2 = dot(FACES[0], FACES[0]);
  var adjacent = (i, j) => i !== j && Math.abs(dot(FACES[i], FACES[j]) / n2 - 1 / Math.sqrt(5)) < 1e-9;
  function litFaces(code2, order = [0, 1, 2, 3, 4, 5]) {
    return order.map((ax, line) => AXES[ax][code2 >> line & 1]);
  }
  function connected(fs) {
    const s = new Set(fs), seen = /* @__PURE__ */ new Set([fs[0]]), q = [fs[0]];
    while (q.length) {
      const x = q.pop();
      for (const y of s) if (!seen.has(y) && adjacent(x, y)) {
        seen.add(y);
        q.push(y);
      }
    }
    return seen.size === s.size;
  }
  var shape = (fs) => {
    const deg = fs.map((x) => fs.filter((y) => adjacent(x, y)).length).sort();
    const edges = deg.reduce((a, b) => a + b, 0) / 2;
    return `${edges} edges, degrees ${deg.join("")}`;
  };
  function analyse(order = [0, 1, 2, 3, 4, 5]) {
    const out = { sets: 64, complementIsAntipode: 0, connected: 0, caps: 0, shapes: {}, complementSameShape: 0 };
    for (let c = 0; c < 64; c++) {
      const L = litFaces(c, order), C = litFaces(complement(c), order);
      if (L.every((f, i) => AXES[order[i]].includes(f) && C[i] !== f)) out.complementIsAntipode++;
      if (connected(L)) out.connected++;
      if (L.some((f) => L.filter((g) => adjacent(f, g)).length === 5)) out.caps++;
      const sh = shape(L);
      out.shapes[sh] = (out.shapes[sh] || 0) + 1;
      if (shape(C) === sh) out.complementSameShape++;
    }
    return out;
  }
  function kingWenPairs() {
    let inv = 0, cmp = 0, both = 0, none = 0;
    const cmpPairs = [];
    for (let k = 1; k <= 63; k += 2) {
      const a = gateToMacro(k), b = gateToMacro(k + 1);
      const i = reverseLines(a) === b, c = complement(a) === b;
      if (i && c) both++;
      else if (i) inv++;
      else if (c) {
        cmp++;
        cmpPairs.push([k, k + 1]);
      } else none++;
    }
    return { inversionOnly: inv, complementOnly: cmp, complementPairs: cmpPairs, both, none };
  }
  function axisAngles() {
    const s = /* @__PURE__ */ new Set();
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) {
      const a = FACES[AXES[i][0]], b = FACES[AXES[j][0]];
      s.add((Math.acos(Math.abs(dot(a, b)) / n2) * 180 / Math.PI).toFixed(4));
    }
    return [...s];
  }
  if ("file:///eng/x/y.mjs" === `file://${process.argv[1]}`) {
    console.log(AXES.length, "axes");
    console.log(analyse());
    console.log(kingWenPairs());
    console.log("axis-axis angles", axisAngles());
    const perms = [];
    const p = (a, r) => {
      if (!a.length) {
        perms.push(r);
        return;
      }
      a.forEach((x, i) => p(a.filter((_, k) => k !== i), [...r, x]));
    };
    p([0, 1, 2, 3, 4, 5], []);
    const sig = new Set(perms.map((o) => {
      const a = analyse(o);
      a.shapes = Object.entries(a.shapes).sort();
      return JSON.stringify(a);
    }));
    console.log("distinct analyses over all 720 orders:", sig.size);
  }

  // ../../drive-builds/mesh-automaton-extracted/address/vendor/axis-order.mjs
  function buildOrder(centreLine = 5, centreFace = 0) {
    if (!Number.isInteger(centreLine) || centreLine < 1 || centreLine > 6) throw new RangeError("centreLine 1..6");
    const nb = FACES.map((_, i) => i).filter((i) => adjacent(centreFace, i));
    const ring = [nb[0]];
    while (ring.length < 5) {
      const last = ring.at(-1);
      ring.push(nb.find((f) => !ring.includes(f) && adjacent(last, f)));
    }
    const lines = [1, 2, 3, 4, 5, 6].filter((l) => l !== centreLine);
    const yangFace = { [centreLine]: centreFace };
    lines.forEach((l, i) => yangFace[l] = ring[i]);
    const axisOf = (f) => AXES.findIndex((a) => a.includes(f));
    return Object.freeze([1, 2, 3, 4, 5, 6].map((l) => {
      const f = yangFace[l], ax = axisOf(f);
      return Object.freeze({ line: l, axis: ax, yangFace: f, yinFace: AXES[ax].find((g) => g !== f) });
    }));
  }
  var ORDER = buildOrder();
  var facesForHexagram = (code2, order = ORDER) => order.map((o, i) => code2 >> i & 1 ? o.yangFace : o.yinFace);

  // ../../drive-builds/mesh-automaton-extracted/address/vendor/letter-layers.mjs
  var NATO = {
    a: "Alfa",
    b: "Bravo",
    c: "Charlie",
    d: "Delta",
    e: "Echo",
    f: "Foxtrot",
    g: "Golf",
    h: "Hotel",
    i: "India",
    j: "Juliett",
    k: "Kilo",
    l: "Lima",
    m: "Mike",
    n: "November",
    o: "Oscar",
    p: "Papa",
    q: "Quebec",
    r: "Romeo",
    s: "Sierra",
    t: "Tango",
    u: "Uniform",
    v: "Victor",
    w: "Whiskey",
    x: "X-ray",
    y: "Yankee",
    z: "Zulu"
  };
  var NATO_ALT = { a: "Alpha", j: "Juliet" };
  var RUSSIAN = [
    ["\u0410\u0430", "a"],
    ["\u0411\u0431", "b"],
    ["\u0412\u0432", "v"],
    ["\u0413\u0433", "g"],
    ["\u0414\u0434", "d"],
    ["\u0415\u0435", "e, ye"],
    ["\u0401\u0451", "e, ye"],
    ["\u0416\u0436", "zh"],
    ["\u0417\u0437", "z"],
    ["\u0418\u0438", "i"],
    ["\u0419\u0439", "y"],
    ["\u041A\u043A", "k"],
    ["\u041B\u043B", "l"],
    ["\u041C\u043C", "m"],
    ["\u041D\u043D", "n"],
    ["\u041E\u043E", "o"],
    ["\u041F\u043F", "p"],
    ["\u0420\u0440", "r"],
    ["\u0421\u0441", "s"],
    ["\u0422\u0442", "t"],
    ["\u0423\u0443", "u"],
    ["\u0424\u0444", "f"],
    ["\u0425\u0445", "kh"],
    ["\u0426\u0446", "ts"],
    ["\u0427\u0447", "ch"],
    ["\u0428\u0448", "sh"],
    ["\u0429\u0449", "shch"],
    ["\u044A", "\u201C"],
    ["\u042B\u044B", "y"],
    ["\u044C", "\u2018"],
    ["\u042D\u044D", "e"],
    ["\u042E\u044E", "yu"],
    ["\u042F\u044F", "ya"]
  ];
  function natoOf(ch) {
    const k = ch.toLowerCase();
    return NATO[k] ? { word: NATO[k], alt: NATO_ALT[k] || null } : null;
  }
  function translitOf(ch) {
    const k = ch.toLowerCase();
    const ru = RUSSIAN.filter((r) => r[1].split(/,\s*/).indexOf(k) > -1).map((r) => ({ cyr: r[0], printed: r[1] }));
    return { russian: ru.length ? ru : null, greek: null };
  }
  var HER_WORDS_DERIVE = {
    msg14: "Listen you need to do the work to figure out what the words are taking my examples As guiding context.",
    msg15: "But start with all the words that you are given then use the black book to derive each gate and it's lines"
  };
  Object.assign(HER_WORDS_DERIVE, {
    msg16: "Matter is matter it is just leave it as it is matter is as matter does whatever it's just matter is that's why because being is matter duh lol",
    msg17: "I have no idea what your question was about the smelling you can't do that but you can look in colors the book of colors"
  });
  var BANTU_BOOKS = {
    thesis: { name: "Katushemererwe, Computational morphology and Bantu language learning (Groningen 2013)", printedFromPdf: (p) => p - 11 },
    zeller: { name: "Zeller, Locative inversion in Bantu and predication, Linguistics 51(6) 2013", printedFromPdf: (p) => p + 1106 },
    diercks: { name: "Diercks, Information structure is syntactic: Evidence from Bantu languages (2022)", printedFromPdf: (p) => p },
    creissels: { name: "Creissels, Bantu languages: Typology and variation (3rd draft, 2019)", printedFromPdf: (p) => p }
  };
  var BF = (id2, dims, book, pdf, excerpt, tie, status) => ({ id: id2, dims, book, pdfPage: pdf, printedPage: BANTU_BOOKS[book].printedFromPdf(pdf), excerpt, tie, status: status || "investigating (my reading of the tie)" });
  var BANTU_FACTS = [
    BF("vowel-a", ["evolution"], "thesis", 57, "If the noun class prefix has the vowel a (e.g. ba, ma), the initial vowel will be a, thus amata \u2018milk\u2019 abakazi \u2018women\u2019.", "text gives prefix vowel A \u2192 initial vowel a; her vowel A is Evolution (a rule I am only putting side by side)"),
    BF("vowel-i", ["movement"], "thesis", 57, "When the noun prefix has i or -, the initial vowel is e, for example ekitookye, emiti, etc.", "text gives prefix vowel I \u2192 initial vowel e; her vowel I is Movement (side by side only; the OCR prints \u201Ci or -\u201D)"),
    BF("vowel-u", ["design"], "thesis", 57, "The initial vowel is o when the noun class prefix has u, omuntu \u2018person\u2019, omuti \u2018tree\u2019.", "text gives prefix vowel U \u2192 initial vowel o; her vowel U is Design (side by side only)"),
    BF("seeing", ["movement"], "thesis", 72, "\u2018to see\u2019 ... Runyankore-Rukiga kureeba ... Runyoro-Rutooro kurora", "her Movement chain has Seeing; the text gives the verb \u2018to see\u2019 in the two Runyakitara forms (word match only)", "word match; tie to Movement is hers, the text says nothing about it"),
    BF("seeing-redup", ["movement"], "thesis", 116, "kureeba \u2018to see\u2019   kureeba-reeba \u2018seeing (extremely)\u2019", "the text itself glosses \u201Cseeing\u201D, by reduplication of kureeba", "word match"),
    BF("a-movement", ["movement"], "zeller", 1, "locative inversion in Bantu languages is typically analyzed in terms of A-movement of the locative from a VP-internal position to the subject position", "the word Movement is a syntax term here (a different sense from her dimension)", "same word, different sense: investigating"),
    BF("direction-source", ["movement", "space"], "creissels", 10, "locative marking (i.e. the morphological characteristics of phrases specifying the location of an event, or the direction or source of movement with movement verbs) is fully integrated into the noun class system", "the text puts movement (direction, source) and location (space) in one noun-class system", "text states it; tie to her dimensions is my reading"),
    BF("love", ["evolution"], "thesis", 72, "\u2018to love\u2019 ... Runyankore-Rukiga kukunda ... Runyoro-Rutooro kugonza", "her Evolution chain has Love; the text gives \u2018to love\u2019 in the two Runyakitara forms (word match only)", "word match"),
    BF("evolved", ["evolution"], "diercks", 29, "ComP is an innovation specific to the Bantu language family, where focus/givenness evolved to be centrally related to middlefield inflectional properties of the clause", "the text uses the word \u201Cevolved\u201D of a Bantu grammatical feature", "word match; thematic only"),
    BF("body-parts", ["being"], "thesis", 58, "5/6  e-ri-  a-ma-  Some parts of the body  e-ri-isho/a-ma-isho  Eye(s)", "noun-class table row whose semantics is body parts; her Being is the Body (Black Book)", "table row transcribed from the layout text"),
    BF("body-parts-15", ["being"], "thesis", 59, "15/6  o-ku-  a-ma-  Some body parts  o-ku-guru/amaguru  Leg(s)", "second body-part class in the same table", "table row transcribed"),
    BF("mass", ["being"], "thesis", 59, "6a  a-ma-te  Milk  (Mass nouns);  8  e-bi-bembe  Leprosy  (Mass nouns)", "the table calls these classes Mass nouns; her Being chain has Matter (word-level resemblance only)", "investigating; the text does not link mass nouns to Being"),
    BF("touch", ["being"], "thesis", 27, "kukwata \u2018to catch\u2019  kukwata-kwata \u2018to touch \u201Cin a funny way\u201D\u2019 ... kwatisa \u2018cause \u2026 to touch\u2019", "her Being chain has Touch; the text gives the Runyakitara verb forms for \u2018to touch\u2019", "word match"),
    BF("template", ["design"], "creissels", 6, "the structure of Tswana S31 verb forms can be accounted for by a template ... a sequence of positions numbered from \u20134 (the leftmost possible position) to +5 (the rightmost possible position)", "her Design chain has Structure; the text describes verb structure as numbered positions", "word match (structure); tie is thematic"),
    BF("concord", ["design"], "thesis", 57, "the noun class prefix of a noun agrees with all the constituents of a noun phrase, such as adjectives, pronouns and numerals", "the text describes class structure as agreement (concord) running through the phrase", "thematic; no word match"),
    BF("loc-table", ["space"], "thesis", 59, "16  aha-  Location  aha-kaanyima  Behind the house;  17  oku-  Location  oku-zimu  Underground;  18  omu-  Location  omu-nda  In the stomach", "the three locative classes in the Runyakitara table; her Space is place/Presence in the Black Book", "table row transcribed; tie is my reading"),
    BF("loc-zeller", ["space"], "zeller", 29, "a noun class prefix from the locative noun classes 16, 17, or 18 ... K\xFA-m\xFA-saala kw-\xE1-kw\xE1 mu-mu-siiru. ... \u2018A tree fell in the forest.\u2019", "the locative classes again, with a worked sentence", "text states it; tie is my reading"),
    BF("loc-place", ["space"], "creissels", 10, "The languages in question typically have three locative classes ... yields the hypernymic term \u2018place\u2019", "the text says the locative prefixes give the general word \u2018place\u2019", "text states it; tie is my reading")
  ];
  var DOC1_LEVELS = [
    { n: 1, name: "Waveform Substrate", gov: "Possibility, potential, probability", corr: "Quantum fields, Zero-point energy, Human Design \u201Cmonopole\u201D field", dims: ["movement"], why: "text says \u201Cmonopole\u201D; the Black Book gives the Magnetic Monopole to Movement" },
    { n: 3, name: "Collapsing Observer", gov: "Choice, awareness, attention", corr: "Consciousness, Variable Arrows, View/Motivation, the \u201CI\u201D; \u201CPersonality layer\u201D", dims: ["space"], why: "text says \u201CPersonality layer\u201D" },
    { n: 6, name: "Embodiment Container", gov: "Physicalization of the waveform \u2192 form, restriction, location", corr: "Design crystal, personality vehicle, simulation rules, Design layer", dims: ["design"], why: "text says \u201CDesign crystal\u201D and \u201CDesign layer\u201D" },
    { n: 7, name: "Feedback Loops (FEEDBACK & EVOLUTION)", gov: "Correction, evolution, learning, reflection", corr: "Emotions, coherence metrics, shadow integration, reward functions", dims: ["evolution"], why: "heading says \u201CEvolution\u201D" }
  ].map((x) => Object.assign({ cite: "doc1 lines 293-300 and 308-417" }, x));
  var DOC_CONFLICT_LIST = [
    { id: "smell", topic: "Progress is Smell / Smelt", text: "doc1 Tab 1 and Tab 2 both print \u201CProgress is Smell\u201D then \u201CSmelling is Life\u201D (lines 29-30, 10422-10423). Her earlier docs and the Black Book print \u201CProgress is Smelt\u201D. doc1 line 1806 itself pastes \u201Cprogress is Smelt\u201D and line 679 the OCR \u201CPrograss la Jamalt\u201D. Both kept; no ranking.", dims: ["design"] },
    { id: "tab2", topic: "doc1 Tab 1 vs Tab 2", text: "The two tables are the same except Tab 2 reads \u201CSueeing is Landscape\u201D (typo, line 10409) and Tab 2 stops before the Space rows (the export ends at \u201C| **Spa\u201D, line 10424). Tab 1 has all five dimensions.", dims: ["movement", "space"] },
    { id: "names", topic: "Five dimension names", text: "doc1 line 784 names the five as Spirit / Mind / Soul / Body / Heart (keynotes I Create, I Remember, I Design, I Am, I Think / I Communicate). Everywhere else in doc1 and in the Black Book they are Movement / Evolution / Being / Design / Space. Pairing by keynote: Spirit=Movement, Mind=Evolution, Soul=Design, Body=Being, Heart=Space. Not resolved.", dims: DIM_ORDER_C() },
    { id: "keynote-m", topic: "Movement keynote", text: "doc1 table and Section 3 say \u201CI Define\u201D; doc1 line 784 and Black Book p5 say \u201CI Create\u201D. Earlier ruling stands (I Define, majority of printed instances); both shown.", dims: ["movement"] },
    { id: "keynote-s", topic: "Space keynote", text: "doc1 table and Section 3, and Black Book p115, say \u201CI Think\u201D; doc1 line 784 and Black Book p5 say \u201CI Communicate\u201D. Both shown.", dims: ["space"] },
    { id: "centers", topic: "Centers per dimension", text: "doc1 line 5729 maps Movement to G/Sacral/Root, Evolution to Head/Ajna, Being to Solar Plexus/Spleen, Design to Throat/Heart, Space to all centers. The Black Book prints one crystal and center per dimension (Movement G, Evolution Head, Being Ajna with the Atom, Design Ajna, Space Head). They differ for Being (Ajna vs Solar Plexus/Spleen) and Design (Ajna vs Throat/Heart). Not resolved.", dims: DIM_ORDER_C() },
    { id: "plane-names", topic: "Mind, Body, Heart", text: "doc1 uses Mind and Body as dimension names (Evolution, Being) and Heart as the fifth (Space). doc2 uses Mind / Body / Heart as three field planes (alpha electromagnetic-sidereal, beta gravitational-tropical, gamma draconic). doc2 never links planes to the five dimensions. And doc1 line 784 gives the Mind dimension \u201CGravity, Entropy, Memory\u201D while doc2 gives gravity to the Body plane and electromagnetism to the Mind plane. Not resolved.", dims: ["evolution", "being", "space"] },
    { id: "axis", topic: "Individuality diagonal vs vertical", text: "Her earlier words say Individuality is a diagonal process; doc3 \xA7F puts Individuality (D1 Movement) on the Vertical axis and gives the Diagonal axis to Base 4 (Design). Both kept; the earlier ruling that columns are written vertically is unchanged.", dims: ["movement", "design"] },
    { id: "tone-senses", topic: "Senses on the tones", text: "The book of colors p21/p40 tone table runs Smell, Taste, Outer Vision, Inner Vision, Feeling, Touch (tones 1 to 6). doc3 \xA7I puts Hearing on Tone 6 for Movement, Evolution and Design, makes Tone 6 \u201CTouch/Hearing\u201D for Being and \u201CHearing/Music\u201D for Space. The book has no Hearing on its tone table (Hearing is only Base 5's sense on p40). Not resolved.", dims: DIM_ORDER_C() },
    { id: "tone-color", topic: "Tone 5 and colors", text: "doc3 \xA7J says Tone 5 is compatible with Colors 4 and 5. The book of colors groups tones 5 and 6 with colors 5 and 6 (Solar Plexus binary) and colors 3 and 4 with Ajna. Color 4 is Ajna in the book. Not resolved.", dims: DIM_ORDER_C() },
    { id: "base-sense", topic: "Base anchors vs book senses", text: "doc3 \xA7I resonant anchors: Movement Tone 3 (Outer Vision), Evolution Tone 2 (Taste), Being Tone 6 (Touch/Hearing), Design Tone 1 (Smell), Space Tone 6 (Hearing/Music). Book of colors p40 base senses: Seeing, Taste, Touching, Smell, Hearing. These agree for Evolution, Design, Being (Touch), Space (Hearing); Movement agrees only loosely (Seeing vs Outer Vision).", dims: DIM_ORDER_C() },
    { id: "eleven", topic: "11 parameters vs 12 layers", text: "doc2 counts 11 parameters (Gate\u2026House). doc1 lines 4456 and 4694 add Dimension in front (12 layers). Not resolved.", dims: DIM_ORDER_C() },
    { id: "elements", topic: "Design and Space trigrams", text: "doc1 line 5829 lists Lake and Fire for Design and Fire and Lake for Space (the same two trigrams). Not resolved.", dims: ["design", "space"] },
    { id: "gate20", topic: "Gate 20 and Presence", text: "doc3 \xA74 puts Gate 20 (Contemplation, \u201CPresence / Now\u201D) in Base 1 (Movement) with anchor Environment. The Gene Keys spectrum gives Gene Key 20 the Siddhi Presence, and the Black Book (pdf 115) says Space is experienced as Presence. Not resolved; no Space reading is made from it.", dims: ["movement", "space"] },
    { id: "space-eq", topic: "Space and the equations", text: "doc1 line 3 says Space does not belong to the fundamental equations and arises later as a condition produced by the interaction of the fields. The Black Book (pdf 118) says the same of the personality (\u201Chas no substance\u201D). Consistent; yet doc1 gives Space its own row, keynote and Base 5. Kept as printed.", dims: ["space"] }
  ];
  function DIM_ORDER_C() {
    return ["movement", "evolution", "being", "design", "space"];
  }

  // ../../drive-builds/mesh-automaton-extracted/address/resolver.mjs
  var S = JSON.parse(fs_default.readFileSync(new URL("./state-space.data.json", "file:///eng/x/y.mjs")));
  var SIGNS = S.meta.zodiacOrder;
  var SIZES = { gate: 5 + 37 / 60 + 30 / 3600, line: 56 / 60 + 15 / 3600, color: (9 + 22.5 / 60) / 60, tone: (1 + 33.75 / 60) / 60, base: 18.75 / 3600 };
  var COUNTS = { gate: 64, line: 6, color: 6, tone: 6, base: 5 };
  var signIdx = (s) => {
    if (!s) return -1;
    const k = s.trim().slice(0, 3).toLowerCase();
    return SIGNS.findIndex((z) => z.slice(0, 3).toLowerCase() === k);
  };
  var dms = (d, m, s) => +d + +m / 60 + +(s || 0) / 3600;
  function startOf(text, signCol) {
    const re = /(\d{1,2})°(\d{2})'(?:(\d{2}(?:\.\d+)?)")?\s*([A-Za-z]+)?/g;
    const p = [...text.matchAll(re)];
    if (!p.length) return null;
    const a = p[0], b = p[1];
    let si = signIdx(a[4]);
    if (si < 0 && b) {
      si = signIdx(b[4]);
      if (si >= 0 && dms(b[1], b[2], b[3]) < dms(a[1], a[2], a[3])) si = (si + 11) % 12;
    }
    if (si < 0) si = signIdx((signCol || "").split(/[\/ ]/)[0]);
    return si < 0 ? null : +(si * 30 + dms(a[1], a[2], a[3])).toFixed(6);
  }
  var POSITIONS = {};
  for (const g of Object.values(S.gates)) {
    const c = [];
    const unsigned = [];
    for (const x of g.claims) {
      const k = Object.keys(x).find((k2) => /Position/.test(k2)), pos = k && x[k];
      if (pos) {
        const v = startOf(pos, x["Sign"] || x["Astrological Sign"]);
        if (v != null) c.push({ id: x.id, start: v, text: pos });
        else {
          const m = pos.match(/(\d{1,2})°(\d{2})'(\d{2})?/);
          if (m) unsigned.push({ id: x.id, inSign: +dms(m[1], m[2], m[3]).toFixed(6), text: pos });
        }
      }
    }
    for (const n3 of g.notes) {
      const m = (n3.text || "").match(/Location on Mandala:\**\s*(.*)/);
      if (m) {
        const v = startOf(m[1]);
        if (v != null) c.push({ id: n3.id, start: v, text: m[1] });
      }
    }
    const uniq = [...new Set(c.map((x) => x.start))];
    const clash = uniq.length === 1 && unsigned.some((u) => Math.abs(u.inSign - uniq[0] % 30) > 1e-6);
    POSITIONS[g.gate] = { claims: c, unsignedClaims: unsigned, start: uniq.length === 1 && !clash ? uniq[0] : null, conflict: uniq.length > 1 || clash, candidates: uniq };
  }
  function parse(a) {
    if (typeof a === "string") {
      const p = a.split(".").map(Number);
      if (p.length !== 5) throw new RangeError("address is gate.line.color.tone.base");
      a = { gate: p[0], line: p[1], color: p[2], tone: p[3], base: p[4] };
    }
    if (a && a.degree != null) return fromDegree(a.degree);
    for (const k of Object.keys(COUNTS)) if (!Number.isInteger(a[k]) || a[k] < 1 || a[k] > COUNTS[k]) throw new RangeError(`${k} must be 1..${COUNTS[k]}`);
    return { gate: a.gate, line: a.line, color: a.color, tone: a.tone, base: a.base };
  }
  var code = (a) => `${a.gate}.${a.line}.${a.color}.${a.tone}.${a.base}`;
  var indexOf = (a) => ((((a.gate - 1) * 6 + a.line - 1) * 6 + a.color - 1) * 6 + a.tone - 1) * 5 + a.base - 1;
  function fromDegree(deg) {
    deg = +((deg % 360 + 360) % 360).toFixed(9);
    const hits = Object.entries(POSITIONS).filter(([, p2]) => p2.start != null && (deg - p2.start + 360) % 360 < SIZES.gate - 1e-12);
    if (hits.length !== 1) return { unresolved: true, degree: deg, candidates: hits.map((h) => +h[0]), reason: hits.length ? "more than one gate claims this arc" : "no agreed gate position covers this degree (see open questions)" };
    const [g, p] = hits[0];
    let r = (deg - p.start + 360) % 360;
    const out = { gate: +g };
    for (const k of ["line", "color", "tone", "base"]) {
      const i = Math.min(COUNTS[k] - 1, Math.floor(r / SIZES[k] + 1e-9));
      out[k] = i + 1;
      r -= i * SIZES[k];
    }
    return out;
  }
  var first = (arr, key) => arr.map((x) => x[key]).find(Boolean);
  function state(addr) {
    const a = parse(addr);
    if (a.unresolved) return a;
    const g = S.gates[a.gate], anchor = g.claims.filter((c) => c.kind === "gateAnchor");
    const toneRow = S.tones[a.tone].claims[0], baseRow = S.bases[a.base].claims[0];
    const btCell = baseRow[Object.keys(baseRow).find((k) => k.startsWith(`Tone ${a.tone} `))];
    const inList = (s) => new RegExp(`Color ${a.color}\\b`).test(s || "");
    const tc = inList(toneRow["Compatible Colors"]) ? "compatible" : inList(toneRow["Incompatible Combinations"]) ? "incompatible" : "not stated in doc";
    return {
      address: code(a),
      index: indexOf(a),
      ...a,
      gateState: { names: [...new Set(g.claims.map((c) => c["I Ching Name"] || c["Name"]).filter(Boolean))], archetype: first(anchor, "Primary Archetype"), baseAssociation: first(anchor, "Base Association"), entryAnchorNode: first(anchor, "Entry Anchor Node"), zodiac: [...new Set(g.claims.map((c) => c["Zodiac Sign"] || c["Sign"] || c["Astrological Sign"]).filter(Boolean))], house: first(anchor, "House"), centers: [...new Set(g.claims.map((c) => c["Biological Center"] || c["Center"]).filter(Boolean))], harmonic: [...new Set(g.claims.map((c) => c["Harmonic"] || c["Harmonic Gate"]).filter(Boolean))], mandalaStart: POSITIONS[a.gate].start, positionConflict: POSITIONS[a.gate].conflict, notes: g.notes.length, docIds: g.claims.map((c) => c.id) },
      lineState: { orbital: S.lines[a.line].claims.map((c) => ({ line: c["Line #"], orbital: c["Orbital Name"], angle: c["Traversal Angle"], effect: c["Mechanics & Graph Effect"] })), gateLine: g.lines[a.line].map((x) => x.text || x.Expression || x.cells) },
      colorState: { color: a.color, names: S.colors[a.color]?.names ?? [] },
      toneState: { tone: a.tone, name: S.tones[a.tone].name, binaryFraming: toneRow["Binary Operational Framing"], toneColor: tc },
      baseState: { base: a.base, name: S.bases[a.base].name, nonPrimitive: S.bases[a.base].nonPrimitive, baseToneCell: btCell },
      axisRows: S.axes.filter((r) => new RegExp(`Base ${a.base}\\b|D${a.base}\\b`).test(r["Primary Associated Bases"])).map((r) => ({ id: r.id, axis: r["Axis Type"], vector: r["Directional Vector"] })),
      ruleO: { baseTone: btCell && /Valid|Resonant/.test(btCell) ? "valid" : "halt", toneColor: tc, resonance: /Resonant Anchor/.test(btCell || "") ? "amplify 2x" : "standard emission" },
      encodings: encode(a)
    };
  }
  var label = (a) => {
    const st = S;
    return [st.gates[a.gate].claims.find((c) => c["I Ching Name"])?.["I Ching Name"], st.lines[a.line].claims[0]["Orbital Name"], st.colors[a.color]?.names[0], st.tones[a.tone].name, st.bases[a.base].name].join(" ");
  };
  function encode(addr) {
    const a = parse(addr), c = code(a), chars = [...c], words = label(a), wchars = [...words];
    return {
      text: c,
      words,
      B: { binary: chars.map((ch) => ch.codePointAt(0).toString(2).padStart(8, "0")).join(" "), components: Object.fromEntries(Object.keys(COUNTS).map((k) => [k, (a[k] - 1).toString(2).padStart(k === "gate" ? 6 : 3, "0")])), index: indexOf(a).toString(2).padStart(17, "0") },
      H: chars.map((ch) => `&#x${ch.codePointAt(0).toString(16).toUpperCase()};`).join(""),
      A: chars.map((ch) => ch.codePointAt(0)),
      N: wchars.map((ch) => natoOf(ch)?.word ?? (ch === " " ? "(space)" : null)),
      T: wchars.map((ch) => {
        const t = translitOf(ch);
        return { ch, russian: t?.russian?.map((x) => x.cyr) ?? null, greek: t?.greek ?? null };
      }),
      U: chars.map((ch) => "U+" + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")),
      G: { hexagramCode: gateToMacro(a.gate), dodecaFaces: facesForHexagram(gateToMacro(a.gate)), lineAxis: ORDER[a.line - 1] },
      Hx: { index: indexOf(a).toString(16).toUpperCase().padStart(5, "0"), bytes: chars.map((ch) => ch.codePointAt(0).toString(16).toUpperCase().padStart(2, "0")).join(" ") }
    };
  }

  // ../../drive-builds/mesh-automaton-extracted/address/gate.mjs
  var AddressRequired = class extends Error {
  };
  function gate(address, input) {
    if (address == null || address === "") throw new AddressRequired("address required before any input is accepted");
    let a;
    try {
      a = parse(address);
    } catch (e) {
      throw new AddressRequired(`invalid address: ${e.message}`);
    }
    if (a.unresolved) throw new AddressRequired(`address unresolved: ${a.reason}`);
    const s = state(a);
    return { address: code(a), index: s.index, state: s, input };
  }

  // ../../drive-builds/mesh-automaton-extracted/dist/mesh/src/mesh-engine.mjs
  var MAX_GRAMMAR_DEPTH = 4;
  var MAX_GRAMMAR_ACTIONS = 32;
  function isRecord(value) {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }
  function parseJson(raw2, fallback) {
    try {
      return JSON.parse(raw2);
    } catch {
      return fallback;
    }
  }
  function stableSortByPriority(items) {
    return [...items].sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id));
  }
  function interpolateValue(value, context) {
    if (typeof value === "string") {
      const exact = value.match(/^{{([a-zA-Z0-9_]+)}}$/);
      if (exact) return context[exact[1]] ?? value;
      return value.replace(/{{([a-zA-Z0-9_]+)}}/g, (_, key) => String(context[key] ?? `{{${key}}}`));
    }
    if (Array.isArray(value)) return value.map((item) => interpolateValue(item, context));
    if (isRecord(value)) {
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, interpolateValue(item, context)]));
    }
    return value;
  }
  function interpolateAction(action, context) {
    return interpolateValue(action, context);
  }
  function productionFor(rule) {
    const parsed = parseJson(rule.production, { actions: [] });
    if (!isRecord(parsed) || !Array.isArray(parsed.actions)) return { actions: [] };
    return { actions: parsed.actions };
  }
  function expandGrammar(rules, startSymbol, context) {
    const selectedRules = [];
    const actions = [];
    const queue = [{ symbol: startSymbol, depth: 0 }];
    while (queue.length > 0 && actions.length < MAX_GRAMMAR_ACTIONS) {
      const current = queue.shift();
      if (current.depth > MAX_GRAMMAR_DEPTH) continue;
      const rule = stableSortByPriority(rules.filter((candidate) => candidate.enabled && candidate.symbol === current.symbol))[0];
      if (!rule) continue;
      selectedRules.push(rule.id);
      for (const rawAction of productionFor(rule).actions) {
        const action = interpolateAction(rawAction, { ...context, ruleId: rule.id, symbol: current.symbol });
        if (action.type === "expand") {
          queue.push({ symbol: action.symbol, depth: current.depth + 1 });
        } else if (actions.length < MAX_GRAMMAR_ACTIONS) {
          actions.push(action);
        }
      }
    }
    return { startSymbol, selectedRules, actions };
  }
  function asRecord(value) {
    return isRecord(value) ? value : { value };
  }
  function uniqueStrings(value) {
    return Array.isArray(value) ? Array.from(new Set(value.filter((item) => typeof item === "string"))) : [];
  }
  function invokeTool(tool, allTools, payload, seen = /* @__PURE__ */ new Set()) {
    if (seen.has(tool.id)) {
      return { success: false, output: { error: "Pipeline cycle rejected", toolId: tool.id }, stages: [] };
    }
    const nextSeen = new Set(Array.from(seen));
    nextSeen.add(tool.id);
    switch (tool.profile) {
      case "envelope": {
        return {
          success: true,
          output: { payload, envelope: { toolId: tool.id, name: tool.name, capabilities: tool.capabilities } },
          stages: [{ toolId: tool.id, profile: tool.profile, success: true }]
        };
      }
      case "project": {
        const fields = uniqueStrings(tool.configuration.fields);
        const output = fields.reduce((result, field) => {
          if (field in payload) result[field] = payload[field];
          return result;
        }, {});
        return { success: true, output, stages: [{ toolId: tool.id, profile: tool.profile, success: true }] };
      }
      case "merge": {
        const defaults = asRecord(tool.configuration.defaults);
        return {
          success: true,
          output: { ...defaults, ...payload },
          stages: [{ toolId: tool.id, profile: tool.profile, success: true }]
        };
      }
      case "annotate": {
        const labels = uniqueStrings(tool.configuration.labels).sort((a, b) => a.localeCompare(b));
        const prior = uniqueStrings(payload.annotations);
        return {
          success: true,
          output: { ...payload, annotations: Array.from(/* @__PURE__ */ new Set([...prior, ...labels])) },
          stages: [{ toolId: tool.id, profile: tool.profile, success: true }]
        };
      }
      case "pipeline": {
        let current = payload;
        const stages = [];
        for (const toolId of tool.composition) {
          const child = allTools.find((candidate) => candidate.id === toolId);
          if (!child) {
            return { success: false, output: { error: "Pipeline stage unavailable", toolId }, stages };
          }
          const result = invokeTool(child, allTools, current, nextSeen);
          stages.push(...result.stages);
          if (!result.success) return { success: false, output: result.output, stages };
          current = result.output;
        }
        return {
          success: true,
          output: current,
          stages: [{ toolId: tool.id, profile: tool.profile, success: true }, ...stages]
        };
      }
    }
  }
  function findRoute(edges, sourceNodeId, targetNodeId) {
    if (sourceNodeId === targetNodeId) return [];
    const adjacency = /* @__PURE__ */ new Map();
    for (const edge of stableSortByPriority(edges.filter((edge2) => edge2.enabled))) {
      adjacency.set(edge.sourceNodeId, [...adjacency.get(edge.sourceNodeId) ?? [], edge]);
    }
    const queue = [{ nodeId: sourceNodeId, path: [] }];
    const visited = /* @__PURE__ */ new Set([sourceNodeId]);
    while (queue.length > 0) {
      const current = queue.shift();
      for (const edge of adjacency.get(current.nodeId) ?? []) {
        if (current.path.length >= edge.maxHops) continue;
        const path = [...current.path, edge];
        if (edge.targetNodeId === targetNodeId) return path;
        if (!visited.has(edge.targetNodeId)) {
          visited.add(edge.targetNodeId);
          queue.push({ nodeId: edge.targetNodeId, path });
        }
      }
    }
    return null;
  }
  function executeMeshCycle(snapshot, runId, input) {
    const nodeStates = new Map(snapshot.nodes.map((node) => [node.id, { ...node.nodeState }]));
    const events = [];
    let sequence = 0;
    let messageCount = 0;
    let toolInvocations = 0;
    const record = (eventType, payload, nodeId, messageId) => {
      events.push({ sequence: ++sequence, eventType, payload, nodeId, messageId });
    };
    const patchNode = (nodeId, patch) => {
      const prior = nodeStates.get(nodeId) ?? {};
      const next = { ...prior, ...patch };
      nodeStates.set(nodeId, next);
      record("state.patched", { patch }, nodeId);
    };
    const emit = (source, target, intent, payload) => {
      const messageId = `message-${String(messageCount + 1).padStart(3, "0")}`;
      messageCount++;
      const route = findRoute(snapshot.edges, source, target);
      record("message.emitted", { source, target, intent, payload }, source, messageId);
      if (!route) {
        record("message.unroutable", { source, target, intent }, source, messageId);
        return;
      }
      let currentNode = source;
      for (const edge of route) {
        currentNode = edge.targetNodeId;
        record("message.routed", { edgeId: edge.id, from: edge.sourceNodeId, to: edge.targetNodeId, intent }, currentNode, messageId);
        const existing = nodeStates.get(currentNode) ?? {};
        nodeStates.set(currentNode, {
          ...existing,
          messageCount: (typeof existing.messageCount === "number" ? existing.messageCount : 0) + 1,
          lastIntent: intent,
          lastMessageId: messageId,
          lastOrigin: source
        });
      }
      record("message.delivered", { source, target, intent, hops: route.length, payload }, target, messageId);
    };
    const plan = expandGrammar(snapshot.rules, "CYCLE", { input, runId, seed: snapshot.sessionSeed });
    record("grammar.expanded", { startSymbol: plan.startSymbol, selectedRules: plan.selectedRules, actionCount: plan.actions.length });
    for (const action of plan.actions) {
      if (action.type === "emit") {
        emit(action.source, action.target, action.intent, action.payload ?? {});
        continue;
      }
      if (action.type === "patch") {
        patchNode(action.nodeId, action.patch);
        continue;
      }
      if (action.type === "invoke") {
        const tool = snapshot.tools.find((candidate) => candidate.id === action.toolId);
        if (!tool) {
          record("tool.unsatisfied", { toolId: action.toolId, source: action.source, target: action.target }, action.source);
          continue;
        }
        const result = invokeTool(tool, snapshot.tools, action.payload ?? {});
        toolInvocations++;
        record("tool.executed", { toolId: tool.id, name: tool.name, profile: tool.profile, success: result.success, output: result.output, stages: result.stages }, action.source);
        if (result.success) emit(action.source, action.target, "tool.output", result.output);
        continue;
      }
    }
    const summary = {
      grammarRulesApplied: plan.selectedRules.length,
      actionsExpanded: plan.actions.length,
      messages: messageCount,
      toolInvocations,
      events: events.length,
      status: "complete"
    };
    record("run.completed", summary);
    return {
      plan,
      events,
      nodeStates: snapshot.nodes.map((node) => ({ nodeId: node.id, nodeState: nodeStates.get(node.id) ?? {} })),
      summary
    };
  }

  // ../../drive-builds/mesh-automaton-extracted/mesh/local-mesh.mjs
  var n = 0;
  var id = (p) => `${p}-${Date.now().toString(36)}${(n++).toString(36)}`;
  function makeDefaultSession(sessionId = "local", opts = {}) {
    const I = `node-input-${sessionId}`, G = `node-grammar-${sessionId}`, F = `node-factory-${sessionId}`, S2 = `node-state-${sessionId}`, L = `node-ledger-${sessionId}`, T = `tool-annotate-${sessionId}`;
    const node = (id2, label2, kind, capabilities, configuration, x, y) => ({ id: id2, sessionId, label: label2, kind, capabilities, configuration, nodeState: { phase: "ready", messageCount: 0 }, positionX: x, positionY: y });
    const edge = (i, s, t, label2) => ({ id: `edge-0${i}-${sessionId}`, sessionId, sourceNodeId: s, targetNodeId: t, label: label2, priority: 1, maxHops: 8, enabled: true });
    return {
      id: sessionId,
      name: opts.name || "Local mesh",
      description: opts.description || "",
      sessionSeed: opts.seed || "314159",
      defaultInput: opts.defaultInput || "Route a structured signal through the mesh.",
      status: "idle",
      nodes: [
        node(I, "Input relay", "ingress", ["message.receive"], { role: "cycle-entry" }, 92, 262),
        node(G, "Grammar kernel", "grammar", ["grammar.expand"], { symbol: "CYCLE" }, 300, 150),
        node(F, "Tool factory", "factory", ["tool.create", "tool.invoke"], { profiles: ["envelope", "project", "merge", "annotate", "pipeline"] }, 532, 262),
        node(S2, "State weaver", "transform", ["state.propagate"], { behavior: "merge-patches" }, 746, 150),
        node(L, "Trace ledger", "observer", ["trace.record"], { retain: "all" }, 954, 262)
      ],
      edges: [edge(1, I, G, "admit"), edge(2, G, F, "expand"), edge(3, F, S2, "compose"), edge(4, S2, L, "commit")],
      rules: [{ id: `rule-cycle-${sessionId}`, sessionId, name: "Canonical cycle", symbol: "CYCLE", priority: 1, enabled: true, production: JSON.stringify({ actions: [
        { type: "emit", source: I, target: G, intent: "cycle.open", payload: { input: "{{input}}", seed: "{{seed}}" } },
        { type: "emit", source: G, target: F, intent: "grammar.plan", payload: { composition: "envelope > annotate", input: "{{input}}" } },
        { type: "invoke", source: F, target: S2, toolId: T, payload: { input: "{{input}}", origin: "grammar" } },
        { type: "patch", nodeId: S2, patch: { phase: "propagated", lastInput: "{{input}}", lastRun: "{{runId}}" } },
        { type: "emit", source: S2, target: L, intent: "cycle.complete", payload: { input: "{{input}}", status: "complete" } }
      ] }) }],
      tools: [{ id: T, sessionId, name: "Annotate context", profile: "annotate", capabilities: ["context.annotate", "message.transform"], configuration: { labels: ["deterministic", "grammar-driven", "mesh-routed"] }, composition: [], factoryGenerated: true }],
      runs: []
    };
  }
  function runMeshCycle(session2, input, extra = {}) {
    const runId = id("run");
    const effective = (input || "").trim() || session2.defaultInput;
    const run = { id: runId, runSeed: session2.sessionSeed, input: effective, status: "running", ...extra };
    session2.runs.push(run);
    session2.status = "running";
    try {
      const result = executeMeshCycle(session2, runId, effective);
      for (const st of result.nodeStates) {
        const nd = session2.nodes.find((x) => x.id === st.nodeId);
        if (nd) nd.nodeState = st.nodeState;
      }
      Object.assign(run, { status: "complete", summary: result.summary, events: result.events, completedAt: (/* @__PURE__ */ new Date()).toISOString() });
      session2.status = "complete";
      return { session: session2, result, runId };
    } catch (e) {
      Object.assign(run, { status: "failed", summary: { status: "failed", error: e.message } });
      session2.status = "failed";
      throw e;
    }
  }

  // shim/url.mjs
  var fileURLToPath = (u) => String(u);

  // ../../app-catalog/engine/services/address-service.mjs
  var DONOR_ROOT = fileURLToPath(new URL("../donors/Back-up-/vendor/", "file:///eng/x/y.mjs"));
  var SPINE_ADDRESS = DONOR_ROOT + "execution-spine-v0.4.0/src/canonical-address.mjs";
  var KIMI_STATE_SPACE = DONOR_ROOT + "kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/";
  var YNIV_BRIDGE = fileURLToPath(new URL("../donors/recovered/you-n-i-verse-corrected/synthia-bridge.ported.mjs", "file:///eng/x/y.mjs"));
  var YNIV_EDGES = fileURLToPath(new URL("../donors/recovered/you-n-i-verse-corrected/emergent-edge-resolver.ported.mjs", "file:///eng/x/y.mjs"));
  var FOUNDRY_EPHEMERIS = fileURLToPath(new URL("../donors/recovered/foundry-glyphs/server/resonance-engine.ported.mjs", "file:///eng/x/y.mjs"));
  var ADDRESS_FIELDS = Object.freeze([
    "planetary",
    "dimension",
    "gate",
    "line",
    "color",
    "tone",
    "base",
    "degree",
    "minute",
    "second",
    "arc",
    "zodiac",
    "house"
  ]);
  var PROVIDER_ID = "back-up-:execution-spine-canonical-address+kimi-dms-codec";
  var HD_PROVIDER_ID = "back-up-:kimi-human-design";
  var YNIV_PROVIDER_ID = "recovered:yniv-addressing-engine";
  var YNIV_EDGE_PROVIDER_ID = "recovered:yniv-emergent-edge-resolver";
  var FOUNDRY_PROVIDER_ID = "recovered:foundry-glyphs-ephemeris";
  function emptyAddress() {
    return Object.freeze(Object.fromEntries(ADDRESS_FIELDS.map((f) => [f, null])));
  }
  var AddressService = class {
    constructor({ bus = null, capabilityRegistry = null } = {}) {
      this.bus = bus;
      this.capabilityRegistry = capabilityRegistry;
      this.providerId = PROVIDER_ID;
      this._spine = null;
      this._dms = null;
      this._hd = null;
      this._yniv = null;
      this._ynivEdges = null;
      this._ephemeris = null;
    }
    async _load(name, path) {
      try {
        return await import(path);
      } catch (error) {
        const failure = { provider: name, path, error: String(error?.message ?? error) };
        this.bus?.emit("service:provider-failure", failure);
        const err = new Error(`address-service donor provider unavailable: ${name} (${failure.error})`);
        err.cause = error;
        err.providerFailure = failure;
        throw err;
      }
    }
    async spine() {
      return this._spine ??= await this._load("execution-spine-canonical-address", SPINE_ADDRESS);
    }
    async dms() {
      return this._dms ??= await this._load("kimi-addressing-dms", KIMI_STATE_SPACE + "addressing.js");
    }
    async hd() {
      return this._hd ??= await this._load("kimi-human-design", KIMI_STATE_SPACE + "human-design.js");
    }
    async yniv() {
      return this._yniv ??= await this._load("recovered-yniv-addressing", YNIV_BRIDGE);
    }
    async ynivEdges() {
      return this._ynivEdges ??= await this._load("recovered-yniv-edge-resolver", YNIV_EDGES);
    }
    async ephemeris() {
      return this._ephemeris ??= await this._load("recovered-foundry-glyphs-ephemeris", FOUNDRY_EPHEMERIS);
    }
    /**
     * Multi-provider routing decision (recorded, never silent). Provider is chosen
     * by INPUT SHAPE (each provider resolves what it actually implements); the
     * canonical-address authority remains execution-spine (validation role).
     */
    _routingDecision(capability, selected, rationale) {
      const candidates = this.capabilityRegistry ? this.capabilityRegistry.queryCapability(capability).providers.map((p) => p.provider_id) : [];
      const decision = {
        request_id: `req-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        required_capability: capability,
        candidate_providers: candidates,
        selected_provider: selected,
        selection_rationale: rationale,
        timestamp: (/* @__PURE__ */ new Date()).toISOString(),
        outcome_event_id: null
      };
      this.bus?.emit("routing:decision", decision);
      return decision;
    }
    /**
     * Resolve a canonical 13-field address from real input only.
     * Accepted inputs (first match wins, rest recorded as unknown):
     *   { birthDate: 'YYYY-MM-DD', birthTime: 'HH:MM', planet? }  -> HD chart placement (default Sun)
     *   { arcSecond | arc: int }                                  -> DMS wheel decode
     *   { address: {...} }                                        -> normalize/validate onto 13 fields
     */
    async resolveAddress(entityOrEvent = {}, options = {}) {
      const input = entityOrEvent?.input ?? entityOrEvent;
      if (input == null || typeof input !== "object") throw new TypeError("resolveAddress: object input required");
      const strategy = options.strategy ?? "auto";
      if (strategy === "canonical-validation") {
        this._routingDecision(
          "resolve_address",
          "back-up-:execution-spine-canonical-address",
          "strategy=canonical-validation -> canonical-address AUTHORITY validates the completed address (per selection_hints: spine is always the final validator)"
        );
        return this.validateCanonical(input);
      }
      if (strategy === "micro-resolution" && (input.micro || input.macro || input.emergent)) {
        this._routingDecision("resolve_address", YNIV_PROVIDER_ID, "strategy=micro-resolution -> YNIV 13-dim resolution engine (selection_hints: encoding/decoding full 13-dim address arithmetic)");
        return this._resolveViaYNIV(input);
      }
      if (strategy !== "auto") throw new Error(`resolveAddress: strategy '${strategy}' not applicable to this input shape`);
      if (input.birthDate) {
        this._routingDecision("resolve_address", HD_PROVIDER_ID, "birth datum present -> HD chart provider (kimi)");
        return this._resolveFromBirthDatum(input);
      }
      if (input.ephemeris && typeof input.ephemeris === "object") {
        this._routingDecision("calculate_human_design", FOUNDRY_PROVIDER_ID, "ephemeris coordinates present -> Foundry-Glyphs mandala resolver (real ephemeris engine)");
        return this._resolveFromEphemeris(input.ephemeris);
      }
      if (input.micro || input.macro || input.emergent) {
        this._routingDecision("resolve_address", YNIV_PROVIDER_ID, "YNIV coordinate object present (micro/macro/emergent) -> recovered 13-dim address engine");
        return this._resolveViaYNIV(input);
      }
      if (Number.isFinite(input.arcSecond ?? input.arc)) {
        this._routingDecision("resolve_address", this.providerId, "arc-second coordinate present -> kimi DMS wheel codec");
        return this._resolveFromArc(input.arcSecond ?? input.arc);
      }
      if (input.address && typeof input.address === "object") return this.normalizeAddress(input.address);
      return {
        address: emptyAddress(),
        provider: this.providerId,
        resolved_fields: [],
        status: "unknown",
        reason: "no birth datum, arc coordinate, or address supplied"
      };
    }
    async _resolveFromBirthDatum({ birthDate, birthTime = "00:00", planet = "Sun" }) {
      const hd = await this.hd();
      const chart = hd.calculateHumanDesign(birthDate, birthTime, null);
      const placement = (chart.placements ?? []).find((p) => p.planet === planet) ?? null;
      if (!placement) {
        throw new Error(`HD chart returned no placement for planet: ${planet}`);
      }
      const dms2 = await this.dms();
      const arcSecond = Math.round(placement.longitude * 3600);
      const wheel = dms2.addressForArcSec(arcSecond);
      const address = {
        planetary: placement.planet,
        dimension: null,
        // dimension is not derivable from a birth datum alone
        gate: placement.gate,
        line: placement.line,
        color: placement.color,
        tone: placement.tone,
        base: placement.base,
        degree: placement.degree,
        minute: placement.minute,
        second: placement.second,
        arc: arcSecond,
        zodiac: placement.zodiac,
        // sign name string (HD provider convention)
        house: wheel.house
        // 1-8 trigram houses (kimi codec convention)
      };
      return {
        address,
        provider: HD_PROVIDER_ID,
        resolved_fields: ADDRESS_FIELDS.filter((f) => address[f] !== null),
        status: "resolved",
        provider_metadata: {
          wheel_decode: wheel,
          chart_summary: {
            type: chart.type ?? null,
            profile: chart.profile ?? null,
            authority: chart.authority ?? null,
            incarnationCross: chart.incarnationCross ?? null
          }
        },
        chart
      };
    }
    async _resolveFromArc(arcSecond) {
      const dms2 = await this.dms();
      const wheel = dms2.addressForArcSec(arcSecond);
      const hd = await this.hd();
      const zodiac = hd.longitudeToZodiac(arcSecond / 3600);
      const address = {
        planetary: null,
        dimension: null,
        gate: wheel.gate,
        line: wheel.line,
        color: wheel.color,
        tone: wheel.tone,
        base: wheel.base,
        degree: zodiac.degree,
        minute: zodiac.minute,
        second: zodiac.second,
        arc: wheel.arcSecond,
        zodiac: zodiac.sign,
        house: wheel.house
        // 1-8 trigram houses (kimi codec convention)
      };
      return {
        address,
        provider: this.providerId,
        resolved_fields: ADDRESS_FIELDS.filter((f) => address[f] !== null),
        status: "resolved",
        provider_metadata: { wheel_decode: wheel }
      };
    }
    /**
     * YNIV recovered 13-dim address engine (donor synthia-bridge.ts, ported verbatim).
     * Encodes the given coordinates through the REAL donor encoder and decodes back
     * (round-trip proven); contract fields are filled from donor decode output.
     * Conventions (donor): micro gate/line/color/tone are 0-based (0 = gate 1); macro
     * planet 0-12, dimension 0-4, zodiac 0-11, house 0-11; emergent arcSecond is 0-99
     * (quarter-minute), NOT the contract/kimi arc-second coordinate — kept in
     * provider_metadata.yniv only; contract `arc` stays null unless supplied.
     */
    async _resolveViaYNIV(input) {
      const yniv = await this.yniv();
      const address = { ...emptyAddress() };
      const meta = {};
      if (input.micro) {
        const microIndex = yniv.encodeMicro(input.micro);
        const decoded = yniv.decodeMicro(microIndex);
        Object.assign(address, {
          gate: decoded.gate + 1,
          line: decoded.line + 1,
          color: decoded.color + 1,
          tone: decoded.tone + 1,
          base: decoded.base + 1
        });
        meta.micro = { index: microIndex, decoded, size: yniv.MICRO_SIZE, convention: "0-based donor coords -> 1-based contract fields" };
      }
      if (input.macro) {
        const macroIndex = yniv.encodeMacro(input.macro);
        const decoded = yniv.decodeMacro(macroIndex);
        address.planetary = input.macro.planet_name ?? String(decoded.planet);
        address.dimension = input.macro.dimension_name ?? String(decoded.dimension);
        address.zodiac = null;
        address.house = decoded.house + 1;
        meta.macro = { index: macroIndex, decoded, size: yniv.MACRO_SIZE, note: "planetary/dimension/zodiac are numeric in the donor macro layer; left as raw indices unless caller supplies names" };
        if (input.macro.zodiac_name) address.zodiac = input.macro.zodiac_name;
        if (address.planetary !== null && input.macro.planet_name === void 0) meta.macro.planetary_raw = address.planetary;
      }
      if (input.emergent) {
        meta.emergent = { ...input.emergent, note: "donor emergent layer (degree 0-360, minute/second 0-60, arcSecond 0-99); arcSecond is NOT contract 'arc'" };
        if (Number.isFinite(input.emergent.degree)) address.degree = input.emergent.degree;
        if (Number.isFinite(input.emergent.minute)) address.minute = input.emergent.minute;
        if (Number.isFinite(input.emergent.second)) address.second = input.emergent.second;
      }
      return {
        address,
        provider: YNIV_PROVIDER_ID,
        resolved_fields: ADDRESS_FIELDS.filter((f) => address[f] !== null),
        status: "resolved",
        provider_metadata: { yniv: meta, round_trip: true }
      };
    }
    /**
     * Foundry-Glyphs real ephemeris resolver (mandala wheel, Fagan-Bradley
     * sidereal, draconic) — DIFFERENT calculation engine from kimi HD
     * (approximate ephemeris). Both retained; no winner picked.
     * Input: { sign, degree, minute, second } (tropical ecliptic coordinates).
     */
    async _resolveFromEphemeris({ sign, degree = 0, minute = 0, second = 0 }) {
      const eph = await this.ephemeris();
      const act = eph.getHDActivation(sign, degree, minute, second);
      const address = {
        planetary: null,
        dimension: null,
        gate: act.gate,
        line: act.line,
        color: act.color,
        tone: act.tone,
        base: act.base,
        degree: act.degree,
        minute: act.minute,
        second: act.second,
        arc: Math.round(act.eclipticDeg * 3600),
        // absolute ecliptic arc-seconds (contract arc convention)
        zodiac: act.sign,
        house: null
        // mandala wheel provides no house; unknown stays null
      };
      return {
        address,
        provider: FOUNDRY_PROVIDER_ID,
        resolved_fields: ADDRESS_FIELDS.filter((f) => address[f] !== null),
        status: "resolved",
        provider_metadata: {
          activation: act,
          engine: `Foundry-Glyphs mandala wheel (GATE_SEQUENCE from 0\xB0 Aries gate 25, GATE_ARC 5\xB037'30")`,
          conversions: { sidereal: "tropicalToSidereal(Fagan-Bradley)", draconic: "tropicalToDraconic" }
        }
      };
    }
    /**
     * resolveRelationship via the recovered EmergentEdgeResolver (36 HD channels
     * as executable functions). Gates are contract 1-64; donor is 0-based.
     */
    async resolveEdge(addressA, addressB, stateOverlays = {}) {
      const edges = await this.ynivEdges();
      const resolver = new edges.EmergentEdgeResolver();
      const toNodeState = (a, extra) => ({
        micro: { gate: a.gate - 1, line: (a.line ?? 1) - 1, color: (a.color ?? 1) - 1, tone: (a.tone ?? 1) - 1, base: (a.base ?? 1) - 1 },
        macro: { planet: 0, dimension: 0, zodiac: 0, house: 0 },
        emergent: { degree: a.degree ?? 0, minute: a.minute ?? 0, second: a.second ?? 0, arcSecond: 0 },
        amplitude: extra.amplitude ?? 0.8,
        phase: extra.phase ?? 0.5,
        coherence: extra.coherence ?? 0.9,
        dimension: extra.dimension ?? 0
      });
      const s = toNodeState(addressA, stateOverlays.source ?? {});
      const t = toNodeState(addressB, stateOverlays.target ?? {});
      const result = resolver.resolve(s, t);
      this.bus?.emit("address-service:edge-resolved", { edgeType: result.edgeType, gates: [addressA.gate, addressB.gate] });
      return { ...result, provider: YNIV_EDGE_PROVIDER_ID };
    }
    /**
     * Canonical validation by the AUTHORITY provider (execution-spine).
     * Accepts a spine-form address (with arcAxis) directly, or YNIV coordinate
     * parts ({micro, macro, emergent}) which are first resolved by the real YNIV
     * engine and then translated into spine form for validation. Contract `arc`
     * (kimi arc-seconds) is NOT translated into arcAxis — that mapping is a
     * documented open reconciliation; only donor-native arcSecond (0-99) maps to
     * arcAxis.arcUnit (1-99). Validation failures are REAL results, surfaced.
     */
    async validateCanonical(input = {}) {
      const spine = await this.spine();
      let spineAddress;
      if (input.address?.arcAxis) {
        spineAddress = input.address;
      } else if (input.micro && input.macro && input.emergent) {
        const resolved = await this._resolveViaYNIV(input);
        const a = resolved.address;
        const missing = [];
        if (!input.macro) missing.push("macro");
        spineAddress = {
          planetary: input.macro.planet + 1,
          dimension: spine.DIMENSIONS[input.macro.dimension],
          gate: a.gate,
          line: a.line,
          color: a.color,
          tone: a.tone,
          base: a.base,
          degree: input.emergent.degree ?? 0,
          minute: input.emergent.minute ?? 0,
          second: input.emergent.second ?? 0,
          arcAxis: { arcUnit: (input.emergent.arcSecond ?? 0) + 1, ...input.emergent.axis ? { axis: input.emergent.axis } : {} },
          zodiac: input.macro.zodiac + 1,
          house: input.macro.house + 1
        };
        if (missing.length) return { valid: false, provider: "back-up-:execution-spine-canonical-address", reason: `missing parts: ${missing.join(",")}` };
      } else {
        return {
          valid: null,
          status: "unknown",
          provider: "back-up-:execution-spine-canonical-address",
          reason: "no spine-form address (arcAxis) or YNIV coordinate parts supplied; contract-form addresses are not translated (arc<->arcAxis reconciliation pending)"
        };
      }
      try {
        spine.validateCanonicalAddress(spineAddress);
        return {
          valid: true,
          provider: "back-up-:execution-spine-canonical-address",
          canonical_key: spine.canonicalAddressKey(spineAddress),
          spine_address: spineAddress
        };
      } catch (error) {
        const result = { valid: false, provider: "back-up-:execution-spine-canonical-address", error: String(error?.message ?? error), spine_address: spineAddress };
        this.bus?.emit("address-service:validation-failed", result);
        return result;
      }
    }
    /** Normalize a partial/canonical address onto exactly the 13 contract fields. */
    async normalizeAddress(partial) {
      const address = { ...emptyAddress() };
      const resolved = [];
      for (const f of ADDRESS_FIELDS) {
        if (partial[f] !== void 0 && partial[f] !== null) {
          address[f] = partial[f];
          resolved.push(f);
        }
      }
      if (partial.arcSecond !== void 0 && address.arc === null) {
        address.arc = partial.arcSecond;
        resolved.push("arc");
      }
      return { address, provider: this.providerId, resolved_fields: resolved, status: resolved.length ? "resolved" : "unknown" };
    }
    /** Field-by-field comparison + donor Fu Xi Hamming distance when both gates known. */
    async compareAddresses(a, b) {
      const na = (await this.normalizeAddress(a?.address ?? a)).address;
      const nb = (await this.normalizeAddress(b?.address ?? b)).address;
      const shared = [];
      const differing = [];
      for (const f of ADDRESS_FIELDS) {
        if (na[f] === null || nb[f] === null) continue;
        (String(na[f]) === String(nb[f]) ? shared : differing).push(f);
      }
      let hammingDistance = null;
      if (Number.isInteger(na.gate) && Number.isInteger(nb.gate)) {
        const dms2 = await this.dms();
        hammingDistance = dms2.hamming(dms2.gateBits(na.gate), dms2.gateBits(nb.gate));
      }
      return {
        a: na,
        b: nb,
        shared_fields: shared,
        differing_fields: differing,
        hamming_distance: hammingDistance,
        same_address: shared.length === ADDRESS_FIELDS.length && differing.length === 0,
        provider: this.providerId
      };
    }
    /**
     * Honest structural relationship: only what the donor math supports.
     * relational_class stays 'unknown' unless evidence warrants more.
     */
    async resolveRelationship(a, b, context = {}) {
      const cmp = await this.compareAddresses(a, b);
      const structural = [];
      if (cmp.shared_fields.includes("gate")) structural.push("shared_gate");
      if (cmp.shared_fields.includes("line")) structural.push("shared_line");
      if (cmp.hamming_distance !== null) {
        if (cmp.hamming_distance === 0) structural.push("identical_gate_binary");
        if (cmp.hamming_distance === 6) structural.push("complementary_gates");
      }
      let edge = null;
      if (Number.isInteger(cmp.a.gate) && Number.isInteger(cmp.b.gate) && context.useEdgeResolver !== false) {
        edge = await this.resolveEdge(cmp.a, cmp.b, context.stateOverlays ?? {});
        if (edge.compatible && edge.edgeType !== "NONE") structural.push(`channel:${edge.edgeType}`);
      }
      return {
        a: cmp.a,
        b: cmp.b,
        comparison: cmp,
        structural_relations: structural,
        edge: edge ? {
          provider: edge.provider,
          edge_type: edge.edgeType,
          edge_name: edge.edgeName,
          compatible: edge.compatible,
          score: edge.score,
          emergent: edge.emergent
        } : null,
        relational_class: "unknown",
        // grammar v1: do not assert co_occurrence/causal without evidence
        context: context ?? null,
        provider: this.providerId
      };
    }
    /**
     * Append-only address history: derived from the shared event log
     * (events where entity is actor or target and an address was recorded).
     */
    async traceAddressHistory(entity, { events = null } = {}) {
      const id2 = typeof entity === "string" ? entity : entity?.identity ?? entity?.id;
      if (!id2) return { identity: null, history: [], status: "unknown", reason: "no entity identity" };
      const log = events ?? [];
      const history = log.filter((e) => e && (e.actor_id === id2 || e.target_id === id2)).filter((e) => e.actor_address || e.target_address || e.event_address).map((e) => ({
        event_id: e.event_id,
        timestamp: e.timestamp,
        role: e.actor_id === id2 ? "actor" : "target",
        address: e.actor_id === id2 ? e.actor_address ?? e.event_address : e.target_address ?? e.event_address,
        trajectory_id: e.trajectory_id ?? null
      }));
      return {
        identity: id2,
        history,
        status: history.length ? "resolved" : "unknown",
        provider: "computer:events/event-emitter"
      };
    }
  };

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/addressing.js
  var addressing_exports = {};
  __export(addressing_exports, {
    KING_WEN_TO_FUXI_DECIMAL: () => KING_WEN_TO_FUXI_DECIMAL2,
    addrKey: () => addrKey,
    addressForArcSec: () => addressForArcSec,
    arcSecForAddress: () => arcSecForAddress,
    gateBits: () => gateBits,
    gateFromBits: () => gateFromBits,
    hamming: () => hamming
  });

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/constants.js
  var WHEEL_ARCSECONDS = 1296e3;
  var GATE_ARCSECONDS = 20250;
  var LINE_ARCSECONDS = 3375;
  var COLOR_ARCSECONDS = 562.5;
  var TONE_ARCSECONDS = 93.75;
  var BASE_ARCSECONDS = 18.75;
  var ZODIAC_ARCSECONDS = 108e3;
  var HOUSE_ARCSECONDS = 162e3;
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function() {
      a |= 0;
      a = a + 1831565813 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/addressing.js
  var KING_WEN_TO_FUXI_DECIMAL2 = Object.freeze({
    1: 63,
    2: 0,
    3: 17,
    4: 34,
    5: 23,
    6: 58,
    7: 2,
    8: 16,
    9: 55,
    10: 59,
    11: 7,
    12: 56,
    13: 61,
    14: 47,
    15: 4,
    16: 8,
    17: 25,
    18: 38,
    19: 3,
    20: 48,
    21: 41,
    22: 37,
    23: 32,
    24: 1,
    25: 57,
    26: 39,
    27: 33,
    28: 30,
    29: 18,
    30: 45,
    31: 28,
    32: 14,
    33: 60,
    34: 15,
    35: 40,
    36: 5,
    37: 53,
    38: 43,
    39: 20,
    40: 10,
    41: 35,
    42: 49,
    43: 31,
    44: 62,
    45: 24,
    46: 6,
    47: 26,
    48: 22,
    49: 29,
    50: 46,
    51: 9,
    52: 36,
    53: 52,
    54: 11,
    55: 13,
    56: 44,
    57: 54,
    58: 27,
    59: 50,
    60: 19,
    61: 51,
    62: 12,
    63: 21,
    64: 42
  });
  var FUXI_DECIMAL_TO_KING_WEN2 = Object.freeze(
    Object.fromEntries(Object.entries(KING_WEN_TO_FUXI_DECIMAL2).map(([g, d]) => [d, Number(g)]))
  );
  var mod = (n3, m) => (n3 % m + m) % m;
  function addressForArcSec(arcSec) {
    const a = mod(arcSec, WHEEL_ARCSECONDS);
    const gate2 = Math.floor(a / GATE_ARCSECONDS) + 1;
    const withinGate = a - (gate2 - 1) * GATE_ARCSECONDS;
    const line = Math.floor(withinGate / LINE_ARCSECONDS) + 1;
    const withinLine = withinGate - (line - 1) * LINE_ARCSECONDS;
    const color = Math.floor(withinLine / COLOR_ARCSECONDS) + 1;
    const withinColor = withinLine - (color - 1) * COLOR_ARCSECONDS;
    const tone = Math.floor(withinColor / TONE_ARCSECONDS) + 1;
    const withinTone = withinColor - (tone - 1) * TONE_ARCSECONDS;
    const base = Math.floor(withinTone / BASE_ARCSECONDS) + 1;
    const degree = Math.floor(a / 3600);
    const minute = Math.floor(a % 3600 / 60);
    const second = a % 60;
    const zodiac = Math.floor(a / ZODIAC_ARCSECONDS) + 1;
    const house = Math.floor(a / HOUSE_ARCSECONDS) + 1;
    return {
      gate: gate2,
      line,
      color,
      tone,
      base,
      degree,
      minute,
      second,
      arcSecond: a,
      zodiac,
      house
    };
  }
  function arcSecForAddress(addr) {
    const { gate: gate2 = 1, line = 1, color = 1, tone = 1, base = 1 } = addr;
    return mod(
      (gate2 - 1) * GATE_ARCSECONDS + (line - 1) * LINE_ARCSECONDS + (color - 1) * COLOR_ARCSECONDS + (tone - 1) * TONE_ARCSECONDS + (base - 1) * BASE_ARCSECONDS,
      WHEEL_ARCSECONDS
    );
  }
  function gateBits(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) {
      throw new RangeError(`gate must be 1-64, got ${gate2}`);
    }
    const d = KING_WEN_TO_FUXI_DECIMAL2[gate2];
    return [0, 1, 2, 3, 4, 5].map((i) => d >> i & 1);
  }
  function gateFromBits(bits) {
    if (!Array.isArray(bits) || bits.length !== 6) {
      throw new RangeError("gateFromBits expects a 6-bit array (line 1 first)");
    }
    const d = bits.reduce((acc, b, i) => acc | (b ? 1 : 0) << i, 0);
    return FUXI_DECIMAL_TO_KING_WEN2[d];
  }
  function hamming(a, b) {
    if (a.length !== b.length) throw new RangeError("hamming: bit arrays must have equal length");
    return a.reduce((n3, bit, i) => n3 + (bit !== b[i] ? 1 : 0), 0);
  }
  function addrKey(addr) {
    return `G${addr.gate}.L${addr.line}.C${addr.color}.T${addr.tone}.B${addr.base}`;
  }

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/human-design.js
  var human_design_exports = {};
  __export(human_design_exports, {
    GATES: () => GATES,
    GATE_BODY_REGION: () => GATE_BODY_REGION,
    GATE_CENTER: () => GATE_CENTER,
    GATE_WHEEL: () => GATE_WHEEL,
    HD_GATE_BINARY_CONFLICTS: () => HD_GATE_BINARY_CONFLICTS,
    HD_GATE_CENTER_VARIANTS: () => HD_GATE_CENTER_VARIANTS,
    HD_PROVENANCE: () => HD_PROVENANCE,
    HD_WHEEL_ANCHOR_VARIANTS: () => HD_WHEEL_ANCHOR_VARIANTS,
    activeChannels: () => activeChannels,
    calculateHumanDesign: () => calculateHumanDesign,
    default: () => human_design_default,
    donorCenterForGate: () => donorCenterForGate,
    generateAgentBlueprint: () => generateAgentBlueprint,
    getMoonLongitude: () => getMoonLongitude,
    getSunLongitude: () => getSunLongitude,
    longitudeToHD: () => longitudeToHD,
    longitudeToZodiac: () => longitudeToZodiac
  });

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/experiments/scale/controls.js
  var DEFAULT_CONTROL_SEED = 201620993;
  function shuffleWith(values, rng = mulberry32(DEFAULT_CONTROL_SEED)) {
    const arr = [...values];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  function controlMappings(mapping, { seed = DEFAULT_CONTROL_SEED, ablateKey = null } = {}) {
    const entries = mapping instanceof Map ? [...mapping.entries()] : Object.entries(mapping);
    const keys = entries.map(([k]) => k).sort();
    const base = new Map(entries);
    const values = keys.map((k) => base.get(k));
    const n3 = keys.length;
    if (n3 === 0) throw new RangeError("controlMappings requires a non-empty mapping");
    const rng = mulberry32(seed);
    const moduloValues = keys.map((_, i) => values[(i + 1) % n3]);
    const shuffledValues = shuffleWith(values, rng);
    const randomValues = keys.map(() => values[Math.floor(rng() * n3)]);
    const toObject = (vals, dropKey = null) => Object.freeze(Object.fromEntries(
      keys.map((k, i) => [k, vals[i]]).filter(([k]) => k !== dropKey)
    ));
    const abKey = ablateKey ?? keys[n3 - 1];
    const source = toObject(values);
    const modulo = toObject(moduloValues);
    const shuffled = toObject(shuffledValues);
    const random = toObject(randomValues);
    const ablated = toObject(values, abKey);
    return Object.freeze({
      seed,
      keys: Object.freeze(keys),
      ablatedKey: abKey,
      conditions: Object.freeze({ source, modulo, shuffled, random, ablated }),
      // Lookup functions (key -> value; ablated returns undefined off-support).
      lookup: Object.freeze({
        source: (k) => source[k],
        modulo: (k) => modulo[k],
        shuffled: (k) => shuffled[k],
        random: (k) => random[k],
        ablated: (k) => ablated[k]
      })
    });
  }
  var CONTROL_PROVENANCE = Object.freeze({
    flatLookup: Object.freeze({ status: "SOURCE_STATEMENT", source: "pure-synthia-phase2-REPAIRED/src/experiments/controls.js" }),
    frequencyBaseline: Object.freeze({ status: "SOURCE_STATEMENT", source: "pure-synthia-phase2-REPAIRED/src/experiments/controls.js" }),
    randomGrammar: Object.freeze({ status: "SOURCE_STATEMENT (behavior) / IMPLEMENTATION_CHOICE (seeded rng replaces Math.random)", source: "pure-synthia-phase2-REPAIRED/src/experiments/controls.js" }),
    shuffledStructure: Object.freeze({ status: "SOURCE_STATEMENT (behavior) / IMPLEMENTATION_CHOICE (seeded rng replaces Math.random)", source: "pure-synthia-phase2-REPAIRED/src/experiments/controls.js" }),
    controlMappings: Object.freeze({ status: "IMPLEMENTATION_CHOICE", source: "this port \u2014 generalization of the source controls to any mapping + seeded rng (primitive->scale experiment seed)" }),
    ablations: Object.freeze({ status: "SOURCE_STATEMENT (semantics) / IMPLEMENTATION_CHOICE (pure registry transforms instead of engine.clone())", source: "pure-synthia-phase2-REPAIRED/src/experiments/controls.js AblationStudy" })
  });

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/claim-status.js
  var CLAIM_STATUS = Object.freeze({
    SOURCE_STATEMENT: "SOURCE_STATEMENT",
    STRUCTURAL_MATH: "STRUCTURAL_MATH",
    DERIVED: "DERIVED",
    PROJECT_HYPOTHESIS: "PROJECT_HYPOTHESIS",
    RENDERER_CONVENTION: "RENDERER_CONVENTION",
    LEARNED: "LEARNED",
    EMPIRICALLY_SUPPORTED: "EMPIRICALLY_SUPPORTED",
    CONFLICT: "CONFLICT",
    CONTROL_ONLY: "CONTROL_ONLY"
  });
  var CLAIM_STATUS_VALUES = Object.freeze(Object.values(CLAIM_STATUS));
  function isClaimStatus(status) {
    return CLAIM_STATUS_VALUES.includes(status);
  }
  function claim(value, { status, source = null, hypothesisId = null, confidence = null, evidence = null } = {}) {
    if (!isClaimStatus(status)) {
      throw new RangeError(`claim: status must be one of ${CLAIM_STATUS_VALUES.join(", ")} (got ${JSON.stringify(status)})`);
    }
    if (status === CLAIM_STATUS.SOURCE_STATEMENT && (typeof source !== "string" || !source)) {
      throw new TypeError("claim: SOURCE_STATEMENT requires a source citation string");
    }
    const record = { value, status, source, hypothesisId, confidence };
    if (evidence !== null && evidence !== void 0) record.evidence = evidence;
    return deepFreeze(record);
  }
  var RESOLUTION_ORDER = Object.freeze([
    Object.freeze([CLAIM_STATUS.SOURCE_STATEMENT]),
    Object.freeze([CLAIM_STATUS.STRUCTURAL_MATH, CLAIM_STATUS.DERIVED]),
    Object.freeze([CLAIM_STATUS.RENDERER_CONVENTION]),
    Object.freeze([CLAIM_STATUS.PROJECT_HYPOTHESIS]),
    Object.freeze([CLAIM_STATUS.LEARNED, CLAIM_STATUS.EMPIRICALLY_SUPPORTED])
  ]);
  var NON_RESOLVING = Object.freeze([CLAIM_STATUS.CONFLICT, CLAIM_STATUS.CONTROL_ONLY]);
  var REQUIRED_CONTROLS = Object.freeze(
    Object.keys(controlMappings({ a: 1, b: 2 }).conditions).filter((k) => k !== "source").sort()
  );
  var PROMOTION_GATE = Object.freeze({
    from: CLAIM_STATUS.PROJECT_HYPOTHESIS,
    to: CLAIM_STATUS.EMPIRICALLY_SUPPORTED,
    requiredControls: REQUIRED_CONTROLS,
    // ['ablated','modulo','random','shuffled'] — from controlMappings
    requires: Object.freeze([
      "pre-registered mapping",
      "held-out test",
      "random/shuffled/modulo controls",
      "ablation",
      "reproducible derivations",
      "non-regression on canonical runtime"
    ])
  });
  function deepFreeze(x) {
    if (x && typeof x === "object" && !Object.isFrozen(x)) {
      for (const v of Object.values(x)) deepFreeze(v);
      Object.freeze(x);
    }
    return x;
  }

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/merged/kingwen.js
  var TABLE2 = Object.freeze([
    63,
    0,
    17,
    34,
    23,
    58,
    2,
    16,
    55,
    59,
    7,
    56,
    61,
    47,
    4,
    8,
    25,
    38,
    3,
    48,
    41,
    37,
    32,
    1,
    57,
    39,
    33,
    30,
    18,
    45,
    28,
    14,
    60,
    15,
    40,
    5,
    53,
    43,
    20,
    10,
    35,
    49,
    31,
    62,
    24,
    6,
    26,
    22,
    29,
    46,
    9,
    36,
    52,
    11,
    13,
    44,
    54,
    27,
    50,
    19,
    51,
    12,
    21,
    42
  ]);
  var KING_WEN_TO_FUXI_DECIMAL3 = Object.freeze(
    Object.fromEntries(TABLE2.map((decimal, index) => [index + 1, decimal]))
  );
  var FUXI_DECIMAL_TO_KING_WEN3 = Object.freeze(
    Object.fromEntries(TABLE2.map((decimal, index) => [decimal, index + 1]))
  );
  var KING_WEN_HEXAGRAM_NAMES2 = Object.freeze([
    "The Creative",
    "The Receptive",
    "Difficulty at the Beginning",
    "Youthful Folly",
    "Waiting (Nourishment)",
    "Conflict",
    "The Army",
    "Holding Together (Union)",
    "The Taming Power of the Small",
    "Treading (Conduct)",
    "Peace",
    "Standstill (Stagnation)",
    "Fellowship with Men",
    "Possession in Great Measure",
    "Modesty",
    "Enthusiasm",
    "Following",
    "Work on What Has Been Spoiled (Decay)",
    "Approach",
    "Contemplation (View)",
    "Biting Through",
    "Grace",
    "Splitting Apart",
    "Return (The Turning Point)",
    "Innocence (The Unexpected)",
    "The Taming Power of the Great",
    "The Corners of the Mouth (Providing Nourishment)",
    "Preponderance of the Great",
    "The Abysmal (Water)",
    "The Clinging (Fire)",
    "Influence (Wooing)",
    "Duration",
    "Retreat",
    "The Power of the Great",
    "Progress",
    "Darkening of the Light",
    "The Family (The Clan)",
    "Opposition",
    "Obstruction",
    "Deliverance",
    "Decrease",
    "Increase",
    "Break-through (Resoluteness)",
    "Coming to Meet",
    "Gathering Together (Massing)",
    "Pushing Upward",
    "Oppression (Exhaustion)",
    "The Well",
    "Revolution (Molting)",
    "The Caldron",
    "The Arousing (Shock, Thunder)",
    "Keeping Still (Mountain)",
    "Development (Gradual Progress)",
    "The Marrying Maiden",
    "Abundance (Fullness)",
    "The Wanderer",
    "The Gentle (The Penetrating, Wind)",
    "The Joyous (Lake)",
    "Dispersion (Dissolution)",
    "Limitation",
    "Inner Truth",
    "Preponderance of the Small",
    "After Completion",
    "Before Completion"
  ]);
  function gateToFuXiDecimal2(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) {
      throw new RangeError("gate must be 1..64.");
    }
    return TABLE2[gate2 - 1];
  }
  function fuXiDecimalToGate2(decimal) {
    const gate2 = FUXI_DECIMAL_TO_KING_WEN3[decimal];
    if (!gate2) {
      throw new RangeError(`No King Wen gate for Fu-Xi decimal ${decimal}.`);
    }
    return gate2;
  }
  function hexagramName2(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) {
      throw new RangeError("gate must be 1..64.");
    }
    return KING_WEN_HEXAGRAM_NAMES2[gate2 - 1];
  }
  function gatePattern2(gate2) {
    const decimal = gateToFuXiDecimal2(gate2);
    return Object.freeze(Array.from({ length: 6 }, (_, i) => decimal >> i & 1));
  }
  var kingwen_default2 = Object.freeze({
    KING_WEN_TO_FUXI_DECIMAL: KING_WEN_TO_FUXI_DECIMAL3,
    FUXI_DECIMAL_TO_KING_WEN: FUXI_DECIMAL_TO_KING_WEN3,
    KING_WEN_HEXAGRAM_NAMES: KING_WEN_HEXAGRAM_NAMES2,
    gateToFuXiDecimal: gateToFuXiDecimal2,
    fuXiDecimalToGate: fuXiDecimalToGate2,
    hexagramName: hexagramName2,
    gatePattern: gatePattern2
  });

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/merged/centers-channels.js
  var CANONICAL_CHANNELS = Object.freeze([
    [1, 8],
    [2, 14],
    [3, 60],
    [4, 63],
    [5, 15],
    [6, 59],
    [7, 31],
    [9, 52],
    [10, 20],
    [10, 34],
    [10, 57],
    [11, 56],
    [12, 22],
    [13, 33],
    [16, 48],
    [17, 62],
    [18, 58],
    [19, 49],
    [20, 34],
    [20, 57],
    [21, 45],
    [23, 43],
    [24, 61],
    [25, 51],
    [26, 44],
    [27, 50],
    [28, 38],
    [29, 46],
    [30, 41],
    [32, 54],
    [34, 57],
    [35, 36],
    [37, 40],
    [39, 55],
    [42, 53],
    [47, 64]
  ].map((pair) => Object.freeze(pair)));
  var CENTERS = Object.freeze({
    Head: Object.freeze([64, 61, 63]),
    Ajna: Object.freeze([47, 24, 4, 11]),
    Throat: Object.freeze([62, 23, 56, 35, 12, 45, 33, 20]),
    G: Object.freeze([1, 13, 25, 46, 2, 15, 10]),
    Heart: Object.freeze([40, 26, 51, 21]),
    Solar: Object.freeze([29, 30, 36, 6, 55, 37, 22]),
    Spleen: Object.freeze([48, 16, 44, 57, 50, 32, 18, 28]),
    Sacral: Object.freeze([5, 14, 29, 34, 59]),
    Root: Object.freeze([58, 38, 54, 19, 39, 41, 53])
  });
  var CENTER_NAMES = Object.freeze(Object.keys(CENTERS));
  function checkGate(g) {
    if (!Number.isInteger(g) || g < 1 || g > 64) throw new RangeError("gate must be 1..64");
  }
  function channelsForGate(g) {
    checkGate(g);
    return CANONICAL_CHANNELS.filter(([a, b]) => a === g || b === g);
  }
  function centerForGate(g) {
    checkGate(g);
    for (const [name, gates] of Object.entries(CENTERS)) {
      if (gates.includes(g)) return name;
    }
    return null;
  }
  function channelPartners(g) {
    return channelsForGate(g).map(([a, b]) => a === g ? b : a);
  }
  var centers_channels_default = Object.freeze({ CANONICAL_CHANNELS, CENTERS, CENTER_NAMES, channelsForGate, centerForGate, channelPartners });

  // ../../app-catalog/engine/donors/Back-up-/vendor/kimi-agent-automata-state-space-merge/pure-synthia-automata/src/state-space/human-design.js
  var GATE_WHEEL = [
    41,
    19,
    13,
    49,
    30,
    55,
    37,
    63,
    22,
    36,
    25,
    17,
    21,
    51,
    42,
    3,
    27,
    24,
    2,
    23,
    8,
    20,
    16,
    35,
    45,
    12,
    15,
    52,
    39,
    53,
    62,
    56,
    31,
    33,
    7,
    4,
    29,
    59,
    40,
    64,
    47,
    6,
    46,
    18,
    48,
    57,
    32,
    50,
    28,
    44,
    1,
    43,
    14,
    34,
    9,
    5,
    26,
    11,
    10,
    58,
    38,
    54,
    61,
    60
  ];
  var DEG_PER_GATE = 360 / 64;
  var DEG_PER_LINE = DEG_PER_GATE / 6;
  var DEG_PER_COLOR = DEG_PER_LINE / 6;
  var DEG_PER_TONE = DEG_PER_COLOR / 6;
  var DEG_PER_BASE = DEG_PER_TONE / 6;
  var GATE_CENTER = {
    1: "G",
    2: "G",
    3: "Sacral",
    4: "Ajna",
    5: "Sacral",
    6: "Solar Plexus",
    7: "G",
    8: "Throat",
    9: "Sacral",
    10: "G",
    11: "Ajna",
    12: "Throat",
    13: "G",
    14: "Sacral",
    15: "G",
    16: "Throat",
    17: "Ajna",
    18: "Spleen",
    19: "Root",
    20: "Throat",
    21: "Heart",
    22: "Throat",
    23: "Throat",
    24: "Ajna",
    25: "G",
    26: "Heart",
    27: "Sacral",
    28: "Spleen",
    29: "Sacral",
    30: "Solar Plexus",
    31: "Throat",
    32: "Spleen",
    33: "Throat",
    34: "Sacral",
    35: "Throat",
    36: "Solar Plexus",
    37: "Solar Plexus",
    38: "Root",
    39: "Solar Plexus",
    40: "Heart",
    41: "Root",
    42: "Sacral",
    43: "Ajna",
    44: "Spleen",
    45: "Throat",
    46: "G",
    47: "Ajna",
    48: "Spleen",
    49: "Solar Plexus",
    50: "Spleen",
    51: "Heart",
    52: "Root",
    53: "Root",
    54: "Root",
    55: "Solar Plexus",
    56: "Throat",
    57: "Spleen",
    58: "Root",
    59: "Sacral",
    60: "Root",
    61: "Head",
    62: "Throat",
    63: "Head",
    64: "Head"
  };
  var GATES = {
    1: {
      number: 1,
      name: "The Creative",
      chineseName: "\u4E7E Qian",
      center: "G",
      circuit: "Individual",
      aminoAcid: "Met",
      element: "Fire",
      binary: "111111",
      organ: "Heart",
      pressure: "Will",
      theme: "Self-Expression / Mutation",
      lines: [
        { line: 1, name: "Creation is independent of will", exalted: "Moon", detriment: "Uranus", keynote: "Time is everything. Patience is a virtue and revolution a vice." },
        { line: 2, name: "Love is light", exalted: "Venus", detriment: "Mars", keynote: "Beauty is in the eye of the beholder." },
        { line: 3, name: "The energy to sustain creative work", exalted: "Mars", detriment: "Earth", keynote: "Trial and error. Material forces can disrupt creativity." },
        { line: 4, name: "Aloneness as the medium of creativity", exalted: "Earth", detriment: "Jupiter", keynote: "Creativity must develop outside of influence." },
        { line: 5, name: "The energy to attract society", exalted: "Mars", detriment: "Uranus", keynote: "The assumption of talent. Eccentricity can handicap endurance." },
        { line: 6, name: "Objectivity", exalted: "Earth", detriment: "Pluto", keynote: "Clear assessment of creative value. Subjective appraisal leads to disappointment." }
      ]
    },
    2: {
      number: 2,
      name: "The Receptive",
      chineseName: "\u5764 Kun",
      center: "G",
      circuit: "Individual",
      aminoAcid: "Ile",
      element: "Earth",
      binary: "000000",
      organ: "Liver",
      pressure: "Heart",
      theme: "Direction / Receptivity",
      lines: [
        { line: 1, name: "Intuition", exalted: "Venus", detriment: "Mars", keynote: "Sensitivity to disharmony. Aesthetics guide direction." },
        { line: 2, name: "Genius", exalted: "Saturn", detriment: "Mars", keynote: "Unconscious alignment of stimulus and response. The natural." },
        { line: 3, name: "Patience", exalted: "Jupiter", detriment: "Uranus", keynote: "Dedication to a lifetime of receptivity. Trial and error in direction." },
        { line: 4, name: "Secretiveness", exalted: "Venus", detriment: "Mars", keynote: "Discretion preserves harmony. Loose lips sink ships." },
        { line: 5, name: "Intelligent application", exalted: "Mercury", detriment: "Earth", keynote: "The strategist. Reasoned management of resources." },
        { line: 6, name: "Fixation", exalted: "Mercury", detriment: "Saturn", keynote: "Unable to see the whole picture. Security may distort awareness." }
      ]
    },
    3: {
      number: 3,
      name: "Difficulty at the Beginning",
      chineseName: "\u5C6F Zhun",
      center: "Sacral",
      circuit: "Individual",
      aminoAcid: "Leu",
      element: "Water",
      binary: "100010",
      organ: "Sacrum",
      pressure: "Mind",
      theme: "Innovation / Ordering",
      lines: [
        { line: 1, name: "Synthesis", exalted: "Jupiter", detriment: "Mars", keynote: "Confusion is natural. Life will explain itself." },
        { line: 2, name: "Immaturity", exalted: "Mars", detriment: "Uranus", keynote: "The unrestrained acceptance of guidance." },
        { line: 3, name: "Survival", exalted: "Venus", detriment: "Pluto", keynote: "Innate knowing of what is sterile and fertile." },
        { line: 4, name: "Charisma", exalted: "Neptune", detriment: "Mars", keynote: "Innate quality which attracts valued guidance." },
        { line: 5, name: "Victimization", exalted: "Mars", detriment: "Earth", keynote: "Confusion gets victimized by those who think they can order it." },
        { line: 6, name: "Surrender", exalted: "Sun", detriment: "Pluto", keynote: "The ultimate maturity to recognize when struggle is futile." }
      ]
    },
    4: {
      number: 4,
      name: "Youthful Folly",
      chineseName: "\u8499 Meng",
      center: "Ajna",
      circuit: "Collective",
      aminoAcid: "Phe",
      element: "Air",
      binary: "010001",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Formulization / Answers",
      lines: [
        { line: 1, name: "Pleasure", exalted: "Moon", detriment: "Earth", keynote: "Perfect timing. The instinct to know when pleasure is rewarded." },
        { line: 2, name: "Acceptance", exalted: "Moon", detriment: "Mars", keynote: "Tolerance and suspension of judgment." },
        { line: 3, name: "Irresponsibility", exalted: "Venus", detriment: "Pluto", keynote: "Where art is more valued than the artist." },
        { line: 4, name: "The Liar", exalted: "Sun", detriment: "Saturn", keynote: "Role playing as an art form. Fantasy nurtures purpose." },
        { line: 5, name: "Seduction", exalted: "Jupiter", detriment: "Pluto", keynote: "Universal answers require energy. The money lenders." },
        { line: 6, name: "Excess", exalted: "Mercury", detriment: "Mars", keynote: "Repeated abuse of norms. The answer doesn't matter, life matters." }
      ]
    },
    5: {
      number: 5,
      name: "Waiting",
      chineseName: "\u9700 Xu",
      center: "Sacral",
      circuit: "Collective",
      aminoAcid: "Ser",
      element: "Wood",
      binary: "111010",
      organ: "Stomach",
      pressure: "Heart",
      theme: "Fixed Rhythms / Patterns",
      lines: [
        { line: 1, name: "Perseverance", exalted: "Mars", detriment: "Earth", keynote: "If the captain must, he goes down with the ship." },
        { line: 2, name: "Inner Peace", exalted: "Venus", detriment: "Pluto", keynote: "Maintaining composure through idealizing tranquility." },
        { line: 3, name: "Compulsiveness", exalted: "Neptune", detriment: "Moon", keynote: "Imagination limits negative effects. Unable to surrender." },
        { line: 4, name: "The Hunter", exalted: "Uranus", detriment: "Sun", keynote: "Waiting as guarantee of survival. Creative genius in passivity." },
        { line: 5, name: "Joy", exalted: "Pluto", detriment: "Pluto", keynote: "Waiting as an aspect of enlightenment. To remain calm is the ultimate aesthetic." },
        { line: 6, name: "Yielding", exalted: "Neptune", detriment: null, keynote: "Waiting is never free from pressure. Growth through the unexpected." }
      ]
    },
    6: {
      number: 6,
      name: "Conflict",
      chineseName: "\u8BBC Song",
      center: "Solar Plexus",
      circuit: "Tribal",
      aminoAcid: "Tyr",
      element: "Metal",
      binary: "010111",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Friction / Intimacy",
      lines: [
        { line: 1, name: " Retreat", exalted: "Jupiter", detriment: "Saturn", keynote: "The need to withdraw from conflict in order to preserve identity." },
        { line: 2, name: "The Guerrilla", exalted: "Mars", detriment: "Moon", keynote: "The turning of the other cheek." },
        { line: 3, name: " The Oral Blaster", exalted: "Earth", detriment: "Neptune", keynote: "The energy to burst through hostility." },
        { line: 4, name: "The Peacemaker", exalted: "Sun", detriment: "Uranus", keynote: "The power to resolve conflict through mutual understanding." },
        { line: 5, name: "Arbitration", exalted: "Mercury", detriment: "Pluto", keynote: "The ability to negotiate and bring about peace." },
        { line: 6, name: "The Bodhisattva", exalted: "Saturn", detriment: "Venus", keynote: "The example of self-sacrifice that inspires transcendence." }
      ]
    },
    7: {
      number: 7,
      name: "The Army",
      chineseName: "\u5E08 Shi",
      center: "G",
      circuit: "Collective",
      aminoAcid: "Gly",
      element: "Earth",
      binary: "010000",
      organ: "G Center",
      pressure: "Will",
      theme: "Role of the Self / Direction",
      lines: [
        { line: 1, name: "The need for authority", exalted: "Moon", detriment: "Mars", keynote: "Democracy fails without leadership." },
        { line: 2, name: "The demagogue", exalted: "Mercury", detriment: "Saturn", keynote: "The power to influence without responsibility." },
        { line: 3, name: "The anarchist", exalted: "Uranus", detriment: "Pluto", keynote: "The refusal to follow any authority." },
        { line: 4, name: "The abdicator", exalted: "Earth", detriment: "Neptune", keynote: "The refusal to accept the responsibility of leadership." },
        { line: 5, name: "The general", exalted: "Mars", detriment: "Venus", keynote: "The ability to lead and direct." },
        { line: 6, name: "The administrator", exalted: "Jupiter", detriment: "Sun", keynote: "The wisdom to manage resources." }
      ]
    },
    8: {
      number: 8,
      name: "Holding Together",
      chineseName: "\u6BD4 Bi",
      center: "Throat",
      circuit: "Tribal",
      aminoAcid: "Ala",
      element: "Water",
      binary: "000010",
      organ: "Throat",
      pressure: "Heart",
      theme: "Contribution / One-ness",
      lines: [
        { line: 1, name: "Honesty", exalted: "Pluto", detriment: "Mercury", keynote: "The quality of truth." },
        { line: 2, name: "The grinch", exalted: "Venus", detriment: "Jupiter", keynote: "The refusal to contribute." },
        { line: 3, name: "The partygoer", exalted: "Saturn", detriment: "Moon", keynote: "The love of communion." },
        { line: 4, name: "Respect", exalted: "Sun", detriment: "Mars", keynote: "The recognition of the value of others." },
        { line: 5, name: "Dharma", exalted: "Uranus", detriment: "Neptune", keynote: "The material contribution as a path to liberation." },
        { line: 6, name: "Communion", exalted: "Earth", detriment: "Pluto", keynote: "The quality of contribution." }
      ]
    },
    9: {
      number: 9,
      name: "Small Taming",
      chineseName: "\u5C0F\u755C Xiao Chu",
      center: "Sacral",
      circuit: "Collective",
      aminoAcid: "Val",
      element: "Fire",
      binary: "111011",
      organ: "Sacrum",
      pressure: "Mind",
      theme: "Focus / Concentration",
      lines: [
        { line: 1, name: "Sagacity", exalted: "Jupiter", detriment: "Mercury", keynote: "The ability to focus on the trivial." },
        { line: 2, name: "Misery loves company", exalted: "Venus", detriment: "Mars", keynote: "The ability to share bad luck." },
        { line: 3, name: "The straw that broke the camel's back", exalted: "Saturn", detriment: "Moon", keynote: "The power to discern the critical detail." },
        { line: 4, name: "The seducer", exalted: "Uranus", detriment: "Sun", keynote: "The ability to influence through focus." },
        { line: 5, name: "The faith healer", exalted: "Neptune", detriment: "Pluto", keynote: "The power of concentrated belief." },
        { line: 6, name: "The bulldozer", exalted: "Mars", detriment: "Earth", keynote: "The power to penetrate despite resistance." }
      ]
    },
    10: {
      number: 10,
      name: "Treading",
      chineseName: "\u5C65 Lu",
      center: "G",
      circuit: "Tribal",
      aminoAcid: "Thr",
      element: "Metal",
      binary: "110111",
      organ: "G Center",
      pressure: "Will",
      theme: "Behavior / Self-Love",
      lines: [
        { line: 1, name: "Modesty", exalted: "Moon", detriment: "Mercury", keynote: "The right action at the right time." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The avoidance of behavior that brings conflict." },
        { line: 3, name: "The martyr", exalted: "Mars", detriment: "Jupiter", keynote: "The sacrifice of self for the sake of others." },
        { line: 4, name: "The opportunist", exalted: "Sun", detriment: "Neptune", keynote: "The ability to take advantage of behavior." },
        { line: 5, name: "The dragon slayer", exalted: "Uranus", detriment: "Pluto", keynote: "The courage to confront the unacceptable." },
        { line: 6, name: "The role model", exalted: "Earth", detriment: "Venus", keynote: "The ultimate expression of behavior." }
      ]
    },
    11: {
      number: 11,
      name: "Peace",
      chineseName: "\u6CF0 Tai",
      center: "Ajna",
      circuit: "Collective",
      aminoAcid: "Asp",
      element: "Wood",
      binary: "111000",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Ideas / Inner Truth",
      lines: [
        { line: 1, name: "Attunement", exalted: "Jupiter", detriment: "Saturn", keynote: "The ability to sense the right idea." },
        { line: 2, name: "Rigidity", exalted: "Mars", detriment: "Uranus", keynote: "The fixed idea." },
        { line: 3, name: "The dreamer", exalted: "Neptune", detriment: "Pluto", keynote: "The power of imagination." },
        { line: 4, name: "The teacher", exalted: "Sun", detriment: "Moon", keynote: "The ability to communicate ideas." },
        { line: 5, name: "The philosopher", exalted: "Mercury", detriment: "Venus", keynote: "The love of wisdom." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The acceptance of limitations." }
      ]
    },
    12: {
      number: 12,
      name: "Standstill",
      chineseName: "\u5426 Pi",
      center: "Throat",
      circuit: "Tribal",
      aminoAcid: "Glu",
      element: "Earth",
      binary: "000111",
      organ: "Throat",
      pressure: "Heart",
      theme: "Caution / Standstill",
      lines: [
        { line: 1, name: "The monk", exalted: "Moon", detriment: "Jupiter", keynote: "The withdrawal from activity." },
        { line: 2, name: "The purist", exalted: "Venus", detriment: "Mars", keynote: "The refusal to compromise." },
        { line: 3, name: "The accuser", exalted: "Saturn", detriment: "Neptune", keynote: "The projection of blame." },
        { line: 4, name: "The prophet", exalted: "Uranus", detriment: "Sun", keynote: "The ability to foresee stagnation." },
        { line: 5, name: "The pragmatist", exalted: "Mercury", detriment: "Pluto", keynote: "The acceptance of standstill." },
        { line: 6, name: "The pessimist", exalted: "Earth", detriment: "Mars", keynote: "The expectation of failure." }
      ]
    },
    13: {
      number: 13,
      name: "Fellowship",
      chineseName: "\u540C\u4EBA Tong Ren",
      center: "G",
      circuit: "Individual",
      aminoAcid: "Asn",
      element: "Fire",
      binary: "101111",
      organ: "G Center",
      pressure: "Will",
      theme: "Listener / The Witness",
      lines: [
        { line: 1, name: "Empathy", exalted: "Moon", detriment: "Mercury", keynote: "The ability to feel what others feel." },
        { line: 2, name: "The bigot", exalted: "Mars", detriment: "Saturn", keynote: "The refusal to listen." },
        { line: 3, name: "The pessimist", exalted: "Jupiter", detriment: "Uranus", keynote: "The expectation of betrayal." },
        { line: 4, name: "The fatigued", exalted: "Sun", detriment: "Neptune", keynote: "The exhaustion from listening." },
        { line: 5, name: "The saviour", exalted: "Venus", detriment: "Pluto", keynote: "The desire to rescue." },
        { line: 6, name: "The optimist", exalted: "Earth", detriment: "Mars", keynote: "The expectation of fellowship." }
      ]
    },
    14: {
      number: 14,
      name: "Great Possession",
      chineseName: "\u5927\u6709 Da You",
      center: "Sacral",
      circuit: "Collective",
      aminoAcid: "Gln",
      element: "Air",
      binary: "111101",
      organ: "Sacrum",
      pressure: "Mind",
      theme: "Power Skills / Resources",
      lines: [
        { line: 1, name: "Money isn't everything", exalted: "Jupiter", detriment: "Saturn", keynote: "The recognition that resources are more than material." },
        { line: 2, name: "Management", exalted: "Mercury", detriment: "Mars", keynote: "The ability to allocate resources." },
        { line: 3, name: "The miser", exalted: "Venus", detriment: "Moon", keynote: "The hoarding of resources." },
        { line: 4, name: "The commander", exalted: "Sun", detriment: "Uranus", keynote: "The ability to direct resources." },
        { line: 5, name: "The liquidiser", exalted: "Neptune", detriment: "Pluto", keynote: "The ability to convert resources." },
        { line: 6, name: "The wallflower", exalted: "Earth", detriment: "Jupiter", keynote: "The refusal to use resources." }
      ]
    },
    15: {
      number: 15,
      name: "Modesty",
      chineseName: "\u8C26 Qian",
      center: "G",
      circuit: "Collective",
      aminoAcid: "Lys",
      element: "Earth",
      binary: "001000",
      organ: "G Center",
      pressure: "Heart",
      theme: "Extremes / Rhythm",
      lines: [
        { line: 1, name: "The egoist", exalted: "Moon", detriment: "Mercury", keynote: "The need for extremes." },
        { line: 2, name: "The discriminator", exalted: "Venus", detriment: "Saturn", keynote: "The ability to choose the extreme." },
        { line: 3, name: "The moderate", exalted: "Mars", detriment: "Jupiter", keynote: "The avoidance of extremes." },
        { line: 4, name: "Thealpha&omega", exalted: "Sun", detriment: "Neptune", keynote: "The acceptance of all extremes." },
        { line: 5, name: "Theslave", exalted: "Uranus", detriment: "Pluto", keynote: "The submission to extremes." },
        { line: 6, name: "The recluse", exalted: "Earth", detriment: "Mars", keynote: "The withdrawal from extremes." }
      ]
    },
    16: {
      number: 16,
      name: "Enthusiasm",
      chineseName: "\u8C6B Yu",
      center: "Throat",
      circuit: "Tribal",
      aminoAcid: "His",
      element: "Thunder",
      binary: "000100",
      organ: "Sacrum",
      pressure: "Mind",
      theme: "Skills / Enthusiasm",
      lines: [
        { line: 1, name: "The violinist", exalted: "Jupiter", detriment: "Saturn", keynote: "The love of skill." },
        { line: 2, name: "The cynic", exalted: "Mars", detriment: "Venus", keynote: "The distrust of enthusiasm." },
        { line: 3, name: "The independen", exalted: "Uranus", detriment: "Pluto", keynote: "The refusal to follow." },
        { line: 4, name: "The leader", exalted: "Sun", detriment: "Moon", keynote: "The ability to lead through skill." },
        { line: 5, name: "The achiever", exalted: "Mercury", detriment: "Neptune", keynote: "The drive for mastery." },
        { line: 6, name: "The shyster", exalted: "Earth", detriment: "Jupiter", keynote: "The manipulation of enthusiasm." }
      ]
    },
    17: {
      number: 17,
      name: "Following",
      chineseName: "\u968F Sui",
      center: "Ajna",
      circuit: "Collective",
      aminoAcid: "Arg",
      element: "Lake",
      binary: "100110",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Opinions / Views",
      lines: [
        { line: 1, name: "The receiver", exalted: "Moon", detriment: "Mercury", keynote: "The openness to opinion." },
        { line: 2, name: "The administrator", exalted: "Saturn", detriment: "Mars", keynote: "The ability to organize opinions." },
        { line: 3, name: "The opportunist", exalted: "Jupiter", detriment: "Uranus", keynote: "The use of opinion for gain." },
        { line: 4, name: "The analyst", exalted: "Sun", detriment: "Neptune", keynote: "The ability to deconstruct opinion." },
        { line: 5, name: "The speaker", exalted: "Venus", detriment: "Pluto", keynote: "The ability to articulate opinion." },
        { line: 6, name: "The collectivist", exalted: "Earth", detriment: "Jupiter", keynote: "The synthesis of opinion." }
      ]
    },
    18: {
      number: 18,
      name: "Decay",
      chineseName: "\u86CA Gu",
      center: "Spleen",
      circuit: "Collective",
      aminoAcid: "Trp",
      element: "Mountain",
      binary: "011001",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Correction / Integrity",
      lines: [
        { line: 1, name: "Conservatism", exalted: "Moon", detriment: "Saturn", keynote: "The resistance to change." },
        { line: 2, name: "The abdicator", exalted: "Venus", detriment: "Mars", keynote: "The refusal to correct." },
        { line: 3, name: "The zealot", exalted: "Jupiter", detriment: "Uranus", keynote: "The fanatical correction." },
        { line: 4, name: "The incompetent", exalted: "Sun", detriment: "Neptune", keynote: "The inability to correct." },
        { line: 5, name: "The psychiatrist", exalted: "Mercury", detriment: "Pluto", keynote: "The correction of the mind." },
        { line: 6, name: "The destroyer", exalted: "Earth", detriment: "Jupiter", keynote: "The destruction of the old." }
      ]
    },
    19: {
      number: 19,
      name: "Approach",
      chineseName: "\u4E34 Lin",
      center: "Root",
      circuit: "Tribal",
      aminoAcid: "Cys",
      element: "Earth",
      binary: "110000",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Wanting / Need",
      lines: [
        { line: 1, name: "The broker", exalted: "Jupiter", detriment: "Saturn", keynote: "The ability to mediate need." },
        { line: 2, name: "The vendor", exalted: "Mercury", detriment: "Mars", keynote: "The selling of need." },
        { line: 3, name: "The collectivist", exalted: "Venus", detriment: "Moon", keynote: "The sharing of need." },
        { line: 4, name: "The fanatic", exalted: "Uranus", detriment: "Sun", keynote: "The obsession with need." },
        { line: 5, name: "The negotiator", exalted: "Neptune", detriment: "Pluto", keynote: "The ability to bargain." },
        { line: 6, name: "The recluse", exalted: "Earth", detriment: "Mars", keynote: "The denial of need." }
      ]
    },
    20: {
      number: 20,
      name: "Contemplation",
      chineseName: "\u89C2 Guan",
      center: "Throat",
      circuit: "Integration",
      aminoAcid: "Arg",
      element: "Wind",
      binary: "000011",
      organ: "Throat",
      pressure: "Mind",
      theme: "Now / Awareness",
      lines: [
        { line: 1, name: "The hermit", exalted: "Moon", detriment: "Mercury", keynote: "The withdrawal into the now." },
        { line: 2, name: "The pragmatist", exalted: "Venus", detriment: "Saturn", keynote: "The acceptance of what is." },
        { line: 3, name: "The experimentalist", exalted: "Mars", detriment: "Jupiter", keynote: "The trial of the now." },
        { line: 4, name: "The Buddha", exalted: "Sun", detriment: "Neptune", keynote: "The transcendence of time." },
        { line: 5, name: "The realist", exalted: "Uranus", detriment: "Pluto", keynote: "The acceptance of limitation." },
        { line: 6, name: "The wizard", exalted: "Earth", detriment: "Mars", keynote: "The transcendence of the now." }
      ]
    },
    21: {
      number: 21,
      name: "Biting Through",
      chineseName: "\u566C\u55D1 Shi He",
      center: "Heart",
      circuit: "Tribal",
      aminoAcid: "Ser",
      element: "Fire",
      binary: "100101",
      organ: "Heart",
      pressure: "Will",
      theme: "Control / The Ego",
      lines: [
        { line: 1, name: "The wolf", exalted: "Jupiter", detriment: "Saturn", keynote: "The drive to control." },
        { line: 2, name: "The diplomat", exalted: "Venus", detriment: "Mars", keynote: "The gentle control." },
        { line: 3, name: "The anchorite", exalted: "Mercury", detriment: "Uranus", keynote: "The control of self." },
        { line: 4, name: "The dictator", exalted: "Sun", detriment: "Neptune", keynote: "The absolute control." },
        { line: 5, name: "The executive", exalted: "Pluto", detriment: "Moon", keynote: "The efficient control." },
        { line: 6, name: "The separatist", exalted: "Earth", detriment: "Jupiter", keynote: "The refusal of control." }
      ]
    },
    22: {
      number: 22,
      name: "Grace",
      chineseName: "\u8D32 Bi",
      center: "Throat",
      circuit: "Individual",
      aminoAcid: "Leu",
      element: "Mountain",
      binary: "101001",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Openness / Grace",
      lines: [
        { line: 1, name: "The second-class citizen", exalted: "Moon", detriment: "Mercury", keynote: "The acceptance of limitation." },
        { line: 2, name: "The mystic", exalted: "Neptune", detriment: "Saturn", keynote: "The power of silence." },
        { line: 3, name: "The enchanter", exalted: "Venus", detriment: "Mars", keynote: "The ability to seduce." },
        { line: 4, name: "The sensualist", exalted: "Jupiter", detriment: "Uranus", keynote: "The love of beauty." },
        { line: 5, name: "The director", exalted: "Sun", detriment: "Pluto", keynote: "The control of grace." },
        { line: 6, name: "The gentility", exalted: "Earth", detriment: "Mars", keynote: "The refinement of openness." }
      ]
    },
    23: {
      number: 23,
      name: "Splitting Apart",
      chineseName: "\u5265 Bo",
      center: "Throat",
      circuit: "Tribal",
      aminoAcid: "Ile",
      element: "Mountain",
      binary: "000001",
      organ: "Throat",
      pressure: "Mind",
      theme: "Assimilation / Explanation",
      lines: [
        { line: 1, name: "The boy scout", exalted: "Jupiter", detriment: "Saturn", keynote: "The naive explanation." },
        { line: 2, name: "The teacher", exalted: "Mercury", detriment: "Mars", keynote: "The ability to explain." },
        { line: 3, name: "The anarchist", exalted: "Uranus", detriment: "Pluto", keynote: "The refusal to explain." },
        { line: 4, name: "The assistant", exalted: "Venus", detriment: "Moon", keynote: "The support of explanation." },
        { line: 5, name: "The guru", exalted: "Sun", detriment: "Neptune", keynote: "The enlightened explanation." },
        { line: 6, name: "The windbag", exalted: "Earth", detriment: "Jupiter", keynote: "The excessive explanation." }
      ]
    },
    24: {
      number: 24,
      name: "Return",
      chineseName: "\u590D Fu",
      center: "Ajna",
      circuit: "Individual",
      aminoAcid: "Met",
      element: "Thunder",
      binary: "100000",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Rationalization / Return",
      lines: [
        { line: 1, name: "The pilgrim", exalted: "Moon", detriment: "Mercury", keynote: "The search for meaning." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal from rationalization." },
        { line: 3, name: "The addict", exalted: "Mars", detriment: "Jupiter", keynote: "The obsessive return." },
        { line: 4, name: "The opportunist", exalted: "Sun", detriment: "Uranus", keynote: "The use of return." },
        { line: 5, name: "The fatalist", exalted: "Neptune", detriment: "Pluto", keynote: "The acceptance of return." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The pragmatic return." }
      ]
    },
    25: {
      number: 25,
      name: "Innocence",
      chineseName: "\u65E0\u5984 Wu Wang",
      center: "G",
      circuit: "Individual",
      aminoAcid: "Phe",
      element: "Heaven",
      binary: "100111",
      organ: "Heart",
      pressure: "Will",
      theme: "Spirit of Self / Universal Love",
      lines: [
        { line: 1, name: "The exorcist", exalted: "Jupiter", detriment: "Saturn", keynote: "The removal of distortion." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The withdrawal from conditioning." },
        { line: 3, name: "The survivor", exalted: "Uranus", detriment: "Pluto", keynote: "The ability to withstand." },
        { line: 4, name: "The addict", exalted: "Sun", detriment: "Neptune", keynote: "The obsession with innocence." },
        { line: 5, name: "The pragmatist", exalted: "Mercury", detriment: "Moon", keynote: "The practical application of innocence." },
        { line: 6, name: "The fool", exalted: "Earth", detriment: "Jupiter", keynote: "The naive acceptance." }
      ]
    },
    26: {
      number: 26,
      name: "Great Taming",
      chineseName: "\u5927\u755C Da Chu",
      center: "Heart",
      circuit: "Tribal",
      aminoAcid: "Tyr",
      element: "Mountain",
      binary: "110001",
      organ: "Heart",
      pressure: "Will",
      theme: "The Egoist / Trickster",
      lines: [
        { line: 1, name: "A bird in the hand", exalted: "Moon", detriment: "Mercury", keynote: "The satisfaction with what one has." },
        { line: 2, name: "The strategist", exalted: "Venus", detriment: "Saturn", keynote: "The ability to plan." },
        { line: 3, name: "The adventurer", exalted: "Mars", detriment: "Uranus", keynote: "The risk-taking ego." },
        { line: 4, name: "The critic", exalted: "Sun", detriment: "Neptune", keynote: "The judgment of ego." },
        { line: 5, name: "The salesman", exalted: "Jupiter", detriment: "Pluto", keynote: "The ability to sell the ego." },
        { line: 6, name: "The authority", exalted: "Earth", detriment: "Mars", keynote: "The ultimate expression of ego." }
      ]
    },
    27: {
      number: 27,
      name: "Nourishment",
      chineseName: "\u9890 Yi",
      center: "Sacral",
      circuit: "Tribal",
      aminoAcid: "Gly",
      element: "Mountain",
      binary: "100001",
      organ: "Sacrum",
      pressure: "Heart",
      theme: "Caring / Selfishness",
      lines: [
        { line: 1, name: "The selfish", exalted: "Jupiter", detriment: "Saturn", keynote: "The need for self-care." },
        { line: 2, name: "The altruist", exalted: "Venus", detriment: "Mars", keynote: "The care for others." },
        { line: 3, name: "The hedonist", exalted: "Mercury", detriment: "Uranus", keynote: "The pursuit of pleasure." },
        { line: 4, name: "The administrator", exalted: "Sun", detriment: "Moon", keynote: "The management of care." },
        { line: 5, name: "The collectivist", exalted: "Neptune", detriment: "Pluto", keynote: "The shared care." },
        { line: 6, name: "The utilitarian", exalted: "Earth", detriment: "Jupiter", keynote: "The practical care." }
      ]
    },
    28: {
      number: 28,
      name: "Great Excess",
      chineseName: "\u5927\u8FC7 Da Guo",
      center: "Spleen",
      circuit: "Individual",
      aminoAcid: "Ala",
      element: "Lake",
      binary: "011110",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Game Player / Risk",
      lines: [
        { line: 1, name: "The gambler", exalted: "Moon", detriment: "Mercury", keynote: "The willingness to risk." },
        { line: 2, name: "The daredevil", exalted: "Mars", detriment: "Saturn", keynote: "The thrill of danger." },
        { line: 3, name: "The adventurer", exalted: "Jupiter", detriment: "Uranus", keynote: "The explorer of risk." },
        { line: 4, name: "The alchemist", exalted: "Sun", detriment: "Neptune", keynote: "The transformation of risk." },
        { line: 5, name: "The survivor", exalted: "Venus", detriment: "Pluto", keynote: "The endurance through risk." },
        { line: 6, name: "The blaze of glory", exalted: "Earth", detriment: "Mars", keynote: "The ultimate risk." }
      ]
    },
    29: {
      number: 29,
      name: "The Abysmal",
      chineseName: "\u574E Kan",
      center: "Sacral",
      circuit: "Collective",
      aminoAcid: "Val",
      element: "Water",
      binary: "010010",
      organ: "Sacrum",
      pressure: "Heart",
      theme: "Commitment / Perseverance",
      lines: [
        { line: 1, name: "The draftee", exalted: "Jupiter", detriment: "Saturn", keynote: "The reluctant commitment." },
        { line: 2, name: "The disciplinarian", exalted: "Venus", detriment: "Mars", keynote: "The structured commitment." },
        { line: 3, name: "The addict", exalted: "Mercury", detriment: "Uranus", keynote: "The obsessive commitment." },
        { line: 4, name: "The perfectionist", exalted: "Sun", detriment: "Moon", keynote: "The flawless commitment." },
        { line: 5, name: "The sensualist", exalted: "Neptune", detriment: "Pluto", keynote: "The pleasure of commitment." },
        { line: 6, name: "The escapee", exalted: "Earth", detriment: "Jupiter", keynote: "The refusal to commit." }
      ]
    },
    30: {
      number: 30,
      name: "The Clinging",
      chineseName: "\u79BB Li",
      center: "Solar Plexus",
      circuit: "Individual",
      aminoAcid: "Thr",
      element: "Fire",
      binary: "101101",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Feelings / Desire",
      lines: [
        { line: 1, name: "The pragmatist", exalted: "Moon", detriment: "Mercury", keynote: "The acceptance of desire." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal from desire." },
        { line: 3, name: "The experimentalist", exalted: "Mars", detriment: "Jupiter", keynote: "The trial of desire." },
        { line: 4, name: "The bon vivant", exalted: "Sun", detriment: "Uranus", keynote: "The enjoyment of desire." },
        { line: 5, name: "The accuser", exalted: "Pluto", detriment: "Neptune", keynote: "The blame of unfulfilled desire." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The pragmatic desire." }
      ]
    },
    31: {
      number: 31,
      name: "Influence",
      chineseName: "\u54B8 Xian",
      center: "Throat",
      circuit: "Collective",
      aminoAcid: "Asp",
      element: "Lake",
      binary: "001110",
      organ: "Throat",
      pressure: "Will",
      theme: "Leading / Influence",
      lines: [
        { line: 1, name: "The confesso", exalted: "Moon", detriment: "Mercury", keynote: "The admission of influence." },
        { line: 2, name: "The influencer", exalted: "Venus", detriment: "Saturn", keynote: "The natural leader." },
        { line: 3, name: "The manipulator", exalted: "Mars", detriment: "Uranus", keynote: "The covert leader." },
        { line: 4, name: "The evangelist", exalted: "Jupiter", detriment: "Neptune", keynote: "The missionary leader." },
        { line: 5, name: "The general", exalted: "Sun", detriment: "Pluto", keynote: "The commander." },
        { line: 6, name: "The philosopher", exalted: "Earth", detriment: "Mars", keynote: "The thought leader." }
      ]
    },
    32: {
      number: 32,
      name: "Duration",
      chineseName: "\u6052 Heng",
      center: "Spleen",
      circuit: "Tribal",
      aminoAcid: "Glu",
      element: "Thunder",
      binary: "011100",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Continuity / Instinct",
      lines: [
        { line: 1, name: "The conservative", exalted: "Moon", detriment: "Saturn", keynote: "The preservation of continuity." },
        { line: 2, name: "The mediator", exalted: "Venus", detriment: "Mars", keynote: "The negotiation of continuity." },
        { line: 3, name: "The pragmatist", exalted: "Mercury", detriment: "Jupiter", keynote: "The practical continuity." },
        { line: 4, name: "The opportunist", exalted: "Uranus", detriment: "Sun", keynote: "The use of continuity." },
        { line: 5, name: "The strategist", exalted: "Neptune", detriment: "Pluto", keynote: "The planned continuity." },
        { line: 6, name: "The destroyer", exalted: "Earth", detriment: "Mars", keynote: "The end of continuity." }
      ]
    },
    33: {
      number: 33,
      name: "Retreat",
      chineseName: "\u9041 Dun",
      center: "Throat",
      circuit: "Collective",
      aminoAcid: "Asn",
      element: "Heaven",
      binary: "001111",
      organ: "Throat",
      pressure: "Mind",
      theme: "Privacy / Retreat",
      lines: [
        { line: 1, name: "The hermit", exalted: "Moon", detriment: "Mercury", keynote: "The withdrawal into privacy." },
        { line: 2, name: "The snob", exalted: "Venus", detriment: "Saturn", keynote: "The elitist retreat." },
        { line: 3, name: "The adventurer", exalted: "Mars", detriment: "Uranus", keynote: "The retreat into experience." },
        { line: 4, name: "The teacher", exalted: "Jupiter", detriment: "Neptune", keynote: "The retreat into knowledge." },
        { line: 5, name: "The recluse", exalted: "Sun", detriment: "Pluto", keynote: "The absolute retreat." },
        { line: 6, name: "The switcher", exalted: "Earth", detriment: "Mars", keynote: "The changing of privacy." }
      ]
    },
    34: {
      number: 34,
      name: "Great Power",
      chineseName: "\u5927\u58EE Da Zhuang",
      center: "Sacral",
      circuit: "Integration",
      aminoAcid: "Gln",
      element: "Thunder",
      binary: "111100",
      organ: "Sacrum",
      pressure: "Will",
      theme: "Power / The Generator",
      lines: [
        { line: 1, name: "The bully", exalted: "Moon", detriment: "Mercury", keynote: "The misuse of power." },
        { line: 2, name: "The communicator", exalted: "Venus", detriment: "Saturn", keynote: "The expression of power." },
        { line: 3, name: "The explorer", exalted: "Mars", detriment: "Jupiter", keynote: "The adventurous power." },
        { line: 4, name: "The strategist", exalted: "Uranus", detriment: "Sun", keynote: "The planned power." },
        { line: 5, name: "The commander", exalted: "Neptune", detriment: "Pluto", keynote: "The directed power." },
        { line: 6, name: "The adept", exalted: "Earth", detriment: "Mars", keynote: "The mastery of power." }
      ]
    },
    35: {
      number: 35,
      name: "Progress",
      chineseName: "\u664B Jin",
      center: "Throat",
      circuit: "Collective",
      aminoAcid: "Lys",
      element: "Fire",
      binary: "000101",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Change / Progress",
      lines: [
        { line: 1, name: "The collector", exalted: "Jupiter", detriment: "Saturn", keynote: "The accumulation of experience." },
        { line: 2, name: "The communicator", exalted: "Venus", detriment: "Mars", keynote: "The sharing of experience." },
        { line: 3, name: "The experimenter", exalted: "Mercury", detriment: "Uranus", keynote: "The trial of change." },
        { line: 4, name: "The opportunist", exalted: "Sun", detriment: "Neptune", keynote: "The use of change." },
        { line: 5, name: "The philanthropist", exalted: "Moon", detriment: "Pluto", keynote: "The generous change." },
        { line: 6, name: "The adventurer", exalted: "Earth", detriment: "Jupiter", keynote: "The restless change." }
      ]
    },
    36: {
      number: 36,
      name: "The Darkening",
      chineseName: "\u660E\u5937 Ming Yi",
      center: "Solar Plexus",
      circuit: "Individual",
      aminoAcid: "His",
      element: "Earth",
      binary: "101000",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Crisis / Growth",
      lines: [
        { line: 1, name: "The spokesperson", exalted: "Moon", detriment: "Mercury", keynote: "The communication of crisis." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal from crisis." },
        { line: 3, name: "The activist", exalted: "Mars", detriment: "Jupiter", keynote: "The confrontation of crisis." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Uranus", keynote: "The resolution of crisis." },
        { line: 5, name: "The godfather", exalted: "Pluto", detriment: "Neptune", keynote: "The authority in crisis." },
        { line: 6, name: "The survivor", exalted: "Earth", detriment: "Mars", keynote: "The endurance of crisis." }
      ]
    },
    37: {
      number: 37,
      name: "The Family",
      chineseName: "\u5BB6\u4EBA Jia Ren",
      center: "Solar Plexus",
      circuit: "Tribal",
      aminoAcid: "Arg",
      element: "Wind",
      binary: "101011",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Friendship / Community",
      lines: [
        { line: 1, name: "The father", exalted: "Moon", detriment: "Saturn", keynote: "The authoritative friend." },
        { line: 2, name: "The mother", exalted: "Venus", detriment: "Mars", keynote: "The nurturing friend." },
        { line: 3, name: "The outsider", exalted: "Uranus", detriment: "Jupiter", keynote: "The estranged friend." },
        { line: 4, name: "The mediator", exalted: "Mercury", detriment: "Neptune", keynote: "The peacemaking friend." },
        { line: 5, name: "The leader", exalted: "Sun", detriment: "Pluto", keynote: "The commanding friend." },
        { line: 6, name: "The teacher", exalted: "Earth", detriment: "Mars", keynote: "The instructing friend." }
      ]
    },
    38: {
      number: 38,
      name: "Opposition",
      chineseName: "\u777D Kui",
      center: "Root",
      circuit: "Individual",
      aminoAcid: "Trp",
      element: "Fire",
      binary: "110101",
      organ: "Root",
      pressure: "Heart",
      theme: "The Fighter / Opposition",
      lines: [
        { line: 1, name: "The refuser", exalted: "Moon", detriment: "Mercury", keynote: "The rejection of opposition." },
        { line: 2, name: "The politician", exalted: "Venus", detriment: "Saturn", keynote: "The negotiation of opposition." },
        { line: 3, name: "The enthusiast", exalted: "Mars", detriment: "Uranus", keynote: "The embrace of opposition." },
        { line: 4, name: "The switcher", exalted: "Jupiter", detriment: "Neptune", keynote: "The changing of sides." },
        { line: 5, name: "The outsider", exalted: "Sun", detriment: "Pluto", keynote: "The permanent opposition." },
        { line: 6, name: "The reconciler", exalted: "Earth", detriment: "Mars", keynote: "The resolution of opposition." }
      ]
    },
    39: {
      number: 39,
      name: "Obstruction",
      chineseName: "\u8E47 Jian",
      center: "Root",
      circuit: "Individual",
      aminoAcid: "Cys",
      element: "Water",
      binary: "001010",
      organ: "Root",
      pressure: "Heart",
      theme: "Provocation / Challenge",
      lines: [
        { line: 1, name: "The dissembler", exalted: "Moon", detriment: "Saturn", keynote: "The hidden provocation." },
        { line: 2, name: "The confrontationalist", exalted: "Mars", detriment: "Venus", keynote: "The direct provocation." },
        { line: 3, name: "The diplomat", exalted: "Mercury", detriment: "Uranus", keynote: "The subtle provocation." },
        { line: 4, name: "The strategist", exalted: "Sun", detriment: "Neptune", keynote: "The planned provocation." },
        { line: 5, name: "The collectivist", exalted: "Jupiter", detriment: "Pluto", keynote: "The group provocation." },
        { line: 6, name: "The troubleshooter", exalted: "Earth", detriment: "Mars", keynote: "The resolution of provocation." }
      ]
    },
    40: {
      number: 40,
      name: "Deliverance",
      chineseName: "\u89E3 Jie",
      center: "Heart",
      circuit: "Tribal",
      aminoAcid: "Arg",
      element: "Thunder",
      binary: "010100",
      organ: "Heart",
      pressure: "Will",
      theme: "Aloneness / Deliverance",
      lines: [
        { line: 1, name: "The hermit", exalted: "Moon", detriment: "Mercury", keynote: "The withdrawal for deliverance." },
        { line: 2, name: "The exorcist", exalted: "Venus", detriment: "Saturn", keynote: "The removal of blockage." },
        { line: 3, name: "The pragmatist", exalted: "Mars", detriment: "Jupiter", keynote: "The practical deliverance." },
        { line: 4, name: "The organizer", exalted: "Sun", detriment: "Uranus", keynote: "The structured deliverance." },
        { line: 5, name: "The redeemer", exalted: "Neptune", detriment: "Pluto", keynote: "The spiritual deliverance." },
        { line: 6, name: "The deliverer", exalted: "Earth", detriment: "Mars", keynote: "The ultimate deliverance." }
      ]
    },
    41: {
      number: 41,
      name: "Decrease",
      chineseName: "\u635F Sun",
      center: "Root",
      circuit: "Collective",
      aminoAcid: "Ser",
      element: "Mountain",
      binary: "110010",
      organ: "Root",
      pressure: "Heart",
      theme: "Contraction / Fantasy",
      lines: [
        { line: 1, name: "The spender", exalted: "Moon", detriment: "Jupiter", keynote: "The reduction of resources." },
        { line: 2, name: "The borrower", exalted: "Venus", detriment: "Mars", keynote: "The use of others' resources." },
        { line: 3, name: "The miser", exalted: "Saturn", detriment: "Uranus", keynote: "The hoarding of resources." },
        { line: 4, name: "The student", exalted: "Mercury", detriment: "Neptune", keynote: "The reduction of knowing." },
        { line: 5, name: "The altruist", exalted: "Sun", detriment: "Pluto", keynote: "The generous decrease." },
        { line: 6, name: "The survivor", exalted: "Earth", detriment: "Mars", keynote: "The endurance through decrease." }
      ]
    },
    42: {
      number: 42,
      name: "Increase",
      chineseName: "\u76CA Yi",
      center: "Sacral",
      circuit: "Collective",
      aminoAcid: "Leu",
      element: "Wind",
      binary: "011001",
      organ: "Root",
      pressure: "Heart",
      theme: "Growth / Expansion",
      lines: [
        { line: 1, name: "The collector", exalted: "Moon", detriment: "Saturn", keynote: "The accumulation of growth." },
        { line: 2, name: "The investor", exalted: "Venus", detriment: "Mars", keynote: "The allocation of growth." },
        { line: 3, name: "The gambler", exalted: "Mars", detriment: "Jupiter", keynote: "The risky growth." },
        { line: 4, name: "The coordinator", exalted: "Sun", detriment: "Uranus", keynote: "The organized growth." },
        { line: 5, name: "The pragmatist", exalted: "Mercury", detriment: "Neptune", keynote: "The practical growth." },
        { line: 6, name: "The evaluator", exalted: "Earth", detriment: "Pluto", keynote: "The assessment of growth." }
      ]
    },
    43: {
      number: 43,
      name: "Breakthrough",
      chineseName: "\u592C Guai",
      center: "Ajna",
      circuit: "Individual",
      aminoAcid: "Ile",
      element: "Lake",
      binary: "111110",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Insight / Breakthrough",
      lines: [
        { line: 1, name: "The prophet", exalted: "Moon", detriment: "Mercury", keynote: "The anticipation of breakthrough." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal before breakthrough." },
        { line: 3, name: "The anarchist", exalted: "Mars", detriment: "Uranus", keynote: "The destructive breakthrough." },
        { line: 4, name: "The communicator", exalted: "Sun", detriment: "Neptune", keynote: "The sharing of insight." },
        { line: 5, name: "The saviour", exalted: "Jupiter", detriment: "Pluto", keynote: "The redemptive breakthrough." },
        { line: 6, name: "The manifesto", exalted: "Earth", detriment: "Mars", keynote: "The declaration of insight." }
      ]
    },
    44: {
      number: 44,
      name: "Coming to Meet",
      chineseName: "\u59E4 Gou",
      center: "Spleen",
      circuit: "Tribal",
      aminoAcid: "Met",
      element: "Heaven",
      binary: "011111",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Alertness / Meeting",
      lines: [
        { line: 1, name: "The vigilant", exalted: "Moon", detriment: "Saturn", keynote: "The watchful alertness." },
        { line: 2, name: "The seducer", exalted: "Venus", detriment: "Mars", keynote: "The attractive alertness." },
        { line: 3, name: "The assassin", exalted: "Mars", detriment: "Jupiter", keynote: "The predatory alertness." },
        { line: 4, name: "The opportunist", exalted: "Uranus", detriment: "Sun", keynote: "The timely alertness." },
        { line: 5, name: "The hunter", exalted: "Jupiter", detriment: "Neptune", keynote: "The pursuit of opportunity." },
        { line: 6, name: "The confessor", exalted: "Earth", detriment: "Pluto", keynote: "The honest alertness." }
      ]
    },
    45: {
      number: 45,
      name: "Gathering",
      chineseName: "\u8403 Cui",
      center: "Throat",
      circuit: "Tribal",
      aminoAcid: "Phe",
      element: "Lake",
      binary: "000110",
      organ: "Throat",
      pressure: "Will",
      theme: "Gathering / The King",
      lines: [
        { line: 1, name: "The chief", exalted: "Jupiter", detriment: "Saturn", keynote: "The leadership of gathering." },
        { line: 2, name: "The caretaker", exalted: "Venus", detriment: "Mars", keynote: "The nurturing gathering." },
        { line: 3, name: "The hoarder", exalted: "Mercury", detriment: "Uranus", keynote: "The accumulation of resources." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Neptune", keynote: "The negotiation of gathering." },
        { line: 5, name: "The benevolent", exalted: "Moon", detriment: "Pluto", keynote: "The generous gathering." },
        { line: 6, name: "The destroyer", exalted: "Earth", detriment: "Mars", keynote: "The dispersal of gathering." }
      ]
    },
    46: {
      number: 46,
      name: "Pushing Upward",
      chineseName: "\u5347 Sheng",
      center: "G",
      circuit: "Individual",
      aminoAcid: "Tyr",
      element: "Earth",
      binary: "010110",
      organ: "G Center",
      pressure: "Will",
      theme: "Determination / Pushing Up",
      lines: [
        { line: 1, name: "The collector", exalted: "Moon", detriment: "Mercury", keynote: "The accumulation before ascent." },
        { line: 2, name: "The pragmatist", exalted: "Venus", detriment: "Saturn", keynote: "The practical ascent." },
        { line: 3, name: "The orator", exalted: "Mars", detriment: "Jupiter", keynote: "The persuasive ascent." },
        { line: 4, name: "The student", exalted: "Sun", detriment: "Uranus", keynote: "The learning ascent." },
        { line: 5, name: "The climber", exalted: "Neptune", detriment: "Pluto", keynote: "The ambitious ascent." },
        { line: 6, name: "The departed", exalted: "Earth", detriment: "Mars", keynote: "The transcendent ascent." }
      ]
    },
    47: {
      number: 47,
      name: "Oppression",
      chineseName: "\u56F0 Kun",
      center: "Ajna",
      circuit: "Collective",
      aminoAcid: "Gly",
      element: "Lake",
      binary: "010011",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Transmutation / Realization",
      lines: [
        { line: 1, name: "The victim", exalted: "Moon", detriment: "Saturn", keynote: "The acceptance of oppression." },
        { line: 2, name: "The prisoner", exalted: "Venus", detriment: "Mars", keynote: "The confined realization." },
        { line: 3, name: "The architect", exalted: "Mercury", detriment: "Uranus", keynote: "The structural realization." },
        { line: 4, name: "The alchemist", exalted: "Sun", detriment: "Neptune", keynote: "The transformative realization." },
        { line: 5, name: "The historian", exalted: "Jupiter", detriment: "Pluto", keynote: "The retrospective realization." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The pragmatic realization." }
      ]
    },
    48: {
      number: 48,
      name: "The Well",
      chineseName: "\u4E95 Jing",
      center: "Spleen",
      circuit: "Collective",
      aminoAcid: "Ala",
      element: "Water",
      binary: "011010",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Depth / Solutions",
      lines: [
        { line: 1, name: "The inquisitor", exalted: "Moon", detriment: "Saturn", keynote: "The probing of depth." },
        { line: 2, name: "The collector", exalted: "Venus", detriment: "Mars", keynote: "The accumulation of solutions." },
        { line: 3, name: "The thinker", exalted: "Jupiter", detriment: "Uranus", keynote: "The contemplative solution." },
        { line: 4, name: "The teacher", exalted: "Sun", detriment: "Neptune", keynote: "The sharing of depth." },
        { line: 5, name: "The master", exalted: "Mercury", detriment: "Pluto", keynote: "The authoritative solution." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Jupiter", keynote: "The practical depth." }
      ]
    },
    49: {
      number: 49,
      name: "Revolution",
      chineseName: "\u9769 Ge",
      center: "Solar Plexus",
      circuit: "Tribal",
      aminoAcid: "Val",
      element: "Lake",
      binary: "101110",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Principles / Revolution",
      lines: [
        { line: 1, name: "The idealist", exalted: "Moon", detriment: "Mercury", keynote: "The theoretical revolution." },
        { line: 2, name: "The pragmatist", exalted: "Venus", detriment: "Saturn", keynote: "The practical revolution." },
        { line: 3, name: "The anarchist", exalted: "Mars", detriment: "Jupiter", keynote: "The destructive revolution." },
        { line: 4, name: "The negotiator", exalted: "Sun", detriment: "Uranus", keynote: "The diplomatic revolution." },
        { line: 5, name: "The collectivist", exalted: "Jupiter", detriment: "Pluto", keynote: "The group revolution." },
        { line: 6, name: "The reactionary", exalted: "Earth", detriment: "Mars", keynote: "The conservative revolution." }
      ]
    },
    50: {
      number: 50,
      name: "The Cauldron",
      chineseName: "\u9F0E Ding",
      center: "Spleen",
      circuit: "Tribal",
      aminoAcid: "Thr",
      element: "Fire",
      binary: "011101",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Values / The Law",
      lines: [
        { line: 1, name: "The cook", exalted: "Jupiter", detriment: "Saturn", keynote: "The preparation of values." },
        { line: 2, name: "The dissembler", exalted: "Venus", detriment: "Mars", keynote: "The hidden values." },
        { line: 3, name: "The addict", exalted: "Mercury", detriment: "Uranus", keynote: "The obsessive values." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Neptune", keynote: "The negotiation of values." },
        { line: 5, name: "The priest", exalted: "Moon", detriment: "Pluto", keynote: "The spiritual values." },
        { line: 6, name: "The destroyer", exalted: "Earth", detriment: "Jupiter", keynote: "The end of values." }
      ]
    },
    51: {
      number: 51,
      name: "The Arousing",
      chineseName: "\u9707 Zhen",
      center: "Heart",
      circuit: "Individual",
      aminoAcid: "Asp",
      element: "Thunder",
      binary: "100100",
      organ: "Heart",
      pressure: "Will",
      theme: "Shock / Initiation",
      lines: [
        { line: 1, name: "The sentinel", exalted: "Moon", detriment: "Saturn", keynote: "The watchful shock." },
        { line: 2, name: "The monk", exalted: "Venus", detriment: "Mars", keynote: "The contemplative shock." },
        { line: 3, name: "The pursuer", exalted: "Mars", detriment: "Jupiter", keynote: "The active shock." },
        { line: 4, name: "The mediator", exalted: "Mercury", detriment: "Uranus", keynote: "The negotiation of shock." },
        { line: 5, name: "The thrill seeker", exalted: "Sun", detriment: "Pluto", keynote: "The love of shock." },
        { line: 6, name: "The fatalist", exalted: "Earth", detriment: "Neptune", keynote: "The acceptance of shock." }
      ]
    },
    52: {
      number: 52,
      name: "Keeping Still",
      chineseName: "\u826E Gen",
      center: "Root",
      circuit: "Collective",
      aminoAcid: "Glu",
      element: "Mountain",
      binary: "001001",
      organ: "Root",
      pressure: "Heart",
      theme: "Stillness / Concentration",
      lines: [
        { line: 1, name: "The thinker", exalted: "Moon", detriment: "Mercury", keynote: "The contemplative stillness." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal into stillness." },
        { line: 3, name: "The pragmatist", exalted: "Mars", detriment: "Jupiter", keynote: "The practical stillness." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Uranus", keynote: "The negotiation of stillness." },
        { line: 5, name: "The strategist", exalted: "Jupiter", detriment: "Pluto", keynote: "The planned stillness." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The acceptance of limitation." }
      ]
    },
    53: {
      number: 53,
      name: "Development",
      chineseName: "\u6E10 Jian",
      center: "Root",
      circuit: "Collective",
      aminoAcid: "Asn",
      element: "Wind",
      binary: "001011",
      organ: "Root",
      pressure: "Heart",
      theme: "Evolution / Beginning",
      lines: [
        { line: 1, name: "The pioneer", exalted: "Jupiter", detriment: "Saturn", keynote: "The initiation of development." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The solitary development." },
        { line: 3, name: "The adventurer", exalted: "Mars", detriment: "Uranus", keynote: "The risky development." },
        { line: 4, name: "The opportunist", exalted: "Sun", detriment: "Neptune", keynote: "The use of development." },
        { line: 5, name: "The pragmatist", exalted: "Mercury", detriment: "Pluto", keynote: "The practical development." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Jupiter", keynote: "The acceptance of evolution." }
      ]
    },
    54: {
      number: 54,
      name: "The Marrying Maiden",
      chineseName: "\u5F52\u59B9 Gui Mei",
      center: "Root",
      circuit: "Collective",
      aminoAcid: "Gln",
      element: "Thunder",
      binary: "110100",
      organ: "Root",
      pressure: "Heart",
      theme: "Ambition / Drive",
      lines: [
        { line: 1, name: "The influence", exalted: "Moon", detriment: "Mercury", keynote: "The guided ambition." },
        { line: 2, name: "The dissembler", exalted: "Venus", detriment: "Saturn", keynote: "The hidden ambition." },
        { line: 3, name: "The opportunist", exalted: "Mars", detriment: "Jupiter", keynote: "The use of ambition." },
        { line: 4, name: "The alpha&omega", exalted: "Sun", detriment: "Uranus", keynote: "The total ambition." },
        { line: 5, name: "The seducer", exalted: "Neptune", detriment: "Pluto", keynote: "The attractive ambition." },
        { line: 6, name: "The politician", exalted: "Earth", detriment: "Mars", keynote: "The strategic ambition." }
      ]
    },
    55: {
      number: 55,
      name: "Abundance",
      chineseName: "\u4E30 Feng",
      center: "Solar Plexus",
      circuit: "Individual",
      aminoAcid: "Lys",
      element: "Thunder",
      binary: "101100",
      organ: "Solar Plexus",
      pressure: "Heart",
      theme: "Spirit / Abundance",
      lines: [
        { line: 1, name: "The merchant", exalted: "Moon", detriment: "Saturn", keynote: "The commercial spirit." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The withdrawal from abundance." },
        { line: 3, name: "The experimenter", exalted: "Uranus", detriment: "Pluto", keynote: "The trial of spirit." },
        { line: 4, name: "The alchemist", exalted: "Sun", detriment: "Neptune", keynote: "The transformation of abundance." },
        { line: 5, name: "The benevolent", exalted: "Jupiter", detriment: "Mercury", keynote: "The generous spirit." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The pragmatic abundance." }
      ]
    },
    56: {
      number: 56,
      name: "The Wanderer",
      chineseName: "\u65C5 Lu",
      center: "Throat",
      circuit: "Collective",
      aminoAcid: "His",
      element: "Fire",
      binary: "001101",
      organ: "Throat",
      pressure: "Mind",
      theme: "Stimulation / Wanderer",
      lines: [
        { line: 1, name: "The novelty seeker", exalted: "Jupiter", detriment: "Saturn", keynote: "The love of the new." },
        { line: 2, name: "The pragmatist", exalted: "Venus", detriment: "Mars", keynote: "The practical travel." },
        { line: 3, name: "The adventurer", exalted: "Mars", detriment: "Uranus", keynote: "The risky travel." },
        { line: 4, name: "The student", exalted: "Mercury", detriment: "Neptune", keynote: "The learning travel." },
        { line: 5, name: "The sage", exalted: "Sun", detriment: "Pluto", keynote: "The wise travel." },
        { line: 6, name: "The exile", exalted: "Earth", detriment: "Jupiter", keynote: "The permanent travel." }
      ]
    },
    57: {
      number: 57,
      name: "The Gentle",
      chineseName: "\u5DFD Xun",
      center: "Spleen",
      circuit: "Individual",
      aminoAcid: "Arg",
      element: "Wind",
      binary: "011011",
      organ: "Spleen",
      pressure: "Heart",
      theme: "Intuition / Clarity",
      lines: [
        { line: 1, name: "The confessor", exalted: "Moon", detriment: "Mercury", keynote: "The honest intuition." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The solitary intuition." },
        { line: 3, name: "The acupuncturist", exalted: "Mars", detriment: "Jupiter", keynote: "The precise intuition." },
        { line: 4, name: "The artist", exalted: "Sun", detriment: "Uranus", keynote: "The creative intuition." },
        { line: 5, name: "The utilitarian", exalted: "Jupiter", detriment: "Pluto", keynote: "The practical intuition." },
        { line: 6, name: "The utilitarian", exalted: "Earth", detriment: "Neptune", keynote: "The application of clarity." }
      ]
    },
    58: {
      number: 58,
      name: "The Joyous",
      chineseName: "\u5151 Dui",
      center: "Root",
      circuit: "Tribal",
      aminoAcid: "Trp",
      element: "Lake",
      binary: "110110",
      organ: "Root",
      pressure: "Heart",
      theme: "Joy / The Stimulator",
      lines: [
        { line: 1, name: "The inventor", exalted: "Moon", detriment: "Saturn", keynote: "The creation of joy." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The solitary joy." },
        { line: 3, name: "The hedonist", exalted: "Mars", detriment: "Jupiter", keynote: "The pursuit of joy." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Uranus", keynote: "The sharing of joy." },
        { line: 5, name: "The sensualist", exalted: "Neptune", detriment: "Pluto", keynote: "The pleasure of joy." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The acceptance of limitation." }
      ]
    },
    59: {
      number: 59,
      name: "Dispersion",
      chineseName: "\u6DA3 Huan",
      center: "Sacral",
      circuit: "Tribal",
      aminoAcid: "Cys",
      element: "Wind",
      binary: "010111",
      organ: "Sacrum",
      pressure: "Heart",
      theme: "Sexuality / Intimacy",
      lines: [
        { line: 1, name: "The libertine", exalted: "Moon", detriment: "Mercury", keynote: "The unrestrained sexuality." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal from sexuality." },
        { line: 3, name: "The experimenter", exalted: "Mars", detriment: "Uranus", keynote: "The trial of sexuality." },
        { line: 4, name: "The bon vivant", exalted: "Jupiter", detriment: "Neptune", keynote: "The enjoyment of sexuality." },
        { line: 5, name: "The polygamist", exalted: "Sun", detriment: "Pluto", keynote: "The multiple intimacies." },
        { line: 6, name: "The puritan", exalted: "Earth", detriment: "Mars", keynote: "The restriction of sexuality." }
      ]
    },
    60: {
      number: 60,
      name: "Limitation",
      chineseName: "\u8282 Jie",
      center: "Root",
      circuit: "Collective",
      aminoAcid: "Arg",
      element: "Water",
      binary: "110010",
      organ: "Root",
      pressure: "Heart",
      theme: "Limitation / Acceptance",
      lines: [
        { line: 1, name: "The acceptor", exalted: "Jupiter", detriment: "Saturn", keynote: "The embrace of limitation." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The withdrawal from limitation." },
        { line: 3, name: "The pragmatist", exalted: "Mercury", detriment: "Uranus", keynote: "The practical limitation." },
        { line: 4, name: "The opportunist", exalted: "Sun", detriment: "Neptune", keynote: "The use of limitation." },
        { line: 5, name: "The mediator", exalted: "Moon", detriment: "Pluto", keynote: "The negotiation of limitation." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Jupiter", keynote: "The acceptance of reality." }
      ]
    },
    61: {
      number: 61,
      name: "Inner Truth",
      chineseName: "\u4E2D\u5B5A Zhong Fu",
      center: "Head",
      circuit: "Collective",
      aminoAcid: "Ser",
      element: "Wind",
      binary: "110011",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Mystery / Inner Truth",
      lines: [
        { line: 1, name: "The listener", exalted: "Moon", detriment: "Saturn", keynote: "The receptivity to truth." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The solitary truth." },
        { line: 3, name: "The believer", exalted: "Jupiter", detriment: "Uranus", keynote: "the faith in truth." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Neptune", keynote: "The negotiation of truth." },
        { line: 5, name: "The mystic", exalted: "Mercury", detriment: "Pluto", keynote: "The intuitive truth." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The pragmatic truth." }
      ]
    },
    62: {
      number: 62,
      name: "Small Excess",
      chineseName: "\u5C0F\u8FC7 Xiao Guo",
      center: "Throat",
      circuit: "Collective",
      aminoAcid: "Leu",
      element: "Thunder",
      binary: "001100",
      organ: "Throat",
      pressure: "Mind",
      theme: "Detail / Caution",
      lines: [
        { line: 1, name: "The accountant", exalted: "Moon", detriment: "Mercury", keynote: "The precise detail." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Saturn", keynote: "The withdrawal from detail." },
        { line: 3, name: "The experimenter", exalted: "Mars", detriment: "Jupiter", keynote: "The trial of detail." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Uranus", keynote: "The negotiation of detail." },
        { line: 5, name: "The master", exalted: "Jupiter", detriment: "Pluto", keynote: "the authoritative detail." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Neptune", keynote: "The practical detail." }
      ]
    },
    63: {
      number: 63,
      name: "After Completion",
      chineseName: "\u65E2\u6D4E Ji Ji",
      center: "Head",
      circuit: "Collective",
      aminoAcid: "Ile",
      element: "Water",
      binary: "101010",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Doubt / Completion",
      lines: [
        { line: 1, name: "The doubter", exalted: "Moon", detriment: "Saturn", keynote: "The suspicion of completion." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The withdrawal after completion." },
        { line: 3, name: "The saboteur", exalted: "Mars", detriment: "Uranus", keynote: "The destruction of completion." },
        { line: 4, name: "The meditator", exalted: "Sun", detriment: "Neptune", keynote: "The contemplation of completion." },
        { line: 5, name: "The survivor", exalted: "Jupiter", detriment: "Pluto", keynote: "The endurance through completion." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mercury", keynote: "The acceptance of imperfection." }
      ]
    },
    64: {
      number: 64,
      name: "Before Completion",
      chineseName: "\u672A\u6D4E Wei Ji",
      center: "Head",
      circuit: "Tribal",
      aminoAcid: "Met",
      element: "Fire",
      binary: "010101",
      organ: "Ajna",
      pressure: "Mind",
      theme: "Confusion / Incompletion",
      lines: [
        { line: 1, name: "The dreamer", exalted: "Moon", detriment: "Saturn", keynote: "The vision of completion." },
        { line: 2, name: "The hermit", exalted: "Venus", detriment: "Mars", keynote: "The withdrawal before completion." },
        { line: 3, name: "The enthusiast", exalted: "Mars", detriment: "Jupiter", keynote: "The drive for completion." },
        { line: 4, name: "The mediator", exalted: "Sun", detriment: "Uranus", keynote: "The negotiation of completion." },
        { line: 5, name: "The achiever", exalted: "Neptune", detriment: "Pluto", keynote: "The attainment of completion." },
        { line: 6, name: "The realist", exalted: "Earth", detriment: "Mars", keynote: "The acceptance of incompletion." }
      ]
    }
  };
  var GATE_BODY_REGION = {
    1: "head",
    2: "head",
    7: "head",
    13: "head",
    25: "head",
    46: "head",
    8: "throat",
    12: "throat",
    16: "throat",
    20: "throat",
    22: "throat",
    23: "throat",
    31: "throat",
    33: "throat",
    35: "throat",
    43: "throat",
    45: "throat",
    56: "throat",
    62: "throat",
    21: "chest",
    26: "chest",
    40: "chest",
    51: "chest",
    52: "spine",
    5: "sacral",
    9: "sacral",
    14: "sacral",
    29: "sacral",
    34: "sacral",
    42: "sacral",
    50: "pelvis",
    38: "legs",
    39: "legs",
    41: "legs",
    53: "legs",
    54: "legs"
  };
  function getJulianDay(date) {
    const Y = date.getUTCFullYear();
    const M = date.getUTCMonth() + 1;
    const D = date.getUTCDate() + date.getUTCHours() / 24 + date.getUTCMinutes() / 1440 + date.getUTCSeconds() / 86400;
    const A = Math.floor((14 - M) / 12);
    const y = Y + 4800 - A;
    const m = M + 12 * A - 3;
    return D + Math.floor((153 * m + 2) / 5) + 365 * y + Math.floor(y / 4) - Math.floor(y / 100) + Math.floor(y / 400) - 32045;
  }
  function getSunLongitude(julianDay) {
    const n3 = julianDay - 2451545;
    const L0 = (280.46 + 0.9856474 * n3) % 360;
    const g = (357.528 + 0.9856003 * n3) % 360 * (Math.PI / 180);
    const lambda = (L0 + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)) % 360;
    return lambda < 0 ? lambda + 360 : lambda;
  }
  function getMoonLongitude(julianDay) {
    const n3 = julianDay - 2451545;
    const L = (218.316 + 13.176396 * n3) % 360;
    const M = (134.963 + 13.064993 * n3) % 360;
    const F = (93.272 + 13.22935 * n3) % 360;
    const Mrad = M * Math.PI / 180;
    const Frad = F * Math.PI / 180;
    const lambda = L + 6.289 * Math.sin(Mrad) + 1.274 * Math.sin(2 * Mrad - Frad) + 0.658 * Math.sin(2 * Mrad) + 0.214 * Math.sin(2 * Frad);
    return (lambda % 360 + 360) % 360;
  }
  function longitudeToHD(longitude) {
    const normalized = ((longitude - 58) % 360 + 360) % 360;
    const gateIndex = Math.floor(normalized / DEG_PER_GATE);
    const gate2 = GATE_WHEEL[gateIndex % 64];
    const degInGate = normalized % DEG_PER_GATE;
    const line = Math.min(6, Math.floor(degInGate / DEG_PER_LINE) + 1);
    const degInLine = degInGate % DEG_PER_LINE;
    const color = Math.min(6, Math.floor(degInLine / DEG_PER_COLOR) + 1);
    const degInColor = degInLine % DEG_PER_COLOR;
    const tone = Math.min(6, Math.floor(degInColor / DEG_PER_TONE) + 1);
    const degInTone = degInColor % DEG_PER_TONE;
    const base = Math.min(5, Math.floor(degInTone / DEG_PER_BASE) + 1);
    return { gate: gate2, line, color, tone, base, degInGate, degInLine, degInColor };
  }
  function longitudeToZodiac(longitude) {
    const signs = [
      "Aries",
      "Taurus",
      "Gemini",
      "Cancer",
      "Leo",
      "Virgo",
      "Libra",
      "Scorpio",
      "Sagittarius",
      "Capricorn",
      "Aquarius",
      "Pisces"
    ];
    const deg = longitude % 360 / 30;
    const signIndex = Math.floor(deg);
    const d = (deg - signIndex) * 30;
    const degree = Math.floor(d);
    const minute = Math.floor((d - degree) * 60);
    const second = Math.floor(((d - degree) * 60 - minute) * 60);
    return { sign: signs[signIndex], degree, minute, second };
  }
  function getNodeLongitude(julianDay) {
    const n3 = julianDay - 2451545;
    const omega = (125.04 - 0.052954 * n3) % 360;
    const north = (omega % 360 + 360) % 360;
    return { north, south: (north + 180) % 360 };
  }
  function getPlanetLongitude(planet, jd) {
    if (planet === "Sun") return getSunLongitude(jd);
    if (planet === "Moon") return getMoonLongitude(jd);
    const data = {
      Mercury: { period: 87.97, epoch: 252.25 },
      Venus: { period: 224.7, epoch: 181.98 },
      Mars: { period: 686.98, epoch: 355.43 },
      Jupiter: { period: 4332.59, epoch: 34.35 },
      Saturn: { period: 10759.22, epoch: 50.08 },
      Uranus: { period: 30688.5, epoch: 314.55 },
      Neptune: { period: 60182, epoch: 304.35 },
      Pluto: { period: 90465, epoch: 238.93 }
    };
    if (planet === "Earth") {
      return (getSunLongitude(jd) + 180) % 360;
    }
    if (data[planet]) {
      const d = jd - 2451545;
      const lon = (data[planet].epoch + d / data[planet].period * 360) % 360;
      return (lon + 360) % 360;
    }
    if (planet === "North Node" || planet === "South Node") {
      const nodes = getNodeLongitude(jd);
      return planet === "North Node" ? nodes.north : nodes.south;
    }
    return 0;
  }
  function calculateHumanDesign(birthDate, birthTime, _birthLocation) {
    const [hours, minutes] = birthTime.split(":").map(Number);
    const date = new Date(birthDate);
    date.setUTCHours(hours, minutes, 0, 0);
    const jd = getJulianDay(date);
    const designOffsetDays = -88;
    const jdDesign = jd + designOffsetDays;
    const planets = [
      "Sun",
      "Earth",
      "Moon",
      "North Node",
      "South Node",
      "Mercury",
      "Venus",
      "Mars",
      "Jupiter",
      "Saturn",
      "Uranus",
      "Neptune",
      "Pluto"
    ];
    const placements = planets.map((planet) => {
      const lon = getPlanetLongitude(planet, jd);
      const hd = longitudeToHD(lon);
      const zod = longitudeToZodiac(lon);
      return {
        planet,
        gate: hd.gate,
        line: hd.line,
        color: hd.color,
        tone: hd.tone,
        base: hd.base,
        longitude: lon,
        zodiac: zod.sign,
        degree: zod.degree,
        minute: zod.minute,
        second: zod.second
      };
    });
    const designPlacements = planets.map((planet) => {
      const lon = getPlanetLongitude(planet, jdDesign);
      const hd = longitudeToHD(lon);
      const zod = longitudeToZodiac(lon);
      return {
        planet,
        gate: hd.gate,
        line: hd.line,
        color: hd.color,
        tone: hd.tone,
        base: hd.base,
        longitude: lon,
        zodiac: zod.sign,
        degree: zod.degree,
        minute: zod.minute,
        second: zod.second
      };
    });
    const consciousSun = placements.find((p) => p.planet === "Sun");
    const consciousEarth = placements.find((p) => p.planet === "Earth");
    const consciousMoon = placements.find((p) => p.planet === "Moon");
    const designSun = designPlacements.find((p) => p.planet === "Sun");
    const designEarth = designPlacements.find((p) => p.planet === "Earth");
    const designMoon = designPlacements.find((p) => p.planet === "Moon");
    const profile = `${consciousSun.line}/${designSun.line}`;
    const allGates = /* @__PURE__ */ new Set();
    for (const p of placements) allGates.add(p.gate);
    for (const p of designPlacements) allGates.add(p.gate);
    const definedCenters = {
      Head: false,
      Ajna: false,
      Throat: false,
      G: false,
      Heart: false,
      "Solar Plexus": false,
      Sacral: false,
      Spleen: false,
      Root: false
    };
    for (const gate2 of allGates) {
      const center = GATE_CENTER[gate2];
      if (center) definedCenters[center] = true;
    }
    const hasSacral = definedCenters.Sacral;
    const hasEmo = definedCenters["Solar Plexus"];
    const hasSpleen = definedCenters.Spleen;
    const hasHeart = definedCenters.Heart;
    const hasRoot = definedCenters.Root;
    const hasG = definedCenters.G;
    const hasThroat = definedCenters.Throat;
    let type;
    if (!hasSacral && !hasEmo && !hasSpleen && !hasHeart && !hasRoot && !hasG && !hasThroat) {
      type = "Reflector";
    } else if (hasSacral && hasThroat && (hasEmo || hasHeart || hasSacral)) {
      type = "Manifesting Generator";
    } else if (hasSacral) {
      type = "Generator";
    } else if (hasThroat && (hasHeart || hasEmo || hasRoot || hasSpleen)) {
      type = "Manifestor";
    } else {
      type = "Projector";
    }
    let authority;
    if (type === "Reflector") authority = "Lunar";
    else if (hasEmo) authority = "Emotional";
    else if (hasSacral) authority = "Sacral";
    else if (hasSpleen) authority = "Splenic";
    else if (hasHeart) authority = "Ego";
    else if (hasG && hasThroat) authority = "Self-Projected";
    else if (definedCenters.Ajna || definedCenters.Head) authority = "Mental";
    else authority = "Lunar";
    const definedCount = Object.values(definedCenters).filter(Boolean).length;
    let definition;
    if (definedCount === 0) definition = "No";
    else if (definedCount <= 2) definition = "Single";
    else if (definedCount <= 3) definition = "Split";
    else if (definedCount <= 5) definition = "Triple Split";
    else definition = "Quadruple";
    const crossGates = [consciousSun.gate, designSun.gate, consciousEarth.gate, designEarth.gate];
    const gateNames = crossGates.map((g) => GATES[g]?.name ?? `Gate ${g}`);
    const incarnationCross = `Cross of ${gateNames[0]} & ${gateNames[1]}`;
    const profileFirst = consciousSun.line;
    const variable = {
      digestion: profileFirst <= 2 ? "Cold" : "Hot",
      environment: profileFirst <= 3 ? "Natural" : "Urban",
      motivation: profileFirst <= 4 ? "Need" : "Desire",
      perspective: profileFirst <= 5 ? "Personal" : "Transpersonal"
    };
    return {
      type,
      authority,
      profile,
      definition,
      incarnationCross,
      variable,
      centers: definedCenters,
      channels: [],
      // Channels require exact connections; simplified for now
      placements: [...placements, ...designPlacements],
      consciousSun,
      consciousEarth,
      designSun,
      designEarth,
      consciousMoon,
      designMoon,
      nodes: {
        north: placements.find((p) => p.planet === "North Node"),
        south: placements.find((p) => p.planet === "South Node")
      }
    };
  }
  var TYPE_ARCHETYPE = {
    Generator: { name: "Resonant Engine", purpose: "To respond to life and build sustainable energy through correct work" },
    "Manifesting Generator": { name: "Acceleration Matrix", purpose: "To find the fastest path and iterate through multiple passions" },
    Projector: { name: "Guidance Lens", purpose: "To see systems clearly and direct energy where it flows best" },
    Manifestor: { name: "Initiation Spark", purpose: "To catalyze change and birth new forms through independent action" },
    Reflector: { name: "Mirroring Prism", purpose: "To reflect the health of systems and reveal what is hidden" }
  };
  function generateAgentBlueprint(chart) {
    const typeInfo = TYPE_ARCHETYPE[chart.type];
    const qualities = [];
    const seenGates = /* @__PURE__ */ new Set();
    for (const p of chart.placements) {
      const gateData = GATES[p.gate];
      if (!gateData) continue;
      if (seenGates.has(p.gate)) continue;
      seenGates.add(p.gate);
      const lineData = gateData.lines[p.line - 1];
      qualities.push({
        category: `${p.planet} (${gateData.center})`,
        name: `${gateData.name} ${p.gate}.${p.line}`,
        description: lineData?.keynote ?? gateData.theme,
        intensity: 0.7 + p.line / 6 * 0.3,
        gate: p.gate,
        line: p.line
      });
    }
    const bodyRegions = [];
    for (const gate2 of seenGates) {
      const region = GATE_BODY_REGION[gate2] ?? "core";
      bodyRegions.push({
        action: region,
        mind: gate2 <= 32 ? "head" : "gut",
        emotion: GATE_CENTER[gate2] ?? "core"
      });
    }
    const resonance = [];
    const discordance = [];
    switch (chart.type) {
      case "Generator":
        resonance.push("Responds to clear questions", "Sustained energy output", "Satisfaction through completion");
        discordance.push("Initiating without prompt", "Mental decision-making", "Frustration from incomplete work");
        break;
      case "Manifesting Generator":
        resonance.push("Rapid iteration", "Parallel processing", "Informing before action");
        discordance.push("Skipping steps", "Not informing others", "Anger from interruption");
        break;
      case "Projector":
        resonance.push("Being recognized and invited", "Guiding energy efficiently", "One-on-one mastery");
        discordance.push("Pushing for opportunities", "Working without rest", "Bitterness from non-recognition");
        break;
      case "Manifestor":
        resonance.push("Freedom to initiate", "Informing to reduce resistance", "Independent creation");
        discordance.push("Being controlled", "Not informing", "Anger from suppression");
        break;
      case "Reflector":
        resonance.push("Healthy environments", "Lunar decision cycles", "Mirroring others accurately");
        discordance.push("Pressure to decide quickly", "Toxic spaces", "Disappointment from haste");
        break;
    }
    return {
      chart,
      archetype: typeInfo.name,
      purpose: typeInfo.purpose,
      qualities,
      resonance,
      discordance,
      bodyRegions
    };
  }
  var HD_SOURCE = "combined_agent_os_v2(3)/Kimi_Agent_Human Design Agent Builder/app/src/lib/humanDesign.ts";
  var HD_PROVENANCE = Object.freeze({
    gateWheel: claim(GATE_WHEEL, {
      status: CLAIM_STATUS.SOURCE_STATEMENT,
      source: HD_SOURCE + " GATE_WHEEL (Rave Mandala order; executable anchor 58.0 deg \u2014 see HD_WHEEL_ANCHOR_VARIANTS)"
    }),
    gateCenter: claim(GATE_CENTER, {
      status: CLAIM_STATUS.SOURCE_STATEMENT,
      source: HD_SOURCE + " GATE_CENTER (complete 64-gate -> center map; fills the 12 gates our CENTERS leaves unassigned)"
    }),
    gates: claim(GATES, {
      status: CLAIM_STATUS.SOURCE_STATEMENT,
      source: HD_SOURCE + " GATES (names, lines, exalted/detriment, amino-acid/organ/pressure attributions \u2014 donor attributions, not independently attested)"
    }),
    gateBodyRegion: claim(GATE_BODY_REGION, {
      status: CLAIM_STATUS.SOURCE_STATEMENT,
      source: HD_SOURCE + ' GATE_BODY_REGION ("from organism body.pdf", per donor comment; 36 gates covered)'
    })
  });
  var HD_WHEEL_ANCHOR_VARIANTS = Object.freeze([
    claim({ anchorDegrees: 58, gate41At: "58.0 deg tropical (0 deg Gemini region)" }, {
      status: CLAIM_STATUS.CONFLICT,
      source: HD_SOURCE + " longitudeToHD() executable code (primary behavior of this port)",
      evidence: { kind: "wheel-anchor", note: "code subtracts 58.0 before wheel lookup" }
    }),
    claim({ anchorDegrees: 311.75, gate41At: "~311.75 deg tropical (0 deg Aquarius)" }, {
      status: CLAIM_STATUS.CONFLICT,
      source: HD_SOURCE + " header comment on GATE_WHEEL (contradicts its own code)",
      evidence: { kind: "wheel-anchor", note: "donor comment vs donor code disagreement" }
    }),
    claim({ anchorDegrees: 46.75, gate41At: "46 deg 45 min (start of Fuxi-complement half-circle)" }, {
      status: CLAIM_STATUS.CONFLICT,
      source: "SynthAi.CompleteSuite/SynthAi.HumanDesign/MandalaGeometry.cs (see state-space/correspondences.js MANDALA_GEOMETRY)",
      evidence: { kind: "wheel-anchor", note: "algorithmic derivation from Fuxi complements; independent of this module" }
    })
  ]);
  var HD_GATE_BINARY_CONFLICTS = Object.freeze(
    Array.from({ length: 64 }, (_, i) => i + 1).filter((g) => GATES[g].binary !== gatePattern2(g).join("")).map((g) => Object.freeze({
      gate: g,
      ours: gatePattern2(g).join(""),
      donor: GATES[g].binary,
      claim: Object.freeze(claim(
        { gate: g, donorBinary: GATES[g].binary, primaryBinary: gatePattern2(g).join("") },
        {
          status: CLAIM_STATUS.CONFLICT,
          source: HD_SOURCE + " GATES[" + g + "].binary vs src/merged/kingwen.js gatePattern(" + g + ") (canonically verified: ours matches trigram composition)"
        }
      ))
    }))
  );
  var HD_GATE_CENTER_VARIANTS = (() => {
    const oursMap = {};
    for (const [center, gates] of Object.entries(CENTERS)) {
      for (const g of gates) (oursMap[g] ||= []).push(center);
    }
    const conflicts = [];
    const gapFills = [];
    for (let g = 1; g <= 64; g++) {
      const donor = GATE_CENTER[g].replace("Solar Plexus", "Solar");
      const ours = oursMap[g] || [];
      if (ours.length === 0) {
        gapFills.push(Object.freeze({ gate: g, donorCenter: GATE_CENTER[g] }));
      } else if (!ours.includes(donor) || ours.length > 1) {
        conflicts.push(Object.freeze({
          gate: g,
          ours: Object.freeze([...ours]),
          donor: GATE_CENTER[g],
          claim: Object.freeze(claim(
            { gate: g, donorCenter: GATE_CENTER[g], primaryCenters: [...ours] },
            {
              status: CLAIM_STATUS.CONFLICT,
              source: HD_SOURCE + " GATE_CENTER[" + g + "] vs src/merged/centers-channels.js CENTERS (ours primary; donor variant preserved)"
            }
          ))
        }));
      }
    }
    return Object.freeze({ conflicts: Object.freeze(conflicts), gapFills: Object.freeze(gapFills) });
  })();
  function activeChannels(chart) {
    const gates = new Set((chart.placements || []).map((p) => p.gate));
    return CANONICAL_CHANNELS.filter(([a, b]) => gates.has(a) && gates.has(b)).map(([a, b]) => `${a}-${b}`);
  }
  function donorCenterForGate(gate2) {
    if (!Number.isInteger(gate2) || gate2 < 1 || gate2 > 64) throw new RangeError("gate must be 1..64.");
    return GATE_CENTER[gate2] ?? null;
  }
  var human_design_default = Object.freeze({
    GATE_WHEEL,
    GATE_CENTER,
    GATES,
    GATE_BODY_REGION,
    getSunLongitude,
    getMoonLongitude,
    longitudeToHD,
    longitudeToZodiac,
    calculateHumanDesign,
    generateAgentBlueprint,
    HD_PROVENANCE,
    HD_WHEEL_ANCHOR_VARIANTS,
    HD_GATE_BINARY_CONFLICTS,
    HD_GATE_CENTER_VARIANTS,
    activeChannels,
    donorCenterForGate
  });

  // entry.mjs
  var MODS = { "kimi-addressing-dms": addressing_exports, "kimi-human-design": human_design_exports };
  var svc = new AddressService();
  svc._load = async (name) => {
    if (!MODS[name]) throw new Error("provider not bundled on device: " + name);
    return MODS[name];
  };
  var FIELDS = ADDRESS_FIELDS;
  function format13(a) {
    if (!a) throw new AddressRequired("address required");
    return FIELDS.map((f) => a[f] === null || a[f] === void 0 || a[f] === "" ? "-" : String(a[f])).join(".");
  }
  function parse13(s) {
    const p = String(s ?? "").trim().split(".");
    if (p.length !== 13) throw new AddressRequired("address must have exactly 13 fields (planetary.dimension.gate.line.color.tone.base.degree.minute.second.arc.zodiac.house), got " + p.length);
    const a = {};
    FIELDS.forEach((f, i) => {
      const v = p[i];
      if (v === "") throw new AddressRequired(f + " is empty: use - when a level does not exist");
      a[f] = v === "-" ? null : /^-?\d+$/.test(v) ? +v : v;
    });
    return a;
  }
  function gate13(s, input) {
    const a = parse13(s);
    const core = ["gate", "line", "color", "tone", "base"];
    if (core.some((k) => a[k] === null)) throw new AddressRequired("gate.line.color.tone.base cannot be dashes for a gated action");
    const g = gate(core.map((k) => a[k]).join("."), input);
    return { address: format13(a), core: g.address, index: g.index };
  }
  function canWrite(perms, containerId, who) {
    const sp = perms.spaces?.[containerId] ?? perms.default;
    if ((sp.residents || []).includes(who)) return { ok: true, why: who + " is resident of " + containerId };
    if (sp.resonance?.enabled && sp.resonance.criteria) return { ok: false, why: "resonance criteria present but not implemented on device (not defined by Alexis yet)" };
    return { ok: false, why: who + " is not resident of " + containerId + " and resonance is off/unset" };
  }
  async function intake(arcSecond, container) {
    const r = await svc.resolveAddress({ arcSecond });
    const a = { ...r.address, planetary: container?.planetary ?? null, dimension: container?.dimension ?? null };
    const st = state(["gate", "line", "color", "tone", "base"].map((k) => a[k]).join("."));
    if (st.unresolved) throw new AddressRequired("state-space did not resolve");
    return { address: a, key: format13(a), gateNames: st.gateState?.names || [], toneName: st.toneState?.name || null };
  }
  var session;
  function resetSession() {
    session = makeDefaultSession("device");
    return session.id;
  }
  function meshCycle(addr13, who, perms, containerId, input) {
    const g = gate13(addr13, input);
    const w = canWrite(perms, containerId, who);
    if (!w.ok) throw new AddressRequired("write refused: " + w.why);
    session ??= makeDefaultSession("device");
    const r = runMeshCycle(session, input, { address: g.address });
    return { address: g.address, core: g.core, runId: r.runId, summary: r.result.summary, events: r.result.events.length, runs: session.runs.length };
  }
  function safe(fn) {
    try {
      const v = fn();
      return JSON.stringify({ ok: true, v });
    } catch (e) {
      return JSON.stringify({ ok: false, err: (e && e.name) + ": " + (e && e.message) });
    }
  }
  return __toCommonJS(entry_exports);
})();
