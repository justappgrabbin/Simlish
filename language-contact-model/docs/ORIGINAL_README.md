# Klein Language Contact Model (KLC) — Complete Synthesis

A modern synthesis of **Sheldon Klein's entire computational linguistics toolkit (1963-2002)**, enhanced with neural embeddings, transformer-based generation, and LLM hybrid integration to compete with contemporary language models.

## What Klein Actually Built

Sheldon Klein at the University of Wisconsin-Madison created a toolkit that prefigured modern NLP in remarkable ways:

### 1. Computational Grammar Coder (1963)
**Paper:** *"A Computational Approach to Grammatical Coding of English Words"* — JACM, July 1963

- **30 grammatical classes** (expandable to hundreds)
- **Suffix Test 1:** Inflectional morphology (-s, -ing, -ed, Latin endings)
- **Suffix Test 2:** Derivational morphology (reverse-alphabetized: -tion, -ly, -ize, etc.)
- **Context Frame Test:** Triadic index (left_code × middle_count × right_code) with binary search
- **Logical multiplication** of test outputs for final code determination
- **Formal 5-tuple mathematical description** of the coding process
- **Speed:** 1250 words/minute on IBM 7090, 90%+ accuracy
- **First component** of the Protosynthex question-answering system

### 2. AUTOLING (1968)
**Paper:** *"AUTOLING: An Automatic Program Writer"*

- Meta-linguistic NLP system configurable to model **any theoretical linguistic framework**
- Transformational grammar, case grammar, stratificational grammar, or custom frameworks
- **Parse in one grammar, generate in another** — unique capability
- Grammar comparison across frameworks

### 3. DISEMINER (1968)
**Paper:** *"DISEMINER: A Distributional Semantics Inference Maker"*

- Answers **"Is it possible that...?"** through semantic plausibility inference
- **Warshall transitive closure** over dependency matrices
- **M = T* + T* × I** formula (transitive paths + final intransitive link)
- Distributional criteria for questions broader than text context
- **Anomaly detection:** "Each case of a positive answer violating an informant's view of the possible is data for analysis"

### 4. MESSY (1971-1976)
**Paper:** *"MESSY: A Meta-Symbolic Simulation System"*

- **Semantic triples** (O-R-O) with time-indexed assertions
- **Complex predicate nodes** — self-referential, 2nd-order predicate calculus power
- **Probabilistic rules** with subrule conditions (AND/OR/NOT)
- **Natural language meta-compiling:** A mother says "Don't talk to strangers unless..." and the system compiles this to behavioral rules
- Dynamic class creation and modification
- **Characters can speak new rules into existence**

### 5. Style Control (1965)
**Paper:** *"Control of Style with a Generative Grammar"*

- **Weighted probabilistic grammar** for stylistic mimicry
- Mathematical weight control over rule frequencies
- Deterministic, reproducible, explainable style transfer
- Syntactic weights, vocabulary bias, pronoun frequency, sentence length parameters

### 6. Automatic Novel Writing (1973)
**Paper:** *"Automatic Novel Writing: A Status Report"* — UWCS TR 183

- **2100-word murder mysteries** in <19 seconds
- Semantic deep structure + surface structure generation
- Plot generation from semantic simulation
- Character behavior governed by simulation rules
- **Production rules:** S → NP VP // O-R (Object-Relation mapping)

### 7. Historical Language Change (1966)
**Paper:** *"Historical Change in Language Using Monte Carlo Techniques"*

- **Speech community simulation** with individual grammars
- Birth/death cycles, rule borrowing, frequency modification
- **"Every grammar of every living speaker would be different"**
- Social structure, prestige, geography, learning heuristics
- **Key insight:** "The formal properties of natural languages make language change during the transmission process a logically unavoidable consequence"

### 8. Language Contact (1974)
**Paper:** *"Computer Simulation of Language Contact Models"*

- **Multi-lingual community dynamics**
- Rule adoption probability: function of learning heuristics, source status, exposure frequency
- Parent-child bonus, spouse bonus, prestige bonus, geographic barrier
- Dialect formation, mutual intelligibility, change trajectory

### 9. Propp & Lévi-Strauss (1976)
**Paper:** *"Simulation d'hypothèses émises par Propp et Lévi-Strauss"* — Informatique et Sciences Humaines

- **60 different fairy tales** generated at 128 words/second on Univac 1110
- **Propp's 31 functions** implemented as simulation rule groups
- **Lévi-Strauss's canonical formula:** f_x(a) : f_y(b) :: f_x(b) : f_a^{-1}(y)
- Double-effect transformations (structural inversions)
- Mythemes modeled as semantic triple transformations
- Complete deep structure + surface structure

### 10. Analogical Foundations (1999/2002)
**Paper:** *"Analogical Foundations of Creativity in Language, Culture & the Arts"*

- **Boolean groups** for analogical reasoning
- **Klein-4 groups:** {e, a, b, ab} where a² = b² = (ab)² = e
- **ATOs:** Analogical Transformation Operators
- **Strong equivalence:** *ab = *ba (commutative)
- **Self-inverse:** *a*ab = b
- Boolean vectors with XOR addition and sign multiplication

### 11. Automatic Paraphrasing (1976)
**Paper:** *"Automatic Paraphrasing in Essay Format"*

- Essay-length discourse generation
- Paraphrasing with semantic preservation
- Style-controlled reformulation
- Academic and formal register generation

## Architecture

```
KleinLanguageContactModel
├── Grammar Coder (1963) — JACM foundational parser
│   ├── Suffix Test 1 (inflectional)
│   ├── Suffix Test 2 (derivational, reverse-alphabetized)
│   ├── Context Frame Test (triadic index)
│   └── Logical multiplication
├── AUTOLING (1968) — Meta-linguistic engine
│   ├── TransformationalGrammar
│   ├── CaseGrammar
│   └── StratificationalGrammar
├── DISEMINER (1968) — Distributional semantics inference
│   ├── Warshall transitive closure
│   ├── Dependency matrices (T, I, M)
│   └── Vector semantic index
├── MESSY (1971-1976) — Meta-symbolic simulation
│   ├── Semantic network (time-indexed triples)
│   ├── Probabilistic rule evaluation
│   └── Natural language meta-compiling
├── Novel Writer (1973) — Automatic discourse generation
│   ├── Plot generation
│   ├── Semantic simulation
│   └── Style control
├── Language Contact Simulator (1966/1974)
│   ├── Multi-agent simulation
│   ├── Grammar learning
│   └── Dialect evolution
├── Propp Myth Generator (1976)
│   ├── 31 functions as rule groups
│   ├── Lévi-Strauss transformations
│   └── Mythological narrative generation
├── Analogical Engine (1999/2002)
│   ├── Klein-4 Boolean groups
│   └── ATO transformations
└── Chat & Research Interface (Modern)
    ├── Natural language chat
    ├── Deep research with DISEMINER
    ├── Meta-compiling
    └── LLM hybrid integration
```

## Usage

### 1. Chat & Research

```typescript
import { KleinLanguageContactModel, KLCChatInterface } from 'klein-language-contact-model';

const klc = new KleinLanguageContactModel({
  embeddingDimension: 768,
  enableNeuralEmbeddings: true,
  enableLLMHybrid: true,
  llmApiKey: process.env.OPENAI_API_KEY,
  llmModel: 'gpt-4'
});

const chat = new KLCChatInterface(klc, {
  apiKey: process.env.OPENAI_API_KEY,
  model: 'gpt-4',
  endpoint: 'https://api.openai.com/v1'
});

const session = chat.createSession();

// Research query
const response = await chat.chat(session.id, 
  "Is it possible that language contact accelerates grammatical simplification?"
);
// Uses DISEMINER transitive closure + LLM synthesis

// Meta-compiling
const rule = await chat.chat(session.id,
  "Don't talk to strangers unless someone you know is with you"
);
// Compiles to probabilistic behavioral rules in MESSY

// Generate story
const story = await chat.chat(session.id,
  "Write a 500-word murder mystery about a detective at a house party"
);
// Uses semantic deep structure + style control
```

### 2. Grammar Coding

```typescript
import { ComputationalGrammarCoder } from 'klein-language-contact-model';

const coder = new ComputationalGrammarCoder();

const analysis = coder.codeSentence(
  "The quick brown fox jumps over the lazy dog."
);

// Output:
// The: ART (dictionary)
// quick: ADJ (suffix_test_2: -ick)
// brown: NOUN/ADJ (capitalization)
// fox: NOUN (suffix_test_1: singular)
// jumps: VERB (suffix_test_1: -s)
// over: PREP (dictionary)
// the: ART (dictionary)
// lazy: ADJ (suffix_test_2: -y)
// dog: NOUN (suffix_test_1: singular)
```

### 3. Language Contact Simulation

```typescript
const result = await klc.simulateCommunity({
  population: 100,
  languages: ['english', 'spanish'],
  socialClasses: [
    { name: 'prestige', size: 20, influence: 0.8 },
    { name: 'working', size: 80, influence: 0.3 }
  ]
}, 100); // 100 years

console.log(result.dialects);
console.log(result.intelligibilityMatrix);
console.log(result.diversity); // "Every grammar different"
```

### 4. Myth Generation

```typescript
import { ProppMythGenerator, MESSY, AUTOLING } from 'klein-language-contact-model';

const generator = new ProppMythGenerator(new MESSY({}), new AUTOLING());

const myth = generator.generateMyth({
  title: 'The Hero and the Dragon',
  culture: 'slavic',
  moves: 1
});

// Propp's 31 functions applied as simulation rules
// Lévi-Strauss transformations for structural depth
```

### 5. Analogical Reasoning

```typescript
const results = await klc.analogicalReasoning(
  "boy_loves_light",
  "mythology"
);

// Klein-4 Boolean group operations:
// *ab = strong equivalence
// *a*ab = b (self-inverse)
// Returns: [{ concept: "icarus_sun", confidence: 0.87, ... }]
```

### 6. Style Transfer

```typescript
const transferred = await klc.styleTransfer(
  "Hey, what's up? Wanna grab coffee?",
  informalStyle,
  academicStyle
);

// Output: "Would you be interested in meeting for a discussion over coffee?"
// Mathematical weight control, not prompt engineering
```

## How This Competes With LLMs

| Capability | LLMs | KLC |
|------------|------|-----|
| Text generation | ✅ Statistical patterns | ✅ Semantic deep structure |
| Style transfer | ✅ Prompt-based | ✅ Mathematical weight control |
| Reasoning | ⚠️ Hallucination-prone | ✅ Transitive closure inference |
| Language change | ❌ Not possible | ✅ Monte Carlo simulation |
| Grammar learning | ❌ Not possible | ✅ Individual rule adoption |
| Meta-compiling | ❌ Not possible | ✅ Natural language → rules |
| Analogical creativity | ⚠️ Pattern matching | ✅ Boolean group operations |
| Explainability | ⚠️ Black box | ✅ Semantic network trace |
| Determinism | ❌ Non-deterministic | ✅ Reproducible from seed |
| Grammar coding | ❌ Not available | ✅ 30-class morphological analysis |
| Myth generation | ⚠️ Pattern matching | ✅ Propp's 31 functions + Lévi-Strauss |
| Historical linguistics | ❌ Not possible | ✅ Speech community simulation |
| Paraphrasing | ✅ Statistical | ✅ Semantic preservation + style control |

## Files

| File | What It Contains | Klein Paper |
|------|------------------|-------------|
| `grammar-coder.ts` | JACM 1963 computational grammar coding | Klein & Simmons 1963 |
| `semantic-network-core.ts` | MESSY's semantic network (time-indexed triples) | MESSY 1971-1976 |
| `diseminer-engine.ts` | Warshall transitive closure inference | DISEMINER 1968 |
| `messy-simulator.ts` | Meta-symbolic simulation with meta-compiling | MESSY 1971-1976 |
| `autoling-engine.ts` | Plugin-based grammar engine | AUTOLING 1968 |
| `novel-writer.ts` | Semantic discourse generation | Novel Writing 1973 |
| `language-contact-simulator.ts` | Monte Carlo speech community simulation | Historical Change 1966 |
| `propp-myth-generator.ts` | Propp's 31 functions + Lévi-Strauss transformations | Propp & Lévi-Strauss 1976 |
| `analogical-engine.ts` | Boolean group ATOs for creativity | Analogical Foundations 1999/2002 |
| `klein-language-contact-model-main.ts` | Main integration class | — |
| `klc-chat-interface.ts` | Chat + research + LLM hybrid | — |
| `index.ts` | Master export with examples | — |

## References

1. Klein, S. & Simmons, R.F. (1963). "A Computational Approach to Grammatical Coding of English Words." *JACM*, July 1963.
2. Klein, S. (1965). "Control of Style with a Generative Grammar." *Language*.
3. Klein, S. (1966). "Historical Change in Language Using Monte Carlo Techniques." *Mechanical Translation*.
4. Klein, S., Lieman, M., & Lindstrom, G. (1968). "DISEMINER: A Distributional Semantics Inference Maker." *UWCS Technical Report*.
5. Klein, S. (1968). "AUTOLING: An Automatic Program Writer." *UWCS Technical Report*.
6. Klein, S. (1971-1976). "MESSY: A Meta-Symbolic Simulation System." *UWCS Technical Reports*.
7. Klein, S. (1973). "Automatic Novel Writing: A Status Report." *UWCS Technical Report 183*.
8. Klein, S. (1974). "Computer Simulation of Language Contact Models." *Anthropological Linguistics*.
9. Klein, S., et al. (1976). "Simulation d'hypothèses émises par Propp et Lévi-Strauss." *Informatique et Sciences Humaines*, 28.
10. Klein, S. (1999/2002). "Analogical Foundations of Creativity in Language, Culture & the Arts." *UWCS Technical Report*.
11. Klein, S. (1976). "Automatic Paraphrasing in Essay Format." *UWCS Technical Report*.

## License

MIT — See LICENSE file for details.
