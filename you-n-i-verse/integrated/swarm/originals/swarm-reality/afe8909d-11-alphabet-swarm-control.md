# 11 — Alphabet, syntax, and a way of control, next to an alphabet swarm that builds like the I Ching

*Researched 2026-10-03 (PT). Lookup only. No code, no simulation, and no new build. Alexis's phrases are kept as she states them: "alphabet swarm", "way of control", and "build like the I Ching". Where a published idea is set beside one of those phrases, it is a candidate analogy, not her meaning. Claims are **Confirmed**, **Disputed**, or **Unconfirmed**. Human Design material is not used here. No birth data.*

## Verdict

- There is a plain linguistics distinction that can be stated without renaming her phrases. An **alphabet** is a finite set of symbols. A **syntax** is a finite set of rules that combine those symbols. With the right rules, a finite alphabet yields an unbounded set of strings. The I Ching does not do that. It stacks two symbols to a fixed depth of 6 and stops at 64 figures, then (in divination) flips one or more lines to another of those 64.
- The closest published meaning of "a way of control" for a grammar is not her phrase. It is a **control set** (also called a control language): a second language, over the names of the rules, that decides which rules may fire and in what order. Without it, the alphabet's rules run whenever they match. With it, they do not. That device is **Ginsburg and Spanier (1968)**, surveyed in **Dassow and Păun (1989)**.
- Nothing found uses her three phrases together. No control method was chosen for her.
- Ji (1999), as already read from her copy in file 04, is one existing bridge and only that: **L** in the cell-language 6-tuple is an alphabet of **protosemata**, and he applies **double articulation** (meaningless units combining into meaningful units). This note does not add a layer he is not already recorded as stating.
- Real "swarm + grammar" work exists (swarm grammars; colonies of grammars; L-systems, including control sets on growth tables). It gives control devices of its own. It does not say what an "alphabet swarm" is, and it does not build like the I Ching. The phrase "alphabet swarm" was not found as a published term (**Unconfirmed** as a technical term; not found).

## What was read, and what was not changed

- Files already in this folder, used as context and not rewritten: `09-stage-b-linguistics-frameworks.md` (phonology, morphology, syntax/Merge), `04-citations.md` (Ji 1999), `01-iching-line-relations.md` and the index line for `00` (line relations and her primary books). The 3% / 97% figures in Ji stay **human DNA only**, as file 04 says. They are not part of the alphabet definition below.
- Another of her agents, Lil Trinity (id `a7ce42c9-72ad-4372-a241-284d17a13e2f`), has a profile whose task is a "SOURCE-DIRECTED COMPUTATIONAL LANGUAGE CONTACT MODEL": a computational language-contact simulation grounded in Sheldon Klein. That is a different task. It was not edited. A search of `/workspace` also finds existing swarm and language-contact code (for example under `drive-pieces/` and `contact-automata/`). Those files are context only. They are not sources for the claims below.
- Klein's own publication list includes **Klein, S. (1965). "Control of Style with a Generative Grammar." *Language* 41: 619–631**, and **Klein, S. (1996). "Grammars, the I Ching and Levi-Strauss…" *Journal of Quantitative Anthropology* 6: 263–271**. Both titles are **Confirmed** as bibliography, from Klein's selected-publications page at the University of Wisconsin (fetched 2026-10-03). Neither paper was read. The 1965 title is a different published use of the word "control" (style inside one generative grammar). It is not a control set, and it is not her phrase "way of control". The 1996 title puts "grammars" next to "the I Ching" in Klein's work. What the paper actually argues is **Unconfirmed** here.

## 1. Alphabet and syntax

Two different uses of "alphabet" have to stay apart.

- In ordinary writing, an alphabet is a set of letters used to write sounds. That is not what formal language theory means.
- In formal language theory, an **alphabet** is any finite set of symbols. A **language** is a set of finite strings made from that alphabet. A **grammar** is a finite device that lists which strings are in the language. Chomsky's wording, **Confirmed** from the opening of the 1959 paper: "A language is a collection of sentences of finite length all constructed from a finite alphabet (or, where our concern is limited to syntax, a finite vocabulary) of symbols." Because the languages of interest are infinite, the object of study is the finite grammar that enumerates them. Source: Chomsky, N. (1959). "On certain formal properties of grammars." *Information and Control* 2: 137–167. DOI 10.1016/S0019-9958(59)90362-6. The same ranking is introduced in Chomsky, N. (1956). "Three models for the description of language." *IRE Transactions on Information Theory* 2(3): 113–124. DOI 10.1109/TIT.1956.1056813. Both already logged in file 09.

The usual parts of a grammar, as in Hopcroft and Ullman, *Introduction to Automata Theory, Languages, and Computation* (Addison-Wesley, 1979), which Dassow's survey (below) cites as the standard textbook: a finite set of nonterminal symbols, a finite **terminal alphabet**, a finite set of rules, and a start symbol. **Confirmed** as the definition that survey assumes. The 1979 book was not re-read this pass.

**How a finite alphabet plus rules produces an unbounded set.** The alphabet stays finite. The rules may be used again on their own output. Chomsky's 1959 point is that a finite grammar can have an infinite language as its range. A one-line illustration, not a system of hers: alphabet {a, b}, rules that add one symbol and then stop, can list every finite string. Nothing in the alphabet itself sets a maximum length. The rules do, or they do not.

**Syntax, including Merge.** In the narrow sense, syntax is those rules of combination, as against the alphabet they combine. In generative linguistics the current structure-building rule is **Merge**: take two objects already built and form one new object from them, and repeat. A finite stock of atoms and one repeatable operation yield unbounded hierarchical structure. Sources, already **Confirmed** in files 04 and 09: Chomsky, N. (1995). *The Minimalist Program*. MIT Press. Marcolli, M., Chomsky, N., and Berwick, R. C. (2023), "Mathematical Structure of Syntactic Merge," arXiv:2305.18278; book, MIT Press, 2025. A later statement of the same "finite means, infinite output" point was fetched this pass: Chomsky's "Merge and the Strong Minimalist Thesis" (PDF on a Caltech site, citing Berwick and Chomsky 2016 for the "basic property"). The PDF was read for that claim. Its publication details beyond the text of the PDF were not re-checked (**Unconfirmed** as a full bibliographic record; the claim in the PDF is **Confirmed** as what that file says).

**The Chomsky hierarchy** ranks rule systems by how restricted the rules are, not by how big the alphabet is: regular (finite-state), context-free, context-sensitive, unrestricted. The same finite alphabet can sit under any of the four, depending on the rules. File 09 already cites Chomsky 1956 and 1959 for this. A later overview that says the same hierarchy in plain words, fetched this pass: Jäger, G. and Rogers, J. (2012). "Formal language theory: refining the Chomsky hierarchy." *Philosophical Transactions of the Royal Society B* 367: 1956–1970. PMC3367686. **Confirmed** (open full text via PMC).

What this offers for "a way of control" over an alphabet: by itself, nothing extra. The rules *are* the control, in the weak sense that they say which combinations occur. They do not stand outside the grammar and choose among rules. That stronger device is section 2.

## 2. "Control" as a technical term in formal language theory

This section is the closest published meaning of "a way of control" for a grammar. It is **not** her term. Label: **control set / control language / regulated rewriting**.

**Plain version.** Give every rule a name. A derivation is then a sequence of those names. A control set is a second language whose words are the allowed sequences. A derivation counts only if its sequence of rule names is in that second language. The alphabet and the rules are still there. They no longer fire just because a symbol matches. A second sequence has to permit them.

**Where that definition is in print.**

- **Ginsburg, S. and Spanier, E. H. (1968). "Control Sets on Grammars." *Mathematical Systems Theory* 2(2): 159–177.** DOI 10.1007/BF01692513. **Confirmed** as a bibliographic record (researchr, DBLP-style listings, and later surveys). The 1968 paper itself was not re-opened this pass. The working definition used here is the one in the next item, which presents itself as that tradition.
- **Dassow, J. "Grammars with Regulated Rewriting"** (chapter PDF, Otto von Guericke University Magdeburg; fetched in full this pass). He defines a **regularly controlled grammar** as an ordinary context-free grammar plus a regular set R over the rule names. A terminal string is in the language only if it has a derivation whose rule-name string lies in R. His example: the rules can freely write letters, but R forces the two sides of the string to receive the same letter at each step, so the grammar produces { ww | w is a nonempty string of a's and b's }, which context-free rules alone cannot. **Confirmed** from that PDF. He cites the book below as the monograph.
- **Dassow, J. and Păun, Gh. (1989). *Regulated Rewriting in Formal Language Theory*. Berlin: Springer.** **Confirmed** as the book those surveys cite. A publisher preview of the book (fetched) says matrix grammars were defined by Abraham in 1965, and that about twenty-five further restrictions followed, "the most well known" being programmed, random context, indexed, scattered context, and parallel grammars. The book was not read cover to cover.

**The named families, in plain words.** Each one is a way of stopping the rules from running freely. They are not five names for one device, but they are the same family of ideas.

| Device | Plain description | Source | Status |
|---|---|---|---|
| **Control set** | A second language of rule-name sequences. Only those sequences may be used. | Ginsburg and Spanier 1968; Dassow's "regularly controlled grammar" | Definition **Confirmed** from Dassow's chapter. 1968 paper not re-opened. |
| **Matrix grammar** | Rules are packed into fixed lists ("matrices"). Using a matrix means using its rules in that order, as one step. | Abraham 1965, as attributed by Dassow and Păun 1989; definition in Dassow's chapter | Attribution to Abraham 1965 **Confirmed** from the book preview. Abraham's paper was not opened. Usual pagination (Computational Linguistics 4: 61–70) was **not** re-checked (**Unconfirmed** pages). |
| **Programmed grammar** | Each rule carries a success list and a failure list. If the rule applies, the next rule must come from the success list. If the symbol it looks for is absent, the next rule comes from the failure list (appearance checking). | Dassow's chapter, Definition 6. Introduced by Rosenkrantz: conference paper 1967; journal version Rosenkrantz, D. J. (1969), "Programmed grammars and classes of formal languages," *Journal of the ACM* 16(1): 107–131 | The mechanism **Confirmed** from Dassow. The 1967 symposium title is **Confirmed** via the ACM record in search. The 1969 JACM pagination is the standard citation and was **not** re-opened (**Unconfirmed** pages this pass). |
| **Valence grammar** | Each rule carries a value in some number system (a monoid). Values accumulate as rules fire. A finished string counts only if the running total is the neutral element (0 for addition, 1 for multiplication). | Dassow's chapter, Definition 7. Introduced by Păun in 1980 | Mechanism **Confirmed** from Dassow. Year 1980 **Confirmed** from a Fernau and Stiebe snippet quoted in search ("introduced in 1980 by Păun"). Usual citation: Păun, Gh. (1980), "A new generative device: valence grammars," *Revue Roumaine de Mathématiques Pures et Appliquées* 25: 911–924. Those pages were **not** re-opened (**Unconfirmed** pages). |
| **Ordered, conditional, random-context grammars** | A rule may fire only if no higher rule can, or only if the current string matches a required shape, or only if certain symbols are present and certain others are absent. | Dassow's chapter, Definitions 8–11 | **Confirmed** from that chapter. |

Dassow's Theorem 5, in the chapter read, states that programmed grammars and matrix grammars generate the same family (with and without appearance checking). **Confirmed** as his theorem, not re-proved here.

What this offers for "a way of control": it is a second sequence, or a second set of permissions, laid over an alphabet's rules. The alphabet does not control itself. The control device decides which of the alphabet's rules may run. That is as close as the published literature comes to her phrase. It is still only a candidate analogy.

## 3. The I Ching as a generative system

This is how the book builds, not a theory about her "alphabet swarm".

**Stacking.** Two symbols, broken and solid (yin and yang). They are stacked, read from the bottom. Three lines make a trigram. Two trigrams, or six lines, make a hexagram. There are 2^6 = 64 hexagrams. The classical statement of the build, in Legge's public-domain translation of the Xici (Great Appendix), paragraph 70: "Therefore in (the system of) the Yî there is the Grand Terminus, which produced the two elementary Forms. Those two Forms produced the Four emblematic Symbols, which again produced the eight Trigrams." **Confirmed** as that sentence in the Internet Sacred Text Archive search snippet for Legge's Appendix III. The page itself returned a bot check this pass and was not re-opened. Lynn's introduction (the 1994 Wang Bi translation) says the same build in modern words: sixty-four hexagrams, each of six lines, each line solid or broken, formed as combinations of two trigrams. **Confirmed** from the opening pages of that PDF, fetched this pass. (The PDF is a full book; only that structural description is used.)

**Fixed depth, finite set.** The stack stops at 6. There is no seventh line, and no rule that keeps stacking. The generated set has 64 members. That is the opposite of section 1. A finite alphabet plus Merge, or plus a recursive grammar, has no fixed depth. A hexagram does.

**Line change.** In divination, each line is not only yin or yang. It is young (stable) or old (changing). The usual numbers are 6 (old yin), 7 (young yang), 8 (young yin), 9 (old yang). Old yin becomes yang, and old yang becomes yin. The other lines stay. The result is a second hexagram. Wikipedia's "I Ching divination" (fetched this pass) states this, including the names and the direction of the flips. **Confirmed** as that encyclopedia article, which is a secondary source. It is the standard divinatory operation. It is not in the Bronze Age text as a grammatical rule; file 01 already records Rutt's view that the later line-relation principles are not part of Bronze Age practice. The changing-line procedure is a received divinatory method. How early it is was not re-investigated (**Unconfirmed** as to date).

What the flip does, and does not do: it maps one finished stack of 6 to another finished stack of 6. It does not lengthen the string. It does not choose a rewrite rule from a control set. One flip, or several, stays inside the 64.

**King Wen order versus binary (Fu Xi / Shao Yong) order.** These are two ways of *listing* the 64. They are not two ways of building a hexagram. Building is the stack either way.

- **King Wen order** is the received book order. The 64 figures form 32 pairs. In 28 pairs the second is the first turned upside down. The 8 figures that look the same upside down (1 and 2, 27 and 28, 29 and 30, 61 and 62) are paired by flipping every line. **Confirmed** from the Wikipedia article "King Wen sequence", fetched this pass. Traditional authorship by King Wen is legend; the article says the true age and authorship are unknown. **Confirmed** as the article's statement.
- **Binary order**, also called the Fu Xi sequence or the Shao Yong sequence, treats the six lines as a binary number and lists all 64 that way. Wikipedia (same article) says this arrangement originated in the Song dynasty and is believed to be the work of Shao Yong (1011–1077). The King Wen sequence is older. **Confirmed** as that article. A specialist page (biroco.com, "Yijing Dao", fetched via search) describes Shao Yong's diagram as the 64 stacks in binary order, and notes that it reached Leibniz in 1701 through Bouvet. **Confirmed** as that page's account, not as a critical edition.

For "building", the binary order matters only as a complete list of the same fixed stacks. It does not add a rule of growth. The King Wen order matters as the order of the book, and as a pairing by turning the stack over (or, for eight of them, by flipping every line). It is not the operation that builds the stack in the first place.

**Line relations** (correspondence, adjacency, position, centrality) are already set out in file 01, from her primary books. A candidate analogy, not her meaning and not a finding of this note: those relations read a stack that is already built. They are not a control set. This note does not turn them into one.

## 4. Swarm and symbols

Only sources that were actually found. The phrase "alphabet swarm" was not found in them.

**Swarm grammars.** Jacob, C. and von Mammen, S. (2007). "Swarm grammars: growing dynamic structures in 3D agent spaces." *Digital Creativity* 18(1): 54–64. DOI 10.1080/14626260701253622. **Confirmed** (journal page and the authors' PDF, both seen via search; the PDF text was in the search extract). Plain description: an L-system-like grammar says which kinds of agents appear and how one kind is replaced by others. Many agents, not one "turtle", then move in 3D space and leave a branching structure. The authors say three things influence the result: the swarm rules, the agent parameters, and the environment. That is a real control mechanism. It controls 3D growth by agents. It does not control an alphabet that builds like the I Ching. Later papers by the same authors extend this to evolution and to "swarm graph grammars" (von Mammen's 2009 thesis, seen as a PDF extract). **Not fringe**: it is peer-reviewed design research. **Not her phrase.**

**L-systems.** Lindenmayer, A. (1968). "Mathematical models for cellular interactions in development. I. Filaments with one-sided inputs." *Journal of Theoretical Biology* 18: 280–299. Part II, same volume, 300–315. **Confirmed** as bibliographic records (ADS and PubMed). Plain description: every symbol in a string is rewritten at the same time, by a growth rule. That parallel step is the difference from an ordinary grammar, which rewrites one symbol at a time. The rules are the growth control. A later, separate device puts a control set on top: **Ginsburg, S. and Rozenberg, G. (1975). "TOL Schemes and Control Sets." *Information and Control* 27(2): 109–125.** **Confirmed** from the DBLP contents of volume 27 (fetched). The paper was not read, so no theorem is quoted from it. The title, in this literature, means a second sequence chooses which table of parallel rules fires at each step. That is the same idea as section 2, applied to growth rules. Whether it fits an "alphabet swarm" is not decided here.

**Colonies and grammar systems.** A colony is a collection of very simple grammars sharing one string. Each component only produces a finite language. They take turns, or work together, rewriting that shared string. **Confirmed** from Kelemenová, A., "Colonies — the simplest grammar systems" (CEUR-WS Vol-255; PDF fetched). That survey cites the introduction of colonies as Kelemen, J. and Kelemenová, A. (1992). "A grammar-theoretic treatment of multiagent systems." *Cybernetics and Systems* 23: 621–633. The 1992 paper was not re-opened; the citation is **Confirmed** as a reference inside the survey that was read. The wider book is Csuhaj-Varjú, Dassow, Kelemen, and Păun, *Grammar Systems: A Grammatical Approach to Distribution and Cooperation* (Gordon and Breach, 1994), named in that literature and not re-read (**Confirmed** as the standard monograph those papers point to; contents not re-checked). Plain description: control, if that word is used, is "which simple grammar acts on the shared string, and when." Sequential colonies with the basic mode generate the context-free languages (Theorem 1 of the survey read). They are not a swarm in the 3D-agent sense, and they do not build a depth-6 binary stack.

**Cellular automata.** A grid of symbols, each updated by a fixed rule that looks at its neighbors. Standard mathematics, not fringe: von Neumann's self-reproducing automata, and the later study of elementary rules (Wolfram and others). A neighborhood rule is a kind of control: the next symbol is not free. It is not a control set in the Ginsburg–Spanier sense, and it does not stack lines into hexagrams. No source was found that makes a cellular automaton into an "alphabet swarm". **Unconfirmed** that any such source exists; it was not found in this pass. Broader claims that a cellular automaton "is" a cosmology are not used here.

**Fringe flag.** No peer-reviewed source was found for the phrase "alphabet swarm", or for a swarm whose job is to build like the I Ching. Treating swarm grammars, colonies, or cellular automata as if they were that phrase would be an addition. It is not made here.

## 5. Ji's alphabet of protosemata, and double articulation

Source already checked against Alexis's copy in file 04, and not re-opened this pass. Europe PMC timed out. The Wiley page was blocked when file 04 was written. So this section does not go past file 04.

**Ji, Sungchul (1999). "The Linguistics of DNA: Words, Sentences, Grammar, Phonetics, and Semantics." *Annals of the New York Academy of Sciences* 870: 411–417.** DOI 10.1111/j.1749-6632.1999.tb08916.x. **Confirmed** (file 04).

What file 04 records from her copy, and what this note repeats:

- Cell language is a 6-tuple **{L, W, S, G, P, M}**.
- **L is the alphabet**: "a set of basic symbols called protosemata" (the 4 nucleotides, or the 20 amino acids). The other five are words, sentences, grammar, phonetics, and semantics/meaning.
- He uses **first and second articulation** ("double articulation"): small units without meaning combine into units with meaning.

That is the bridge, and the whole of it here. An alphabet of protosemata is the lower level. Words, sentences, and grammar are the higher level he names. Double articulation is the published linguistic idea he applies: meaningless units combine into meaningful ones. File 09 notes that Martinet (1949/1960) and Hockett (1958, "duality of patterning") are the linguistic sources, via the Wikipedia article "Double articulation", which those files treat as a secondary source. The originals were not read.

Not extended. File 04 also records a "loaded carousel", lexical/syntactic/semantic codes at 3%, 100%, and 97% of **human** DNA mass, and the Bhopalator. Those are separate claims. They are not a definition of the alphabet, and the percentages are not used here. Ji's framework is his proposal, not a consensus model of the genome (file 04).

A candidate analogy, not her meaning: Ji's L is an alphabet in the formal sense (a finite set of basic symbols), and his G is a grammar over it. That is a published bridge between an alphabet and a higher syntax. It does not say what an "alphabet swarm" is, and it does not say that the alphabet builds like the I Ching.

## What this does not decide

No control method was chosen for her. A control set, a matrix, a programmed grammar, a valence, a table of L-system rules, a colony of small grammars, and the I Ching line-flip are different devices. This note does not say which of them, if any, is her "way of control", and it does not combine them into a system. "Alphabet swarm", "way of control", and "build like the I Ching" stay her phrases. The published ideas above are candidate analogies only.

## Sources

- Chomsky 1959, *Information and Control* 2: 137–167: https://doi.org/10.1016/S0019-9958(59)90362-6 (opening sentence in the PDF copies indexed by search)
- Chomsky 1956: https://doi.org/10.1109/TIT.1956.1056813
- Jäger and Rogers 2012, PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC3367686/
- Chomsky 1995, *The Minimalist Program*: https://doi.org/10.7551/mitpress/9780262527347.001.0001 (record as in file 09)
- Marcolli, Chomsky, and Berwick: https://arxiv.org/abs/2305.18278 · https://doi.org/10.7551/mitpress/15811.001.0001
- Merge and the Strong Minimalist Thesis, PDF fetched: https://www.its.caltech.edu/~matilde/ChomskyMergeStrongMinimalistThesisElements2023.pdf
- Ginsburg and Spanier 1968: https://doi.org/10.1007/BF01692513
- Dassow, "Grammars with Regulated Rewriting" (chapter PDF read): http://theo.cs.ovgu.de/dassow/tarraphd.pdf
- Dassow and Păun 1989, Springer; preview read for the Abraham 1965 attribution and the list of regulation types
- Dassow, Păun, and Salomaa, "Grammars with controlled derivations," in Rozenberg and Salomaa (eds.), *Handbook of Formal Languages*, vol. 2, Springer, 1997, pp. 101–154 (cited inside Dassow's chapter; chapter not re-read)
- Ginsburg and Rozenberg 1975, DBLP contents of *Information and Control* 27: https://sigmod.org/publications/dblp/db/journals/iandc/iandc27.html
- Lindenmayer 1968, parts I and II, *Journal of Theoretical Biology* 18
- Jacob and von Mammen 2007: https://doi.org/10.1080/14626260701253622 · authors' PDF: https://downloads.hci.informatik.uni-wuerzburg.de/Jacob2007qf.pdf
- Kelemenová, "Colonies — the simplest grammar systems": https://ceur-ws.org/Vol-255/paper01.pdf
- Ji 1999, as in file 04: https://doi.org/10.1111/j.1749-6632.1999.tb08916.x
- Legge, Appendix III, paragraph 70, Internet Sacred Text Archive (search snippet; page bot-blocked on fetch): https://sacred-texts.com/ich/icap3-1.htm
- Lynn, *The Classic of Changes* (1994), opening structural description, PDF fetched
- King Wen sequence (secondary): https://en.wikipedia.org/wiki/King_Wen_sequence
- I Ching divination, changing lines 6, 7, 8, 9 (secondary): https://en.wikipedia.org/wiki/I_Ching_divination
- Shao Yong binary diagram (secondary): https://biroco.com/yijing/sequence.htm
- Klein's publication list (titles only): https://pages.cs.wisc.edu/~sklein/sklein.html
