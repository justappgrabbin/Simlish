// Canonical birth-data registry. Simulates the "existing birthday system".
// This store is the ONLY source of birth data; games must never hold their own copy.

const CENTERS = ['head', 'ajna', 'throat', 'g', 'heart', 'sacral', 'solarPlexus', 'spleen', 'root'];

function makeGate(gate, line, center, planet, designation) {
  return { gate, line, center, planet, designation }; // designation: 'personality' | 'design'
}

// Full Human Design chart + I Ching gate state for the pre-seeded test user.
const SEED_BIRTH_DATA = {
  'bd-test-001': {
    birthDataId: 'bd-test-001',
    name: 'Test Traveler',
    birth: {
      date: '1990-04-17',
      time: '06:42',
      timezone: '+08:00',
      place: { city: 'Guilin', country: 'China', lat: 25.28, lon: 110.29 },
    },
    humanDesign: {
      type: 'Generator',
      authority: 'Sacral',
      profile: '3/5',
      definition: 'Split Definition',
      strategy: 'To Respond',
      signature: 'Satisfaction',
      notSelfTheme: 'Frustration',
      incarnationCross: 'Right Angle Cross of the Unexpected (31/41 | 27/28)',
      centers: {
        head: { defined: false },
        ajna: { defined: false },
        throat: { defined: true },
        g: { defined: true },
        heart: { defined: false },
        sacral: { defined: true },
        solarPlexus: { defined: false },
        spleen: { defined: true },
        root: { defined: true },
      },
      channels: [
        { id: '34-20', gates: [34, 20], centers: ['sacral', 'throat'], name: 'Charisma' },
        { id: '27-50', gates: [27, 50], centers: ['sacral', 'spleen'], name: 'Preservation' },
        { id: '41-30', gates: [41, 30], centers: ['root', 'solarPlexus'], name: 'Recognition', note: 'hanging on solarPlexus side' },
        { id: '28-38', gates: [28, 38], centers: ['spleen', 'root'], name: 'Struggle' },
        { id: '31-7', gates: [31, 7], centers: ['throat', 'g'], name: 'The Alpha' },
      ],
      gates: [
        makeGate(31, 2, 'throat', 'Sun', 'personality'),
        makeGate(41, 4, 'root', 'Earth', 'personality'),
        makeGate(27, 1, 'sacral', 'Moon', 'personality'),
        makeGate(34, 6, 'sacral', 'Mars', 'personality'),
        makeGate(20, 3, 'throat', 'Venus', 'personality'),
        makeGate(28, 5, 'spleen', 'Mercury', 'design'),
        makeGate(38, 2, 'root', 'Mars', 'design'),
        makeGate(50, 1, 'spleen', 'Jupiter', 'design'),
        makeGate(7, 4, 'g', 'Saturn', 'design'),
        makeGate(30, 6, 'solarPlexus', 'Pluto', 'design'),
      ],
      variables: { digestion: 'Consecutive', environment: 'Mountains', awareness: 'Focused', perspective: 'Power' },
    },
    iching: {
      activeHexagram: 41, // Decrease — Sun/Earth personality
      activeLine: 4,
      activeGateStates: { 31: 2, 41: 4, 27: 1, 34: 6, 20: 3, 28: 5, 38: 2, 50: 1, 7: 4, 30: 6 },
    },
  },
};

export class BirthDataStore {
  constructor(extraRecords = {}) {
    // Deep-clone seeds so callers can never mutate the canonical registry.
    this._records = new Map();
    for (const [id, rec] of Object.entries({ ...SEED_BIRTH_DATA, ...extraRecords })) {
      this._records.set(id, structuredClone(rec));
    }
  }

  has(birthDataId) {
    return this._records.has(birthDataId);
  }

  listIds() {
    return [...this._records.keys()];
  }

  // Returns a deep clone — canonical record is never shared by reference.
  get(birthDataId) {
    const rec = this._records.get(birthDataId);
    if (!rec) throw new Error(`BirthDataStore: no birth data for id "${birthDataId}"`);
    return structuredClone(rec);
  }

  getHumanDesign(birthDataId) {
    return this.get(birthDataId).humanDesign;
  }

  getIChingState(birthDataId) {
    return this.get(birthDataId).iching;
  }

  getDefinedCenters(birthDataId) {
    const hd = this.getHumanDesign(birthDataId);
    return CENTERS.filter((c) => hd.centers[c]?.defined);
  }
}

export { CENTERS };
