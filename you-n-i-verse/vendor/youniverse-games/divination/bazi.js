// BaZi (Four Pillars) calculator — divination engine core.
// Gregorian birth date/time/longitude/timezone -> true solar time adjustment
// -> Four Pillars (year/month/day/hour stem-branch) + element analysis.
// No dependencies; deterministic.

export const STEMS = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
export const BRANCHES = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

export const STEM_ELEMENTS = ['wood', 'wood', 'fire', 'fire', 'earth', 'earth', 'metal', 'metal', 'water', 'water'];
// 寅卯 wood, 巳午 fire, 申酉 metal, 亥子 water, 丑辰未戌 earth
export const BRANCH_ELEMENTS = ['water', 'earth', 'wood', 'wood', 'earth', 'fire', 'fire', 'earth', 'metal', 'metal', 'earth', 'water'];

export const FIVE_ELEMENTS = ['wood', 'fire', 'earth', 'metal', 'water'];
// Generating cycle: wood feeds fire, fire makes earth, earth bears metal, metal enriches water, water nourishes wood.
export const GENERATES = { wood: 'fire', fire: 'earth', earth: 'metal', metal: 'water', water: 'wood' };
// Controlling cycle: wood parts earth, earth dams water, water quenches fire, fire melts metal, metal chops wood.
export const CONTROLS = { wood: 'earth', earth: 'water', water: 'fire', fire: 'metal', metal: 'wood' };

// Sexagenary reference epoch: 2000-01-01 is a 戊午 day (documented almanac fact).
// 戊 = stem index 4, 午 = branch index 6 -> cycle index 54 (54%10==4, 54%12==6).
export const DAY_REFERENCE = { date: '2000-01-01', cycleIndex: 54 };

function daysBetweenUTC(dateA, dateB) {
  const ms = Date.UTC(dateB.y, dateB.m - 1, dateB.d) - Date.UTC(dateA.y, dateA.m - 1, dateA.d);
  return Math.round(ms / 86400000);
}

function mod(n, m) { return ((n % m) + m) % m; }

/**
 * True solar time: shift local clock time by longitude offset from the
 * timezone's standard meridian (4 minutes per degree), plus equation-of-time
 * approximation (simple sinusoid, max ~±14 min).
 */
export function trueSolarTime({ date, time, longitude, timezone }) {
  const [y, m, d] = date.split('-').map(Number);
  const [hh, mm] = time.split(':').map(Number);
  if (!Number.isFinite(longitude)) throw new Error('bazi: longitude required for true solar time');
  const tzHours = typeof timezone === 'string'
    ? (timezone.startsWith('+') || timezone.startsWith('-') ? parseFloat(timezone.slice(0, 3)) + parseFloat(timezone.slice(4)) / 60 * Math.sign(parseFloat(timezone)) : parseFloat(timezone))
    : timezone;
  if (!Number.isFinite(tzHours)) throw new Error(`bazi: bad timezone "${timezone}"`);
  const standardMeridian = tzHours * 15;
  const longitudeCorrectionMin = (longitude - standardMeridian) * 4;
  // Equation of time approximation (minutes).
  const dayOfYear = Math.floor((Date.UTC(y, m - 1, d) - Date.UTC(y, 0, 0)) / 86400000);
  const B = (2 * Math.PI * (dayOfYear - 81)) / 364;
  const eot = 9.87 * Math.sin(2 * B) - 7.53 * Math.cos(B) - 1.5 * Math.sin(B);
  const totalMin = hh * 60 + mm + longitudeCorrectionMin + eot;
  const shifted = new Date(Date.UTC(y, m - 1, d, 0, Math.round(totalMin)));
  return {
    year: shifted.getUTCFullYear(), month: shifted.getUTCMonth() + 1, day: shifted.getUTCDate(),
    hour: shifted.getUTCHours(), minute: shifted.getUTCMinutes(),
    correctionMinutes: +(longitudeCorrectionMin + eot).toFixed(2),
  };
}

/**
 * Compute the Four Pillars.
 * @param {object} birth - {date:'YYYY-MM-DD', time:'HH:MM', longitude, timezone}
 */
export function computeFourPillars(birth) {
  if (!birth?.date || !birth?.time) throw new Error('bazi: birth.date and birth.time required');
  const solar = trueSolarTime({
    date: birth.date, time: birth.time,
    longitude: birth.longitude ?? birth.lon,
    timezone: birth.timezone,
  });

  // --- Year pillar (approximation: 立春 ~ Feb 4) ---
  let yYear = solar.year;
  if (solar.month < 2 || (solar.month === 2 && solar.day < 4)) yYear -= 1;
  const yearCycle = mod(yYear - 4, 60); // 4 AD = 甲子
  const yearStem = yearCycle % 10, yearBranch = yearCycle % 12;

  // --- Month pillar (solar-term approximation: terms begin ~6th of month) ---
  let mMonth = solar.month, mYear = yYear;
  if (solar.day < 6) { mMonth -= 1; if (mMonth === 0) { mMonth = 12; mYear -= 1; } }
  const monthBranch = mMonth % 12; // Jan->丑(1), Feb->寅(2), ..., Dec->子(0)
  const monthYearStem = mod(mYear - 4, 10);
  const firstMonthStem = mod((monthYearStem % 5) * 2 + 2, 10); // stem of 寅月
  const monthStem = mod(firstMonthStem + mod(monthBranch - 2, 12), 10);

  // --- Day pillar (real sexagenary day count from reference epoch) ---
  const ref = DAY_REFERENCE;
  const [ry, rm, rd] = ref.date.split('-').map(Number);
  const delta = daysBetweenUTC({ y: ry, m: rm, d: rd }, { y: solar.year, m: solar.month, d: solar.day });
  const dayCycle = mod(ref.cycleIndex + delta, 60);
  const dayStem = dayCycle % 10, dayBranch = dayCycle % 12;

  // --- Hour pillar (from true solar hour + day stem) ---
  const hourBranch = Math.floor(((solar.hour + 1) % 24) / 2);
  const hourStartStem = mod((dayStem % 5) * 2, 10); // stem of 子时
  const hourStem = mod(hourStartStem + hourBranch, 10);

  const pillar = (s, b) => ({
    stem: STEMS[s], branch: BRANCHES[b], stemIndex: s, branchIndex: b,
    stemElement: STEM_ELEMENTS[s], branchElement: BRANCH_ELEMENTS[b],
    text: STEMS[s] + BRANCHES[b],
  });

  return {
    year: pillar(yearStem, yearBranch),
    month: pillar(monthStem, monthBranch),
    day: pillar(dayStem, dayBranch),
    hour: pillar(hourStem, hourBranch),
    trueSolarTime: solar,
  };
}

/** Element analysis: counts over all 8 characters, dominant & weakest. */
export function analyzeElements(pillars) {
  const counts = Object.fromEntries(FIVE_ELEMENTS.map((e) => [e, 0]));
  for (const key of ['year', 'month', 'day', 'hour']) {
    counts[pillars[key].stemElement] += 1;
    counts[pillars[key].branchElement] += 1;
  }
  const sorted = FIVE_ELEMENTS.slice().sort((a, b) => counts[b] - counts[a] || FIVE_ELEMENTS.indexOf(a) - FIVE_ELEMENTS.indexOf(b));
  return {
    counts,
    dominant: sorted[0],
    weakest: sorted[sorted.length - 1],
    generatingPair: [sorted[0], GENERATES[sorted[0]]],
    controllingWeak: Object.keys(CONTROLS).find((k) => CONTROLS[k] === sorted[sorted.length - 1]),
  };
}

/** Full BaZi chart from a BirthDataStore birth record ({date,time,place:{lon},timezone}). */
export function computeBaZi(birthRecord) {
  const b = birthRecord?.birth ?? birthRecord;
  const pillars = computeFourPillars({
    date: b.date, time: b.time,
    longitude: b.longitude ?? b.place?.lon ?? b.place?.longitude,
    timezone: b.timezone,
  });
  const elements = analyzeElements(pillars);
  return { pillars, elements };
}
