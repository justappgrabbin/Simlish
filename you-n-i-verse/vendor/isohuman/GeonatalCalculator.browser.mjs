import * as astronomy_engine_1 from './astronomy-engine.mjs';
const GATE_WHEEL_START = 302;
const GATE_ORDER = [
    41, 19, 13, 49, 30, 55, 37, 63, 22, 36, 25, 17, 21, 51, 42, 3,
    27, 24, 2, 23, 8, 20, 16, 35, 45, 12, 15, 52, 39, 53, 62, 56,
    31, 33, 7, 4, 29, 59, 40, 64, 47, 6, 46, 18, 48, 57, 32, 50,
    28, 44, 1, 43, 14, 34, 9, 5, 26, 11, 10, 58, 38, 54, 61, 60,
];
const BODIES = [
    ["Sun", astronomy_engine_1.Body.Sun],
    ["Earth", "Earth"],
    ["North Node", "North Node"],
    ["South Node", "South Node"],
    ["Moon", astronomy_engine_1.Body.Moon],
    ["Mercury", astronomy_engine_1.Body.Mercury],
    ["Venus", astronomy_engine_1.Body.Venus],
    ["Mars", astronomy_engine_1.Body.Mars],
    ["Jupiter", astronomy_engine_1.Body.Jupiter],
    ["Saturn", astronomy_engine_1.Body.Saturn],
    ["Uranus", astronomy_engine_1.Body.Uranus],
    ["Neptune", astronomy_engine_1.Body.Neptune],
    ["Pluto", astronomy_engine_1.Body.Pluto],
];
const PERSPECTIVES = [
    "being",
    "movement",
    "evolution",
    "design",
    "space",
];
export function calculateGeonatalChart(timestamp, location) {
    validateLocation(location);
    const personalitySun = tropicalLongitude(astronomy_engine_1.Body.Sun, timestamp);
    const designTimestamp = findDesignTimestamp(timestamp, personalitySun);
    const trueNode = trueNodeLongitude(timestamp);
    const ayanamsa = faganBradleyAyanamsa(timestamp);
    const placements = [];
    for (const orientation of ["personality", "design"]) {
        const date = orientation === "personality" ? timestamp : designTimestamp;
        const node = trueNodeLongitude(date);
        const sun = tropicalLongitude(astronomy_engine_1.Body.Sun, date);
        for (const [planetary, body] of BODIES) {
            const tropical = body === "Earth"
                ? normalize(sun + 180)
                : body === "North Node"
                    ? node
                    : body === "South Node"
                        ? normalize(node + 180)
                        : tropicalLongitude(body, date);
            for (const frame of [
                "tropical",
                "sidereal_fagan_bradley",
                "draconic_tropical_true",
            ]) {
                const longitude = frame === "tropical"
                    ? tropical
                    : frame === "sidereal_fagan_bradley"
                        ? normalize(tropical - faganBradleyAyanamsa(date))
                        : normalize(tropical - node);
                placements.push(makePlacement(planetary, "being", frame, orientation, longitude, date, location));
            }
        }
    }
    const perspectives = Object.fromEntries(PERSPECTIVES.map((perspective) => [
        perspective,
        placements.map((placement) => ({ ...placement, dimension: perspective })),
    ]));
    return {
        timestamp: timestamp.toISOString(),
        designTimestamp: designTimestamp.toISOString(),
        location,
        houseMethod: "equal",
        trueNodeLongitude: trueNode,
        faganBradleyAyanamsa: ayanamsa,
        placements,
        perspectives,
        methods: {
            ephemeris: "astronomy-engine 2.1.19 geocentric true-ecliptic-of-date",
            sidereal: "Fagan-Bradley J2000 offset with general-precession rate",
            draconic: "tropical longitude minus interpolated true ascending lunar node",
            design: "timestamp at 88 degrees of prior solar longitude",
            houses: "equal houses from calculated tropical ascendant",
        },
    };
}
export function humanDesignResolution(longitude) {
    const angle = normalize(longitude - GATE_WHEEL_START);
    const gateIndex = Math.floor(angle / 5.625);
    let remainder = angle % 5.625;
    const line = Math.floor(remainder / 0.9375) + 1;
    remainder %= 0.9375;
    const color = Math.floor(remainder / 0.15625) + 1;
    remainder %= 0.15625;
    const tone = Math.floor(remainder / (0.15625 / 6)) + 1;
    remainder %= 0.15625 / 6;
    const base = Math.min(5, Math.floor(remainder / (0.15625 / 6 / 5)) + 1);
    return { gate: GATE_ORDER[gateIndex], line, color, tone, base };
}
export function faganBradleyAyanamsa(date) {
    const j2000 = Date.UTC(2000, 0, 1, 12);
    const tropicalYears = (date.getTime() - j2000) / (365.2425 * 86400000);
    return normalize(24.7366667 + (50.290966 / 3600) * tropicalYears);
}
function makePlacement(planetary, dimension, frame, orientation, longitude, date, location) {
    const zodiacPosition = zodiacBreakdown(longitude);
    const ascendant = equalHouseAscendant(date, location);
    const house = Math.floor(normalize(longitude - ascendant) / 30) + 1;
    return {
        planetary,
        dimension,
        frame,
        orientation,
        ...zodiacPosition,
        house,
        ascendingSide: house,
        ...humanDesignResolution(longitude),
    };
}
function zodiacBreakdown(longitude) {
    const normalized = normalize(longitude);
    const zodiac = Math.floor(normalized / 30) + 1;
    const withinSign = normalized % 30;
    const degree = Math.floor(withinSign);
    const minuteFloat = (withinSign - degree) * 60;
    const minute = Math.floor(minuteFloat);
    const secondFloat = (minuteFloat - minute) * 60;
    const second = Math.floor(secondFloat);
    const arc = Math.min(99, Math.floor((secondFloat - second) * 100));
    return { longitude: normalized, zodiac, degree, minute, second, arc };
}
function tropicalLongitude(body, date) {
    if (body === astronomy_engine_1.Body.Moon)
        return normalize((0, astronomy_engine_1.EclipticGeoMoon)(date).lon);
    return normalize((0, astronomy_engine_1.Ecliptic)((0, astronomy_engine_1.GeoVector)(body, date, true)).elon);
}
function trueNodeLongitude(date) {
    let event = (0, astronomy_engine_1.SearchMoonNode)(new Date(date.getTime() - 35 * 86400000));
    let previous = event;
    while (event.time.date.getTime() <= date.getTime()) {
        previous = event;
        event = (0, astronomy_engine_1.NextMoonNode)(event);
    }
    const before = nodeLongitudeAtEvent(previous);
    const after = nodeLongitudeAtEvent(event);
    const span = event.time.date.getTime() - previous.time.date.getTime();
    const fraction = (date.getTime() - previous.time.date.getTime()) / span;
    const delta = signedDelta(before, after);
    return normalize(before + delta * fraction);
}
function nodeLongitudeAtEvent(event) {
    const moon = (0, astronomy_engine_1.EclipticGeoMoon)(event.time).lon;
    return normalize(event.kind === astronomy_engine_1.NodeEventKind.Ascending ? moon : moon + 180);
}
function findDesignTimestamp(natal, natalSun) {
    let lowDays = 70;
    let highDays = 110;
    for (let iteration = 0; iteration < 60; iteration += 1) {
        const days = (lowDays + highDays) / 2;
        const candidate = new Date(natal.getTime() - days * 86400000);
        const separation = normalize(natalSun - tropicalLongitude(astronomy_engine_1.Body.Sun, candidate));
        if (separation < 88)
            lowDays = days;
        else
            highDays = days;
    }
    return new Date(natal.getTime() - ((lowDays + highDays) / 2) * 86400000);
}
function equalHouseAscendant(date, location) {
    const lst = normalize((0, astronomy_engine_1.SiderealTime)(date) * 15 + location.longitude);
    const radians = Math.PI / 180;
    const theta = lst * radians;
    const latitude = location.latitude * radians;
    const obliquity = 23.4392911 * radians;
    const ascendant = Math.atan2(-Math.cos(theta), Math.sin(obliquity) * Math.tan(latitude) +
        Math.cos(obliquity) * Math.sin(theta)) / radians;
    return normalize(ascendant);
}
function signedDelta(from, to) {
    return ((to - from + 540) % 360) - 180;
}
function normalize(value) {
    return ((value % 360) + 360) % 360;
}
function validateLocation(location) {
    if (!Number.isFinite(location.latitude) ||
        location.latitude < -90 ||
        location.latitude > 90) {
        throw new Error("Latitude must be between -90 and 90 degrees");
    }
    if (!Number.isFinite(location.longitude) ||
        location.longitude < -180 ||
        location.longitude > 180) {
        throw new Error("Longitude must be between -180 and 180 degrees");
    }
}
//# sourceMappingURL=GeonatalCalculator.js.map