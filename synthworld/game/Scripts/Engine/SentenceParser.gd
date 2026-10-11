extends Node
class_name SentenceParser

const ALLOWED_KEYS := ["agent", "verb", "object", "at", "when", "because", "scene", "with", "note"]
const ALLOWED_VERBS := ["build", "repair", "move", "emote", "spawn", "set", "light", "play"]

static func parse_line(line: String) -> Dictionary:
    var ast := {}
    var errors: Array[String] = []
    for raw_piece in line.split("|"):
        var piece := raw_piece.strip_edges()
        if piece.is_empty():
            continue
        var split_at := piece.find(":")
        if split_at < 1:
            errors.append("Missing key:value separator in '%s'" % piece)
            continue
        var key := piece.substr(0, split_at).strip_edges()
        var value := piece.substr(split_at + 1).strip_edges()
        if key not in ALLOWED_KEYS:
            errors.append("Unknown key '%s'" % key)
            continue
        ast[key] = value
    if not ast.has("agent"):
        errors.append("Missing required key 'agent'")
    if not ast.has("verb"):
        errors.append("Missing required key 'verb'")
    elif String(ast.verb) not in ALLOWED_VERBS:
        errors.append("Unsupported verb '%s'" % ast.verb)
    return {"ok": errors.is_empty(), "ast": ast, "errors": errors}

static func parse_position(value: String) -> Vector3:
    var cleaned := value.strip_edges().trim_prefix("(").trim_suffix(")")
    var parts := cleaned.split(",")
    if parts.size() != 3:
        return Vector3.ZERO
    return Vector3(float(parts[0]), float(parts[1]), float(parts[2]))

static func parse_with(value: String) -> Dictionary:
    var out := {}
    for pair in value.split(","):
        var idx := pair.find("=")
        if idx > 0:
            out[pair.substr(0, idx).strip_edges()] = pair.substr(idx + 1).strip_edges()
    return out
