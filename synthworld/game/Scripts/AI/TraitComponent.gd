
extends Node
# Attach to an agent to apply trait modifiers.
# Public API:
#   set_traits(ids: Array[String])
#   get_stat_mult(name: String) -> float
#   apply_drive_modifiers(drives: Dictionary) -> Dictionary
#   bias_action_score(action: String, base_score: float, context: Dictionary) -> float

var trait_ids: Array = []
var stats_mult := {
    "speedMult": 1.0,
    "repairAmountMult": 1.0,
    "buildSpeedMult": 1.0
}
var drive_add := {} # e.g., {"creation": 0.1}

func set_traits(ids: Array) -> void:
    trait_ids = ids.duplicate()
    _recalc_cache()

func _recalc_cache() -> void:
    stats_mult = {"speedMult":1.0,"repairAmountMult":1.0,"buildSpeedMult":1.0}
    drive_add = {}
    for tid in trait_ids:
        var td = TraitSystem.get_trait(tid)
        if td.is_empty(): continue
        # Stats multipliers
        if td.has("modifiers") and td.modifiers.has("stats"):
            for k in td.modifiers.stats.keys():
                var m = float(td.modifiers.stats[k])
                stats_mult[k] = (stats_mult.get(k, 1.0)) * m
        # Drive additive tweaks
        if td.has("modifiers") and td.modifiers.has("drives"):
            for k in td.modifiers.drives.keys():
                drive_add[k] = (drive_add.get(k, 0.0)) + float(td.modifiers.drives[k])

func get_stat_mult(name: String) -> float:
    return float(stats_mult.get(name, 1.0))

func apply_drive_modifiers(drives: Dictionary) -> Dictionary:
    var out := drives.duplicate()
    for k in drive_add.keys():
        out[k] = clamp(float(out.get(k,0.0)) + float(drive_add[k]), 0.0, 1.0)
    return out

func bias_action_score(action: String, base_score: float, context: Dictionary) -> float:
    var score := base_score
    for tid in trait_ids:
        var td = TraitSystem.get_trait(tid)
        if td.is_empty(): continue
        var bias = td.get("modifiers", {}).get("action_bias", {}).get(action, {})
        if bias.has("scoreAdd"):
            score += float(bias.scoreAdd)
        if bias.has("scoreMult"):
            score *= float(bias.scoreMult)
    return score
