extends CharacterBody3D

const SPEED := 2.2

var drives := {
    "social": 0.5,
    "creation": 0.5,
    "rest": 0.5,
    "exploration": 0.5,
    "hunger": 0.2
}

var cooldowns := {}
var rng := RandomNumberGenerator.new()

func _ready() -> void:
    rng.randomize()

func _physics_process(delta: float) -> void:
    # simple wander to show life
    if velocity.length() < 0.1 and rng.randf() < 0.02:
        var dir = Vector3(rng.randf_range(-1,1), 0, rng.randf_range(-1,1)).normalized()
        velocity = dir * SPEED
    move_and_slide()

    # fake metabolism
    drives.hunger = clamp(drives.hunger + delta * 0.02, 0.0, 1.0)
    drives.rest = clamp(drives.rest - delta * 0.01, 0.0, 1.0)
    # every second, pick an action by utility
    if Engine.get_physics_frames() % 60 == 0:
        _decide_and_act()

func _decide_and_act() -> void:
    var actions = BehaviorService.get_available_actions()
    var best = null
    var best_score := -1e9
    for a in actions:
        var expr: String = a.get("score", "0")
        var s = Utility.score(expr, drives)
        if s > best_score and _is_ready(a):
            best = a
            best_score = s
    if best:
        _perform(best)

func _perform(a: Dictionary) -> void:
    var action: String = a.get("action", "")
    var cd: float = float(a.get("cooldownSec", 60))
    cooldowns[action] = Time.get_ticks_msec()/1000.0 + cd

    match action:
        "CookMeal":
            drives.hunger = max(0.0, drives.hunger - 0.6)
            drives.creation = clamp(drives.creation + 0.1, 0.0, 1.0)
        "Socialize":
            drives.social = clamp(drives.social + 0.2, 0.0, 1.0)
        "Wander":
            var dir = Vector3(rng.randf_range(-1,1),0,rng.randf_range(-1,1)).normalized()
            velocity = dir * SPEED
        _:
            pass

func _is_ready(a: Dictionary) -> bool:
    var action: String = a.get("action","")
    if not cooldowns.has(action):
        return true
    var now = Time.get_ticks_msec()/1000.0
    return now >= float(cooldowns[action])
