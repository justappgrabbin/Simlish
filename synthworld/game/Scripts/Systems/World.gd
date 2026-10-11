extends Node3D

const AGENT_SCENE := preload("res://Scenes/Agents/Agent.tscn")
const TREE_SCENE := preload("res://Scenes/Props/Tree.tscn")
const DUMMY_DAMAGE_INTERVAL := 3.0

@onready var props_root := $Props
var _timer := 0.0
var world_state := {"mood.harmony": 0.7}

func _ready() -> void:
    var controls := CanvasLayer.new()
    controls.set_script(load("res://Scripts/Systems/WorldControls.gd"))
    add_child(controls)
    # Preserve the original utility-driven living agent alongside Cynthia.
    var agent := AGENT_SCENE.instantiate()
    agent.name = "Generator"
    add_child(agent)
    agent.global_position = Vector3(-3, 0, 0)
    place_tree(Vector3(5, 0, 5))
    # Execute supplied CNL after the world is fully ready.
    call_deferred("_execute_seed_sentences")

func _process(delta: float) -> void:
    _timer += delta
    if _timer >= DUMMY_DAMAGE_INTERVAL:
        _timer = 0.0
        var damaged = _get_random_prop()
        if damaged and damaged.has_method("damage"):
            damaged.damage(5)

func _execute_seed_sentences() -> void:
    SentenceService.execute_all(self)

func place_tree(pos: Vector3) -> Node3D:
    var tree := TREE_SCENE.instantiate()
    props_root.add_child(tree)
    tree.global_position = pos
    return tree

func repair_props_near(pos: Vector3, radius: float) -> int:
    var count := 0
    for prop in props_root.get_children():
        if prop.has_method("is_damaged") and prop.is_damaged() and prop.global_position.distance_to(pos) <= radius:
            prop.repair(10)
            count += 1
    return count

func has_damaged_props() -> bool:
    for prop in props_root.get_children():
        if prop.has_method("is_damaged") and prop.is_damaged():
            return true
    return false

func set_world_state(key: String, value) -> void:
    world_state[key] = value

func _get_random_prop():
    var c := props_root.get_child_count()
    if c == 0:
        return null
    return props_root.get_child(randi() % c)

# Store resident world objects separately from the agent's build queue.
func save_world() -> bool:
    var objects: Array = []
    for prop in props_root.get_children():
        var p: Vector3 = prop.position
        objects.append({"kind": "tree", "position": [p.x, p.y, p.z], "health": prop.health})
    var file := FileAccess.open("user://world.json", FileAccess.WRITE)
    if file == null:
        return false
    file.store_string(JSON.stringify({"version": 1, "state": world_state, "objects": objects}))
    return file.get_error() == OK

func load_world() -> bool:
    if not FileAccess.file_exists("user://world.json"):
        return false
    var data = JSON.parse_string(FileAccess.get_file_as_string("user://world.json"))
    if not data is Dictionary or data.get("version") != 1 or not data.get("objects") is Array or not data.get("state") is Dictionary:
        return false
    # Validate every object before altering the running world.
    for object in data.objects:
        if not object is Dictionary or object.get("kind") != "tree":
            return false
        var p = object.get("position")
        if not p is Array or p.size() != 3:
            return false
        for coordinate in p:
            if not (coordinate is float or coordinate is int) or not is_finite(float(coordinate)):
                return false
        if not (object.get("health") is float or object.get("health") is int):
            return false
    for prop in props_root.get_children():
        props_root.remove_child(prop)
        prop.queue_free()
    world_state = data.state
    for object in data.objects:
        var p: Array = object.position
        var tree := place_tree(Vector3(p[0], p[1], p[2]))
        tree.health = clampi(int(object.health), 0, tree.max_health)
    return true
