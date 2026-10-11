extends CanvasLayer

var world: Node3D
var status: Label
var planting := false
var orbit := 0.0
var distance := 16.0

func _ready() -> void:
    world = get_parent()
    var panel := VBoxContainer.new()
    panel.position = Vector2(16, 16)
    panel.add_theme_constant_override("separation", 8)
    add_child(panel)
    for label in ["Plant tree", "Save world", "Load world", "Zoom in", "Zoom out"]:
        var button := Button.new()
        button.text = label
        button.custom_minimum_size = Vector2(160, 48)
        panel.add_child(button)
        button.pressed.connect(_action.bind(label))
    status = Label.new()
    status.text = "Drag to orbit. Plant, then tap the ground."
    panel.add_child(status)
    _camera()

func _action(action: String) -> void:
    match action:
        "Plant tree":
            planting = true
            status.text = "Tap the ground to plant a tree."
        "Save world":
            status.text = "World saved on this device." if world.save_world() else "Could not save world."
        "Load world":
            status.text = "World restored." if world.load_world() else "No valid saved world yet."
        "Zoom in":
            distance = maxf(5, distance - 3)
            _camera()
        "Zoom out":
            distance = minf(65, distance + 3)
            _camera()

func _unhandled_input(event: InputEvent) -> void:
    if event is InputEventScreenDrag:
        orbit -= event.relative.x * 0.008
        _camera()
    elif event is InputEventMouseMotion and event.button_mask & MOUSE_BUTTON_MASK_LEFT:
        orbit -= event.relative.x * 0.008
        _camera()
    elif planting and ((event is InputEventScreenTouch and event.pressed) or (event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT)):
        var camera: Camera3D = world.get_node("Camera3D")
        var point = Plane(Vector3.UP, 0).intersects_ray(camera.project_ray_origin(event.position), camera.project_ray_normal(event.position))
        if point != null and absf(point.x) <= 100 and absf(point.z) <= 100:
            world.place_tree(point)
            planting = false
            status.text = "Tree planted. Save to keep this world."

func _camera() -> void:
    var camera: Camera3D = world.get_node("Camera3D")
    camera.position = Vector3(sin(orbit) * distance, distance * 0.65, cos(orbit) * distance)
    camera.look_at(Vector3.ZERO)
