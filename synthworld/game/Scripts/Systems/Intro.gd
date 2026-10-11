extends Control

@onready var title: Label = $Center/Stack/Title
@onready var subtitle: Label = $Center/Stack/Subtitle
@onready var hint: Label = $Center/Stack/Hint
var elapsed := 0.0
var leaving := false

func _ready() -> void:
    title.modulate.a = 0.0
    subtitle.modulate.a = 0.0
    hint.modulate.a = 0.0
    var tween = create_tween()
    tween.tween_property(title, "modulate:a", 1.0, 0.65)
    tween.tween_interval(0.25)
    tween.tween_property(subtitle, "modulate:a", 1.0, 0.55)
    tween.tween_interval(0.35)
    tween.tween_property(hint, "modulate:a", 0.75, 0.45)

func _process(delta: float) -> void:
    elapsed += delta
    $Stars.rotation += delta * 0.035
    if elapsed > 4.2 and not leaving:
        _enter_world()

func _unhandled_input(event: InputEvent) -> void:
    if (event is InputEventScreenTouch and event.pressed) or (event is InputEventMouseButton and event.pressed) or (event is InputEventKey and event.pressed):
        _enter_world()

func _enter_world() -> void:
    if leaving:
        return
    leaving = true
    var tween = create_tween()
    tween.tween_property(self, "modulate:a", 0.0, 0.35)
    tween.tween_callback(func(): get_tree().change_scene_to_file("res://Scenes/World.tscn"))
