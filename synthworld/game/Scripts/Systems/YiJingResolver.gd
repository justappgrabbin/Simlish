extends Node
class_name YiJingResolver

const SCENES_PATH := "res://data/yijing/scenes.json"

static func get_scene(scene_id: String) -> Dictionary:
    if not FileAccess.file_exists(SCENES_PATH):
        return {}
    var data = JSON.parse_string(FileAccess.get_file_as_string(SCENES_PATH))
    if typeof(data) != TYPE_DICTIONARY:
        return {}
    return data.get(scene_id, {})

static func apply_scene(scene_id: String, world: Node) -> void:
    var scene := get_scene(scene_id)
    for directive in scene.get("directives", []):
        var ast := {
            "agent": "Cynthia",
            "verb": directive.get("verb", ""),
            "object": directive.get("object", ""),
            "with": JSON.stringify(directive.get("with", {}))
        }
        SentenceExecutor.execute(ast, world)
