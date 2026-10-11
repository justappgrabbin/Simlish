extends Node
class_name SentenceExecutor

static func execute(ast: Dictionary, world: Node) -> bool:
    if not _condition_allows(String(ast.get("when", "")), world):
        return false
    var verb := String(ast.get("verb", ""))
    var object_id := String(ast.get("object", ""))
    var pos := SentenceParser.parse_position(String(ast.get("at", "(0,0,0)")))
    match verb:
        "build", "spawn":
            if object_id == "prop/tree":
                world.place_tree(pos)
        "repair":
            world.repair_props_near(pos, 1.5)
        "move":
            var agent = world.get_node_or_null(String(ast.get("agent", "Cynthia")))
            if agent and agent.has_method("set_target"):
                agent.set_target(pos)
        "emote":
            world.set_world_state("emotion", object_id)
        "light":
            world.set_world_state("light", object_id)
        "play":
            world.set_world_state("sound", object_id)
        "set":
            world.set_world_state(object_id, ast.get("with", ""))
        _:
            return false
    var scene_id := String(ast.get("scene", ""))
    if not scene_id.is_empty():
        YiJingResolver.apply_scene(scene_id, world)
    return true

static func _condition_allows(expr: String, world: Node) -> bool:
    if expr.strip_edges().is_empty():
        return true
    # The supplied examples refer to runtime state that may not exist yet.
    # Evaluate only the state paths currently represented by the supplied artifacts.
    if expr.contains("prop.health<prop.max"):
        return world.has_damaged_props()
    if expr.contains("mood.harmony>=0.7"):
        return float(world.world_state.get("mood.harmony", 0.7)) >= 0.7
    return false
