extends SceneTree

func _initialize() -> void:
    call_deferred("run")

func run() -> void:
    var world = load("res://Scenes/World.tscn").instantiate()
    root.add_child(world)
    current_scene = world
    await process_frame
    world.place_tree(Vector3(12, 0, -7)).health = 42
    world.set_world_state("custom", "retained")
    assert(world.save_world())
    var count: int = world.get_node("Props").get_child_count()
    world.place_tree(Vector3.ZERO)
    assert(world.load_world())
    assert(world.get_node("Props").get_child_count() == count)
    assert(world.world_state.custom == "retained")
    var found := false
    for tree in world.get_node("Props").get_children():
        if tree.position == Vector3(12, 0, -7):
            assert(tree.health == 42)
            found = true
    assert(found)
    var file := FileAccess.open("user://world.json", FileAccess.WRITE)
    file.store_string('{"version":1,"state":{},"objects":[{"kind":"tree","position":["bad",0,0],"health":100}]}')
    file.close()
    assert(not world.load_world())
    assert(world.get_node("Props").get_child_count() == count)
    print("PASS: world save/restore and invalid-save isolation")
    quit()
