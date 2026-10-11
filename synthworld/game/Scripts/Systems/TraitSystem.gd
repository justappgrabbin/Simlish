
extends Node
# Loads trait definitions from /data/traits. Provides lookup by id.

var traits := {} # id -> Dictionary

func _ready() -> void:
    _load_traits()

func _load_traits() -> void:
    traits.clear()
    var dir := DirAccess.open("res://data/traits")
    if dir:
        dir.list_dir_begin()
        var f = dir.get_next()
        while f != "":
            if f.ends_with(".json"):
                var path = "res://data/traits/%s" % f
                var txt = FileAccess.get_file_as_string(path)
                var data = JSON.parse_string(txt)
                if typeof(data) == TYPE_DICTIONARY and data.has("id"):
                    traits[data.id] = data
            f = dir.get_next()
        dir.list_dir_end()

func get_trait(id: String) -> Dictionary:
    return traits.get(id, {})
