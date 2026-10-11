
extends Node
var nudges: Array = []
func _ready() -> void:
    var path = "res://data/nudges.json"
    if FileAccess.file_exists(path):
        var txt = FileAccess.get_file_as_string(path)
        var data = JSON.parse_string(txt)
        if typeof(data) == TYPE_ARRAY: nudges = data
