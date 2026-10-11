extends Node

const SENTENCES_PATH := "res://data/cnl/sentences.txt"
var parsed_sentences: Array = []

func _ready() -> void:
    reload()

func reload() -> void:
    parsed_sentences.clear()
    if not FileAccess.file_exists(SENTENCES_PATH):
        return
    for raw_line in FileAccess.get_file_as_string(SENTENCES_PATH).split("\n"):
        var line := raw_line.strip_edges()
        if line.is_empty() or line.begins_with("#"):
            continue
        var result := SentenceParser.parse_line(line)
        if result.ok:
            parsed_sentences.append(result.ast)
        else:
            push_warning("CNL rejected: %s" % result.errors)

func execute_all(world: Node) -> int:
    var count := 0
    for ast in parsed_sentences:
        if SentenceExecutor.execute(ast, world):
            count += 1
    return count
