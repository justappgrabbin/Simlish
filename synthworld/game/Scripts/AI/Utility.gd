extends Node

class_name Utility

# Evaluates a linear expression string like "0.6*creation + 0.3*hunger + 0.1*social"
# context: Dictionary of variable -> float
static func score(expr: String, context: Dictionary) -> float:
    var e := Expression.new()
    var vars := context.keys()
    var err = e.parse(expr, vars)
    if err != OK:
        push_warning("Utility parse error for '%s'" % expr)
        return 0.0
    var values := []
    for v in vars:
        values.append(float(context.get(v, 0.0)))
    var res = e.execute(values, null, true)
    if typeof(res) != TYPE_FLOAT and typeof(res) != TYPE_INT:
        return 0.0
    return float(res)
