
extends Node3D
@export var max_health: int = 100
var health: int = max_health
func damage(amount: int) -> void: health = max(0, health - amount)
func repair(amount: int) -> void: health = min(max_health, health + amount)
func is_damaged() -> bool: return health < max_health
