# SynthWorld mobile world

Imported from the supplied SynthWorld-321-source archive. Existing utility agents,
Cynthia traits, build queue, repairs, and controlled-language execution remain in
place. This is the world foundation; avatar reconstruction engines and additional
game mechanics are not integrated yet.

Open `game/project.godot` in Godot 4.6.3 and run. Drag the world to orbit, use zoom
buttons, choose Plant tree and tap the ground. Save/Load stores tree positions,
health and world state on the device (`user://world.json`). This manual snapshot
does not yet capture camera, agent positions, or pending build orders.

Android export preset is included; exporting requires Godot Android templates,
Android SDK and a configured debug/release keystore. No APK was exported in this
workspace. The existing Simlish Android studio remains a separate app.

Validation:

```
godot --headless --path game --script res://tests/world_persistence.gd
godot --headless --path game res://Scenes/World.tscn --quit-after 240
```

Persistence test covers positions, health, world state and rejected malformed
saves without deleting current objects. Touch controls require device testing.
