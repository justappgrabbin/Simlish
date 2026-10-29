# Provenance and local repairs

Imported from the user-supplied MESSY_Synthia_Mesh_Studio_Source.zip:
https://drive.google.com/file/d/13doeviAgbr5NBwFK8DcboTP8uyTf_-19/view

The donor README and runtime header license this implementation MIT. Its original README and SOURCE_MAP are retained. This is a separate semantic workspace; it does not replace the character identity, canonical address authority, or empirical verification layer.

Local repair: build.js now uses a function replacement when embedding source text. JavaScript String.replace replacement strings interpret `$$` as `$`; the original builder therefore corrupted the UI selector declaration and the generated page failed to start. The function replacement preserves source bytes. A build regression parses both embedded scripts and checks the double-dollar selector survives. A link leads to the existing investigation screen.

The workspace's sample people, reality scores and graph assertions are modeled data, not verified observations. Exported checkpoints can be imported into the app's investigation screen as unverified candidates and source evidence. Donor scheduler programs are not executed by that import. Browser persistence and generated artifacts are device-local.
