package org.youniverse.mesh

import android.app.Activity
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.*
import org.youniverse.engine.AddressEngine
import org.youniverse.engine.HerTheme

/** Native mesh automaton: every cycle passes the 13-field address gate + resident check first. */
class MainActivity : Activity() {
    private lateinit var eng: AddressEngine
    private val h = Handler(Looper.getMainLooper())
    private var looping = false
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        eng = AddressEngine(this)
        val pad = 24
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, pad, pad, pad) }
        fun field(hint: String, v: String) = EditText(this).apply { this.hint = hint; setText(v); textSize = 13f }.also { col.addView(it) }
        col.addView(TextView(this).apply { text = "Mesh Automaton (on-device)"; textSize = 20f })
        val addr = field("13-field address", "-.-.15.2.6.6.4.20.37.6.290227.Gemini.2")
        val who = field("who is writing", "admin")
        val cont = field("container", "root")
        val inp = field("input", "pat")
        val log = TextView(this).apply { textSize = 12f; setTextIsSelectable(true) }
        fun cycle() {
            val r = eng.meshCycle(addr.text.toString().trim(), who.text.toString().trim(), cont.text.toString().trim(), inp.text.toString())
            val line = if (r.getBoolean("ok")) { val v = r.getJSONObject("v"); "OK ${v.getString("address")} run=${v.getString("runId")} events=${v.getInt("events")} total runs=${v.getInt("runs")}" } else "REFUSED ${r.getString("err")}"
            Log.i("MeshAutomaton", line); log.setTextColor(if (r.getBoolean("ok")) HerTheme.MESH.ok else HerTheme.MESH.err); log.text = line + "\n" + log.text
        }
        val row = LinearLayout(this)
        row.addView(Button(this).apply { text = "Run cycle"; setOnClickListener { cycle() } })
        row.addView(Button(this).apply { text = "Auto 1/s"; setOnClickListener {
            looping = !looping; text = if (looping) "Stop" else "Auto 1/s"
            val tick = object : Runnable { override fun run() { if (looping) { cycle(); h.postDelayed(this, 1000) } } }; h.post(tick) } })
        col.addView(row)
        col.addView(log)
        val th = HerTheme.MESH; th.apply(col); th.card(log); log.setTextColor(th.dim)
        setContentView(ScrollView(this).apply { setBackgroundColor(th.bg); addView(col) })
        if (intent?.getBooleanExtra("selftest", false) == true) { cycle(); who.setText("guest"); cycle(); who.setText("admin") }
    }
    override fun onDestroy() { looping = false; eng.close(); super.onDestroy() }
}
