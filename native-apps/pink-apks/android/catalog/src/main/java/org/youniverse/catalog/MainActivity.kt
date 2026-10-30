package org.youniverse.catalog

import android.app.Activity
import android.app.AlertDialog
import android.os.Bundle
import android.util.Log
import android.widget.*
import org.json.JSONArray
import org.json.JSONObject
import org.youniverse.engine.AddressEngine
import org.youniverse.engine.HerTheme
import java.io.File

/** Native admin catalog. Nothing is stored without a 13-field address; only residents can add. */
class MainActivity : Activity() {
    private lateinit var eng: AddressEngine
    private lateinit var cat: JSONObject
    private val FIELDS = listOf("planetary","dimension","gate","line","color","tone","base","degree","minute","second","arc","zodiac","house")
    private val local by lazy { File(filesDir, "intake.json") }
    private fun key(a: JSONObject) = FIELDS.joinToString(".") { val v = a.opt(it); if (v == null || v == JSONObject.NULL || v.toString().isEmpty()) "-" else v.toString() }
    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        eng = AddressEngine(this)
        cat = JSONObject(assets.open("catalog.json").bufferedReader().readText())
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 24, 24, 24) }
        val apps = cat.getJSONArray("apps")
        val extra = if (local.exists()) JSONArray(local.readText()) else JSONArray()
        col.addView(TextView(this).apply { text = "Admin Catalog — ${apps.length()} apps + ${extra.length()} added\nroot ${cat.getString("catalogAddressKey")}"; textSize = 15f })
        val name = EditText(this).apply { hint = "new item name" }; val who = EditText(this).apply { hint = "who"; setText("admin") }
        col.addView(name); col.addView(who)
        col.addView(Button(this).apply { text = "Add (address on intake)"; setOnClickListener {
            val w = eng.canWrite("root", who.text.toString().trim())
            if (!w.getBoolean("ok") || !w.getJSONObject("v").getBoolean("ok")) { toast("Refused: " + w.optJSONObject("v")?.optString("why")); return@setOnClickListener }
            val n = name.text.toString().trim(); if (n.isEmpty()) { toast("name needed"); return@setOnClickListener }
            val r = eng.intake(AddressEngine.arcFor(n), null, null)
            if (!r.getBoolean("ok")) { toast("No address, not stored: " + r.optString("err")); return@setOnClickListener }
            val v = r.getJSONObject("v")
            extra.put(JSONObject().put("name", n).put("container", "root").put("owner", who.text.toString()).put("address", v.getJSONObject("address")).put("key", v.getString("key")))
            local.writeText(extra.toString()); Log.i("AdminCatalog", "intake $n -> ${v.getString("key")}"); recreate() } })
        fun show(o: JSONObject, k: String) = col.addView(TextView(this).apply {
            text = "${o.getString("name")}\n  $k  ·  ${o.optString("status", "added")}"; textSize = 13f; layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = 12 }
            setOnClickListener { AlertDialog.Builder(this@MainActivity).setTitle(o.getString("name")).setMessage(o.toString(1)).setPositiveButton("OK", null).show() } })
        for (i in 0 until apps.length()) apps.getJSONObject(i).let { show(it, key(it.getJSONObject("address"))) }
        for (i in 0 until extra.length()) extra.getJSONObject(i).let { show(it, it.getString("key")) }
        val th = HerTheme.CATALOG; th.apply(col); for (i in 0 until col.childCount) (col.getChildAt(i) as? TextView)?.takeIf { it !is EditText && it !is Button && i > 0 }?.let { th.card(it) }
        setContentView(ScrollView(this).apply { setBackgroundColor(th.bg); addView(col) })
        Log.i("AdminCatalog", "loaded ${apps.length()} apps")
    }
    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_LONG).show()
    override fun onDestroy() { eng.close(); super.onDestroy() }
}
