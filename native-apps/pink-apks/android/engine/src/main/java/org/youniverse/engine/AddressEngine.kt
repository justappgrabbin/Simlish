package org.youniverse.engine

import android.content.Context
import com.whl.quickjs.android.QuickJSLoader
import com.whl.quickjs.wrapper.QuickJSContext
import org.json.JSONObject

/** Runs Alexis's existing JS address engine + mesh core in embedded QuickJS (no WebView, no network). */
class AddressEngine(ctx: Context) {
    private val js: QuickJSContext
    val permissions: String = ctx.assets.open("permissions.json").bufferedReader().readText()
    init {
        QuickJSLoader.init()
        js = QuickJSContext.create()
        val ss = ctx.assets.open("state-space.data.json").bufferedReader().readText()
        js.globalObject.setProperty("__STATE_SPACE_JSON", ss)
        js.evaluate(ctx.assets.open("engine.js").bufferedReader().readText(), "engine.js")
        js.globalObject.setProperty("__PERMS_JSON", permissions)
        js.evaluate("var __PERMS=JSON.parse(__PERMS_JSON);")
    }
    private fun q(s: String) = JSONObject.quote(s)
    private fun call(expr: String): JSONObject = JSONObject(js.evaluate("Engine.safe(()=>$expr)") as String)

    fun gate(address13: String, input: String) = call("Engine.gate13(${q(address13)},${q(input)})")
    fun canWrite(container: String, who: String) = call("Engine.canWrite(__PERMS,${q(container)},${q(who)})")
    fun meshCycle(address13: String, who: String, container: String, input: String) =
        call("Engine.meshCycle(${q(address13)},${q(who)},__PERMS,${q(container)},${q(input)})")
    fun format13(json: String) = call("Engine.format13($json)")
    /** Async intake: QuickJS runs pending promise jobs after evaluate. */
    fun intake(arcSecond: Long, planetary: String?, dimension: String?): JSONObject {
        val c = JSONObject().put("planetary", planetary ?: JSONObject.NULL).put("dimension", dimension ?: JSONObject.NULL)
        js.evaluate("globalThis.__r=null;Engine.intake($arcSecond,$c).then(v=>__r=JSON.stringify({ok:true,v}),e=>__r=JSON.stringify({ok:false,err:String(e&&e.message)}))")
        val r = js.evaluate("__r")
        return if (r is String) JSONObject(r) else JSONObject().put("ok", false).put("err", "intake promise did not settle")
    }
    fun close() = js.destroy()

    companion object {
        /** sha256(text) mod 1,296,000 arc-seconds: same intake method as app-catalog/build.mjs. */
        fun arcFor(text: String): Long {
            val h = java.security.MessageDigest.getInstance("SHA-256").digest(text.toByteArray())
            return java.math.BigInteger(1, h).mod(java.math.BigInteger.valueOf(1_296_000)).toLong()
        }
    }
}
