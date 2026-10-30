package org.youniverse.engine
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Test
class EngineTest {
    private val e by lazy { AddressEngine(InstrumentationRegistry.getInstrumentation().targetContext) }
    private val ok = "-.-.15.2.6.6.4.20.37.6.290227.Gemini.2"
    @Test fun gatePasses() { val r = e.gate(ok, "x"); assertTrue(r.toString(), r.getBoolean("ok")); assertEquals(ok, r.getJSONObject("v").getString("address")) }
    @Test fun rejects5() = assertFalse(e.gate("15.2.6.6.4", "x").getBoolean("ok"))
    @Test fun rejects15() = assertFalse(e.gate("$ok.-.-", "x").getBoolean("ok"))
    @Test fun badLine() = assertFalse(e.gate("-.-.15.9.6.6.4.-.-.-.-.-.-", "x").getBoolean("ok"))
    @Test fun guestRefused() = assertFalse(e.meshCycle(ok, "guest", "root", "pat").getBoolean("ok"))
    @Test fun meshRuns() { val r = e.meshCycle(ok, "admin", "root", "pat"); assertTrue(r.toString(), r.getBoolean("ok")); assertTrue(r.getJSONObject("v").getInt("events") > 0) }
    @Test fun intakeMatchesCatalog() { val r = e.intake(290227, null, null); assertTrue(r.toString(), r.getBoolean("ok")); assertEquals(ok, r.getJSONObject("v").getString("key")) }
}
