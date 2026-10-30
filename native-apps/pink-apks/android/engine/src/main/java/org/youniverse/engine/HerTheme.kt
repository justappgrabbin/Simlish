package org.youniverse.engine

import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.TextView

/** Colors taken from Alexis's own pages: mesh-automaton lab/ui-source CSS vars and app-catalog index.tpl.html. */
class HerTheme(val bg: Int, val panel: Int, val card: Int, val border: Int, val text: Int, val dim: Int, val accent: Int, val ok: Int, val err: Int, val mono: Boolean) {
    companion object {
        // mesh-automaton ui-source: --bg #0a0a0f --bg-panel #111118 --bg-card #1a1a24 --border #2a2a3a --text #e0e0e8 --text-dim #8888a0 --accent #6b8cff --success #4ade80 --error #f87171
        val MESH = HerTheme(c("#0a0a0f"), c("#111118"), c("#1a1a24"), c("#2a2a3a"), c("#e0e0e8"), c("#8888a0"), c("#6b8cff"), c("#4ade80"), c("#f87171"), false)
        // app-catalog index.tpl.html: #0f1117 page, #141720/#181b25 panels, #262c3d/#2b3350 borders, #3b5bdb buttons, ui-monospace
        val CATALOG = HerTheme(c("#0f1117"), c("#141720"), c("#181b25"), c("#2b3350"), c("#e0e0e8"), c("#8888a0"), c("#3b5bdb"), c("#4ade80"), c("#f87171"), true)
        private fun c(s: String) = Color.parseColor(s)
    }
    private fun box(fill: Int) = GradientDrawable().apply { setColor(fill); cornerRadius = 12f; setStroke(2, border) }
    fun apply(v: View) {
        when (v) {
            is Button -> { v.setTextColor(Color.WHITE); v.background = GradientDrawable().apply { setColor(accent); cornerRadius = 12f }; v.isAllCaps = false }
            is EditText -> { v.setTextColor(text); v.setHintTextColor(dim); v.background = box(panel); v.setPadding(20, 16, 20, 16) }
            is TextView -> v.setTextColor(text)
        }
        if (v is TextView && mono) v.typeface = Typeface.MONOSPACE
        if (v is ViewGroup) for (i in 0 until v.childCount) apply(v.getChildAt(i))
    }
    fun card(v: TextView) { v.background = box(card); v.setPadding(24, 18, 24, 18) }
}
