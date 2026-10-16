import { EventBus } from './EventBus'

export const MorphingSubstrate = {
  bgCanvas: null,
  bgCtx: null,
  chatCanvas: null,
  chatCtx: null,
  gridW: 0,
  gridH: 0,
  spins: [],
  nextSpins: [],
  currentHexagram: 1,
  targetHexagram: 1,
  hexagramTransition: 0,
  temperature: 2.5,
  coupling: 1.0,
  field: 0.0,
  hexagramPalettes: [],
  animFrame: null,
  chatIntensity: 0,
  chatRipples: [],

  init() {
    this.initPalettes()
    this.initBackground()
    this.initChatSubstrate()
    this.startAnimation()

    EventBus.on('body:phase_changed', (data) => this.onPhaseChange(data))
    EventBus.on('agent:message', () => this.onAgentActivity())
    EventBus.on('agent:speaking', () => this.onAgentActivity())
    EventBus.on('app:opened', () => this.onAppActivity())
    EventBus.on('file:updated', () => this.onFileActivity())
  },

  initPalettes() {
    const phi = 1.618033988749
    for (let i = 0; i < 64; i++) {
      const hue = ((i * phi * 137.5) % 360)
      const sat = 40 + (i % 4) * 15
      const light = 15 + (i % 3) * 10
      const accentHue = (hue + 180) % 360
      this.hexagramPalettes.push({
        base: `hsl(${hue}, ${sat}%, ${light}%)`,
        accent: `hsl(${accentHue}, ${sat + 20}%, ${light + 20}%)`,
        glow: `hsl(${hue}, ${sat}%, ${light + 30}%)`,
        energy: i % 8,
      })
    }
  },

  initBackground() {
    this.bgCanvas = document.createElement('canvas')
    this.bgCanvas.id = 'morphing-bg'
    this.bgCanvas.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;'
    document.body.insertBefore(this.bgCanvas, document.body.firstChild)
    this.bgCtx = this.bgCanvas.getContext('2d')
    this.resizeBackground()
    window.addEventListener('resize', () => this.resizeBackground())
    this.initSpinGrid()
  },

  resizeBackground() {
    this.bgCanvas.width = window.innerWidth
    this.bgCanvas.height = window.innerHeight
    this.gridW = Math.ceil(window.innerWidth / 8)
    this.gridH = Math.ceil(window.innerHeight / 8)
    this.initSpinGrid()
  },

  initSpinGrid() {
    const size = this.gridW * this.gridH
    this.spins = new Float32Array(size)
    this.nextSpins = new Float32Array(size)
    for (let i = 0; i < size; i++) {
      const x = i % this.gridW
      const y = Math.floor(i / this.gridW)
      const domain = Math.sin(x * 0.05) * Math.cos(y * 0.05)
      this.spins[i] = domain + (Math.random() - 0.5) * 0.3
    }
  },

  initChatSubstrate() {
    const panel = document.getElementById('agent-panel')
    if (!panel) return
    this.chatCanvas = document.createElement('canvas')
    this.chatCanvas.id = 'morphing-chat'
    this.chatCanvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;border-radius:12px;opacity:0.3;'
    panel.insertBefore(this.chatCanvas, panel.firstChild)
    this.chatCtx = this.chatCanvas.getContext('2d')
    this.resizeChat()
    new ResizeObserver(() => this.resizeChat()).observe(panel)
  },

  resizeChat() {
    const panel = document.getElementById('agent-panel')
    if (!panel || !this.chatCanvas) return
    this.chatCanvas.width = panel.clientWidth
    this.chatCanvas.height = panel.clientHeight
  },

  updateSpins() {
    const w = this.gridW
    const h = this.gridH
    const J = this.coupling
    const T = this.temperature
    const H = this.field

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x
        const s = this.spins[i]
        let neighborSum = 0
        neighborSum += this.spins[((y - 1 + h) % h) * w + x]
        neighborSum += this.spins[((y + 1) % h) * w + x]
        neighborSum += this.spins[y * w + ((x - 1 + w) % w)]
        neighborSum += this.spins[y * w + ((x + 1) % w)]
        const energy = -J * s * neighborSum - H * s
        const target = neighborSum / 4
        const noise = (Math.random() - 0.5) * T * 0.1
        this.nextSpins[i] = s * 0.7 + (target + noise + H * 0.1) * 0.3
        this.nextSpins[i] = Math.max(-1, Math.min(1, this.nextSpins[i]))
      }
    }
    const temp = this.spins
    this.spins = this.nextSpins
    this.nextSpins = temp
  },

  setHexagram(n) {
    if (n < 1 || n > 64) return
    if (n === this.targetHexagram) return
    this.targetHexagram = n
    this.hexagramTransition = 0
    const energy = (n - 1) % 8
    this.temperature = 1.5 + energy * 0.3
    this.coupling = 0.8 + (7 - energy) * 0.1
  },

  updateHexagramTransition() {
    if (this.currentHexagram === this.targetHexagram) return
    this.hexagramTransition += 0.02
    if (this.hexagramTransition >= 1) {
      this.currentHexagram = this.targetHexagram
      this.hexagramTransition = 0
    }
  },

  getHexagramForPhase(phase) {
    const map = {
      dormant: 1,
      observant: 2,
      active: 14,
      dreaming: 52,
      growing: 42,
      resonant: 63,
    }
    return map[phase] || 1
  },

  renderBackground() {
    const ctx = this.bgCtx
    const w = this.bgCanvas.width
    const h = this.bgCanvas.height
    const gw = this.gridW
    const gh = this.gridH
    const cellW = w / gw
    const cellH = h / gh
    const currPalette = this.hexagramPalettes[this.currentHexagram - 1]
    const targetPalette = this.hexagramPalettes[this.targetHexagram - 1]
    const t = this.hexagramTransition

    ctx.fillStyle = '#030408'
    ctx.fillRect(0, 0, w, h)

    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const i = y * gw + x
        const spin = this.spins[i]
        const intensity = (spin + 1) / 2
        const px = x * cellW
        const py = y * cellH
        const hue = this.lerp(this.parseHue(currPalette.base), this.parseHue(targetPalette.base), t) + intensity * 30
        const sat = this.lerp(this.parseSat(currPalette.base), this.parseSat(targetPalette.base), t)
        const light = this.lerp(this.parseLight(currPalette.base), this.parseLight(targetPalette.base), t) + intensity * 20
        const alpha = 0.3 + intensity * 0.4
        ctx.fillStyle = `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`
        ctx.fillRect(px, py, cellW + 1, cellH + 1)
      }
    }

    this.renderGlow(ctx, w, h, currPalette, targetPalette, t)
  },

  renderGlow(ctx, w, h, currPal, targetPal, t) {
    const gw = this.gridW
    const gh = this.gridH
    for (let y = 2; y < gh - 2; y += 3) {
      for (let x = 2; x < gw - 2; x += 3) {
        const i = y * gw + x
        const spin = Math.abs(this.spins[i])
        if (spin > 0.7) {
          const px = (x / gw) * w
          const py = (y / gh) * h
          const radius = 30 + spin * 50
          const hue = this.lerp(this.parseHue(currPal.accent), this.parseHue(targetPal.accent), t)
          const grad = ctx.createRadialGradient(px, py, 0, px, py, radius)
          grad.addColorStop(0, `hsla(${hue}, 80%, 60%, ${spin * 0.15})`)
          grad.addColorStop(1, 'transparent')
          ctx.fillStyle = grad
          ctx.beginPath()
          ctx.arc(px, py, radius, 0, Math.PI * 2)
          ctx.fill()
        }
      }
    }
  },

  renderChatSubstrate() {
    if (!this.chatCtx || !this.chatCanvas) return
    const ctx = this.chatCtx
    const w = this.chatCanvas.width
    const h = this.chatCanvas.height
    ctx.clearRect(0, 0, w, h)
    const palette = this.hexagramPalettes[this.currentHexagram - 1]
    const time = Date.now() * 0.001
    const lines = 5 + Math.floor(this.chatIntensity * 10)

    for (let i = 0; i < lines; i++) {
      const y = (h / (lines + 1)) * (i + 1)
      const amplitude = 5 + this.chatIntensity * 15
      const frequency = 0.01 + i * 0.005
      const speed = 0.5 + i * 0.2
      ctx.beginPath()
      ctx.strokeStyle = `hsla(${this.parseHue(palette.accent)}, 70%, 50%, ${0.1 + this.chatIntensity * 0.2})`
      ctx.lineWidth = 1 + this.chatIntensity
      for (let x = 0; x < w; x += 2) {
        const wave = Math.sin(x * frequency + time * speed + i) * amplitude
        const yPos = y + wave
        if (x === 0) ctx.moveTo(x, yPos)
        else ctx.lineTo(x, yPos)
      }
      ctx.stroke()
    }

    this.chatRipples = this.chatRipples.filter((r) => {
      r.age += 0.016
      r.radius += 2
      if (r.age > 2) return false
      const alpha = (1 - r.age / 2) * 0.3
      ctx.beginPath()
      ctx.strokeStyle = `hsla(${this.parseHue(palette.glow)}, 80%, 60%, ${alpha})`
      ctx.lineWidth = 2
      ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2)
      ctx.stroke()
      return true
    })

    this.chatIntensity *= 0.995
  },

  onPhaseChange(data) {
    const hex = this.getHexagramForPhase(data.to)
    this.setHexagram(hex)
    this.field = (Math.random() - 0.5) * 0.5
    setTimeout(() => {
      this.field = 0
    }, 1000)
  },

  onAgentActivity() {
    this.chatIntensity = Math.min(1, this.chatIntensity + 0.3)
    if (this.chatCanvas) {
      this.chatRipples.push({
        x: this.chatCanvas.width / 2,
        y: this.chatCanvas.height / 2,
        radius: 5,
        age: 0,
      })
    }
    this.field = (Math.random() - 0.5) * 0.3
  },

  onAppActivity() {
    this.field = (Math.random() - 0.5) * 0.4
    this.chatIntensity = Math.min(1, this.chatIntensity + 0.2)
    setTimeout(() => {
      this.field = 0
    }, 500)
  },

  onFileActivity() {
    const w = this.gridW
    const h = this.gridH
    for (let i = 0; i < 50; i++) {
      const x = Math.floor(Math.random() * w)
      const y = Math.floor(Math.random() * h)
      const idx = y * w + x
      this.spins[idx] = Math.random() > 0.5 ? 1 : -1
    }
  },

  animate() {
    this.updateSpins()
    this.updateHexagramTransition()
    this.renderBackground()
    this.renderChatSubstrate()
    this.animFrame = requestAnimationFrame(() => this.animate())
  },

  startAnimation() {
    this.animate()
  },

  stopAnimation() {
    if (this.animFrame) {
      cancelAnimationFrame(this.animFrame)
      this.animFrame = null
    }
  },

  lerp(a, b, t) {
    return a + (b - a) * t
  },

  parseHue(hsl) {
    const match = hsl.match(/hsl\((\d+)/)
    return match ? parseInt(match[1]) : 0
  },

  parseSat(hsl) {
    const match = hsl.match(/hsl\(\d+,\s*(\d+)%/)
    return match ? parseInt(match[1]) : 50
  },

  parseLight(hsl) {
    const match = hsl.match(/hsl\(\d+,\s*\d+%,\s*(\d+)%/)
    return match ? parseInt(match[1]) : 50
  },

  getHexagramInfo() {
    const hexagrams = [
      '', '乾 Qián — The Creative', '坤 Kūn — The Receptive', '屯 Zhūn — Difficulty',
      '蒙 Méng — Youthful Folly', '需 Xū — Waiting', '訟 Sòng — Conflict',
      '師 Shī — The Army', '比 Bǐ — Holding Together', '小畜 Xiǎo Chù — Small Taming',
      '履 Lǚ — Treading', '泰 Tài — Peace', '否 Pǐ — Standstill',
      '同人 Tóng Rén — Fellowship', '大有 Dà Yǒu — Possession', '謙 Qiān — Modesty',
      '豫 Yù — Enthusiasm', '隨 Suí — Following', '蠱 Gǔ — Work on Decay',
      '臨 Lín — Approach', '觀 Guān — Contemplation', '噬嗑 Shì Kè — Biting Through',
      '賁 Bì — Grace', '剝 Bō — Splitting Apart', '復 Fù — Return',
      '無妄 Wú Wàng — Innocence', '大畜 Dà Chù — Great Taming', '頤 Yí — Nourishment',
      '大過 Dà Guò — Great Preponderance', '坎 Kǎn — The Abysmal', '離 Lí — The Clinging',
      '咸 Xián — Influence', '恆 Héng — Duration', '遯 Dùn — Retreat',
      '大壯 Dà Zhuàng — Great Power', '晉 Jìn — Progress', '明夷 Míng Yí — Darkening',
      '家人 Jiā Rén — Family', '睽 Kuí — Opposition', '蹇 Jiǎn — Obstruction',
      '解 Xiè — Deliverance', '損 Sǔn — Decrease', '益 Yì — Increase',
      '夬 Guài — Breakthrough', '姤 Gòu — Coming to Meet', '萃 Cuì — Gathering',
      '升 Shēng — Pushing Upward', '困 Kùn — Oppression', '井 Jǐng — The Well',
      '革 Gé — Revolution', '鼎 Dǐng — The Cauldron', '震 Zhèn — The Arousing',
      '艮 Gèn — Keeping Still', '漸 Jiàn — Development', '歸妹 Guī Mèi — Marrying',
      '豐 Fēng — Abundance', '旅 Lǚ — The Wanderer', '巽 Xùn — The Gentle',
      '兌 Duì — The Joyous', '渙 Huàn — Dispersion', '節 Jié — Limitation',
      '中孚 Zhōng Fú — Inner Truth', '小過 Xiǎo Guò — Small Preponderance',
      '既濟 Jì Jì — Already Fulfilled', '未濟 Wèi Jì — Not Yet Fulfilled'
    ]
    return {
      current: hexagrams[this.currentHexagram] || 'Unknown',
      target: hexagrams[this.targetHexagram] || 'Unknown',
      transition: this.hexagramTransition,
      temperature: this.temperature.toFixed(2),
      coupling: this.coupling.toFixed(2),
    }
  },
}
