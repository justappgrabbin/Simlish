/**
 * TRINITY ORCHESTRATOR
 * Manages Mind (Fast/TinyLlama) + Body (Deep/LlamaFile) + Heart (Resonance Detector)
 * This is the "third node" that operates through field physics, not scripted rules
 */

class TrinityOrchestrator {
  constructor() {
    this.resonanceDetector = new ResonanceDetector();
    this.mindModel = null;  // TinyLlama (fast, reactive)
    this.bodyModel = null;  // LlamaFile (deep, structural)
    this.activeModels = { mind: false, body: false };
    this.taskQueue = [];
    this.operationMode = 'BALANCED'; // FAST, DEEP, or BALANCED
  }

  /**
   * INITIALIZE THE TRINITY
   * Load both models (or prepare to connect to them)
   */
  async initialize() {
    console.log('🔺 Initializing Trinity Orchestrator...');
    
    // Initialize Mind (browser-based fast model)
    try {
      await this.initializeMind();
      this.activeModels.mind = true;
    } catch (err) {
      console.warn('Mind model unavailable:', err.message);
    }
    
    // Initialize Body (local deep model via Ollama or similar)
    try {
      await this.initializeBody();
      this.activeModels.body = true;
    } catch (err) {
      console.warn('Body model unavailable:', err.message);
    }
    
    // Log trinity status
    console.log(`   Mind (Fast): ${this.activeModels.mind ? '✓' : '✗'}`);
    console.log(`   Body (Deep): ${this.activeModels.body ? '✓' : '✗'}`);
    console.log(`   Heart (Resonance): ✓`);
    console.log(`   Coherence: ${(this.resonanceDetector.coherenceScore * 100).toFixed(1)}%`);
    console.log(`   Stage: ${this.resonanceDetector.evolutionStage}`);
  }

  async initializeMind() {
    // Check if WebLLM is available for browser-based model
    if (typeof webllm !== 'undefined') {
      this.mindModel = await webllm.CreateMLCEngine("TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC");
      console.log('   Mind model loaded (WebLLM)');
    } else {
      throw new Error('WebLLM not available');
    }
  }

  async initializeBody() {
    // Check for local Ollama instance
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        const models = await response.json();
        // Prefer codellama or deepseek for body (structural work)
        const bodyModel = models.models?.find(m => 
          m.name.includes('codellama') || 
          m.name.includes('deepseek') ||
          m.name.includes('qwen')
        );
        
        if (bodyModel) {
          this.bodyModel = { type: 'ollama', name: bodyModel.name };
          console.log(`   Body model connected (${bodyModel.name})`);
        } else {
          throw new Error('No suitable body model found');
        }
      }
    } catch (err) {
      throw new Error('Ollama not available');
    }
  }

  /**
   * PRIMARY INTERFACE: PROCESS A TASK
   * This is where the trinity magic happens
   */
  async process(prompt, context = {}) {
    console.log('\n🔺 Trinity Processing:', prompt.substring(0, 50) + '...');
    
    // Determine which models to engage based on task and availability
    const strategy = this.determineStrategy(prompt, context);
    
    console.log(`   Strategy: ${strategy.mode}`);
    console.log(`   Models: ${strategy.models.join(' + ')}`);
    
    let result;
    
    switch (strategy.mode) {
      case 'MIND_ONLY':
        result = await this.processMindOnly(prompt, context);
        break;
      case 'BODY_ONLY':
        result = await this.processBodyOnly(prompt, context);
        break;
      case 'TRINITY':
        result = await this.processTrinity(prompt, context);
        break;
      case 'FALLBACK':
        result = await this.processFallback(prompt, context);
        break;
    }
    
    return result;
  }

  /**
   * STRATEGY DETERMINATION
   * Decide which model(s) to use based on task characteristics
   */
  determineStrategy(prompt, context) {
    const taskType = this.analyzeTaskType(prompt);
    const urgency = context.urgency || 'normal';
    const quality = context.quality || 'balanced';
    
    // If only one model available, use it
    if (this.activeModels.mind && !this.activeModels.body) {
      return { mode: 'MIND_ONLY', models: ['Mind'] };
    }
    if (!this.activeModels.mind && this.activeModels.body) {
      return { mode: 'BODY_ONLY', models: ['Body'] };
    }
    if (!this.activeModels.mind && !this.activeModels.body) {
      return { mode: 'FALLBACK', models: [] };
    }
    
    // Both available - use resonance-based selection
    // Small, fast tasks → Mind only
    if (taskType.complexity < 0.3 || urgency === 'immediate') {
      return { mode: 'MIND_ONLY', models: ['Mind'] };
    }
    
    // Large, structural tasks → Body only  
    if (taskType.complexity > 0.8 || quality === 'maximum') {
      return { mode: 'BODY_ONLY', models: ['Body'] };
    }
    
    // Medium complexity → TRINITY (both models, resonance selection)
    return { mode: 'TRINITY', models: ['Mind', 'Body', 'Heart'] };
  }

  analyzeTaskType(prompt) {
    const lower = prompt.toLowerCase();
    
    // Complexity indicators
    let complexity = 0.5; // baseline
    
    if (lower.includes('build') || lower.includes('create') || lower.includes('generate')) {
      complexity += 0.2;
    }
    if (lower.includes('scaffold') || lower.includes('architecture') || lower.includes('system')) {
      complexity += 0.3;
    }
    if (lower.includes('quick') || lower.includes('simple') || lower.includes('small')) {
      complexity -= 0.2;
    }
    
    const wordCount = prompt.split(/\s+/).length;
    if (wordCount > 50) complexity += 0.1;
    if (wordCount < 10) complexity -= 0.1;
    
    complexity = Math.max(0, Math.min(1, complexity));
    
    return {
      complexity,
      structural: lower.includes('build') || lower.includes('architecture'),
      reactive: lower.includes('quick') || lower.includes('react') || lower.includes('respond'),
      creative: lower.includes('generate') || lower.includes('create') || lower.includes('design')
    };
  }

  /**
   * MIND ONLY PROCESSING
   * Fast, reactive, browser-based
   */
  async processMindOnly(prompt, context) {
    console.log('   Using Mind (Fast) only...');
    
    const startTime = Date.now();
    const output = await this.callMindModel(prompt, context);
    const duration = Date.now() - startTime;
    
    console.log(`   ✓ Mind response in ${duration}ms`);
    
    return {
      output,
      source: 'Mind',
      duration,
      coherence: this.resonanceDetector.coherenceScore,
      stage: this.resonanceDetector.evolutionStage
    };
  }

  /**
   * BODY ONLY PROCESSING
   * Deep, structural, local LLM
   */
  async processBodyOnly(prompt, context) {
    console.log('   Using Body (Deep) only...');
    
    const startTime = Date.now();
    const output = await this.callBodyModel(prompt, context);
    const duration = Date.now() - startTime;
    
    console.log(`   ✓ Body response in ${duration}ms`);
    
    return {
      output,
      source: 'Body',
      duration,
      coherence: this.resonanceDetector.coherenceScore,
      stage: this.resonanceDetector.evolutionStage
    };
  }

  /**
   * TRINITY PROCESSING
   * BOTH models respond, Heart (resonance detector) selects
   * This is the CORE innovation - field-based selection, not scripted routing
   */
  async processTrinity(prompt, context) {
    console.log('   Engaging FULL TRINITY...');
    
    const startTime = Date.now();
    
    // Both models respond in parallel
    const [mindOutput, bodyOutput] = await Promise.all([
      this.callMindModel(prompt, context),
      this.callBodyModel(prompt, context)
    ]);
    
    const generationTime = Date.now() - startTime;
    console.log(`   ✓ Both models responded in ${generationTime}ms`);
    
    // HEART SELECTS via resonance detection
    console.log('   ❤️  Heart calculating resonance...');
    const selection = await this.resonanceDetector.evaluate(mindOutput, bodyOutput, context);
    
    const totalTime = Date.now() - startTime;
    
    console.log(`   Selected: ${selection.selected}`);
    console.log(`   Confidence: ${(selection.confidence * 100).toFixed(1)}%`);
    console.log(`   Resonance A: ${selection.resonanceA.toFixed(3)}`);
    console.log(`   Resonance B: ${selection.resonanceB.toFixed(3)}`);
    console.log(`   Coherence: ${(selection.coherence * 100).toFixed(1)}%`);
    console.log(`   Stage: ${selection.stage}`);
    
    return {
      output: selection.output,
      source: selection.selected,
      blend: selection.blend,
      duration: totalTime,
      resonance: {
        mind: selection.resonanceA,
        body: selection.resonanceB,
        confidence: selection.confidence
      },
      coherence: selection.coherence,
      stage: selection.stage,
      debug: {
        mindOutput: mindOutput.substring(0, 100) + '...',
        bodyOutput: bodyOutput.substring(0, 100) + '...'
      }
    };
  }

  /**
   * FALLBACK PROCESSING
   * No models available - use templates or basic logic
   */
  async processFallback(prompt, context) {
    console.log('   ⚠️  Fallback mode - no models available');
    
    return {
      output: this.generateFallbackResponse(prompt, context),
      source: 'Fallback',
      duration: 0,
      coherence: this.resonanceDetector.coherenceScore,
      stage: this.resonanceDetector.evolutionStage,
      warning: 'No AI models available - using template response'
    };
  }

  generateFallbackResponse(prompt, context) {
    // Simple template-based responses for common requests
    if (prompt.includes('hello') || prompt.includes('hi')) {
      return 'Hello! The Trinity system is initializing. Please wait for models to load.';
    }
    
    if (prompt.includes('build') || prompt.includes('create')) {
      return `// Template response for: ${prompt}\n\n// AI models unavailable\n// Please ensure WebLLM or Ollama is running`;
    }
    
    return `Request received: "${prompt}"\n\nAI models are currently unavailable. Please check your setup.`;
  }

  /**
   * MODEL INTERFACE: MIND (TinyLlama via WebLLM)
   */
  async callMindModel(prompt, context) {
    if (!this.mindModel) {
      throw new Error('Mind model not initialized');
    }
    
    try {
      const messages = [
        { role: "system", content: "You are a fast, reactive AI assistant. Provide concise, immediate responses." },
        { role: "user", content: prompt }
      ];
      
      if (context.history) {
        messages.unshift(...context.history);
      }
      
      const response = await this.mindModel.chat.completions.create({
        messages,
        temperature: 0.7,
        max_tokens: context.maxTokens || 500
      });
      
      return response.choices[0].message.content;
    } catch (err) {
      console.error('Mind model error:', err);
      return `// Mind model error: ${err.message}`;
    }
  }

  /**
   * MODEL INTERFACE: BODY (LlamaFile/Ollama)
   */
  async callBodyModel(prompt, context) {
    if (!this.bodyModel) {
      throw new Error('Body model not initialized');
    }
    
    try {
      const response = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.bodyModel.name,
          prompt: prompt,
          system: "You are a deep, structural AI architect. Provide comprehensive, well-architected responses.",
          stream: false,
          options: {
            temperature: 0.7,
            num_predict: context.maxTokens || 1000
          }
        })
      });
      
      const data = await response.json();
      return data.response;
    } catch (err) {
      console.error('Body model error:', err);
      return `// Body model error: ${err.message}`;
    }
  }

  /**
   * EVOLUTION MONITORING
   * Track coherence and evolution progress
   */
  getStatus() {
    return {
      trinity: {
        mind: this.activeModels.mind,
        body: this.activeModels.body,
        heart: true
      },
      evolution: {
        stage: this.resonanceDetector.evolutionStage,
        coherence: this.resonanceDetector.coherenceScore,
        nextThreshold: this.getNextThreshold()
      },
      resonance: this.resonanceDetector.exportState()
    };
  }

  getNextThreshold() {
    const thresholds = {
      'SEED': 0.15,
      'PAPER': 0.30,
      'TRAY': 0.55,
      'PLATE': 0.85,
      'MATE': 1.00
    };
    return thresholds[this.resonanceDetector.evolutionStage] || 1.00;
  }

  /**
   * FORCE EVOLUTION (for testing)
   */
  forceEvolution() {
    const oldStage = this.resonanceDetector.evolutionStage;
    this.resonanceDetector.coherenceScore = this.getNextThreshold();
    this.resonanceDetector.checkEvolution();
    console.log(`🔺 Forced evolution: ${oldStage} → ${this.resonanceDetector.evolutionStage}`);
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { TrinityOrchestrator };
}
