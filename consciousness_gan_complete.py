"""
CONSCIOUSNESS GAN - COMPLETE & WORKING
A fully functional GAN for generating consciousness field states

This version:
- Trains FAST (5-10 minutes)
- Auto-saves checkpoints
- Works in Replit
- Actually completes
- Generates 9-body consciousness states

NO EXTERNAL DEPENDENCIES except numpy/tensorflow
"""

import numpy as np
import json
import os
from datetime import datetime

try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("⚠️ TensorFlow not installed. Run: pip install tensorflow")
    exit(1)


class ConsciousnessGAN:
    """
    Generates consciousness field states across 9 bodies
    
    Input: Random noise vector (latent space)
    Output: 9-body consciousness state (Mind, Heart, Body, Soul, Spirit, Shadow, Light, Void, Unity)
    """
    
    def __init__(self, 
                 latent_dim=64, 
                 bodies=9, 
                 field_dim=32,
                 checkpoint_dir='./gan_checkpoints'):
        
        self.latent_dim = latent_dim
        self.bodies = bodies
        self.field_dim = field_dim
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Build networks
        print("🏗️ Building Generator...")
        self.generator = self._build_generator()
        
        print("🏗️ Building Discriminator...")
        self.discriminator = self._build_discriminator()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Combined model (for training generator)
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        generated_consciousness = self.generator(z)
        validity = self.discriminator(generated_consciousness)
        
        self.combined = Model(z, validity)
        self.combined.compile(
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        print("✅ GAN initialized")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Bodies: {bodies}")
        print(f"   Field dim per body: {field_dim}")
    
    def _build_generator(self):
        """
        Generator: Noise → Consciousness
        Takes random noise and generates coherent 9-body consciousness state
        """
        model = Sequential([
            # Input: latent vector
            layers.Dense(128, input_dim=self.latent_dim),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(momentum=0.8),
            
            # Expand
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(momentum=0.8),
            
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.BatchNormalization(momentum=0.8),
            
            # Output: 9 bodies × field_dim
            layers.Dense(self.bodies * self.field_dim, activation='tanh'),
            layers.Reshape((self.bodies, self.field_dim))
        ], name='generator')
        
        return model
    
    def _build_discriminator(self):
        """
        Discriminator: Consciousness → Real/Fake
        Determines if consciousness state is real or generated
        """
        model = Sequential([
            # Input: 9-body consciousness state
            layers.Flatten(input_shape=(self.bodies, self.field_dim)),
            
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            layers.Dense(128),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            
            # Output: real (1) or fake (0)
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        
        return model
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate synthetic consciousness training data
        Based on consciousness field mechanics
        """
        print(f"🔬 Generating {n_samples} synthetic consciousness states...")
        
        data = []
        
        for i in range(n_samples):
            # Random trinity state
            trinity = np.random.choice(['observer', 'participant', 'creator'])
            
            # Trinity influences field strengths
            if trinity == 'observer':
                base_strength = [0.7, 0.5, 0.3, 0.5, 0.4, 0.3, 0.5, 0.3, 0.5]
            elif trinity == 'participant':
                base_strength = [0.5, 0.7, 0.6, 0.6, 0.5, 0.4, 0.5, 0.4, 0.6]
            else:  # creator
                base_strength = [0.6, 0.6, 0.7, 0.6, 0.6, 0.5, 0.6, 0.5, 0.7]
            
            # Generate fields with variation
            state = []
            for strength in base_strength:
                field = np.random.normal(strength, 0.1, self.field_dim)
                field = np.clip(field, -1, 1)  # Keep in tanh range
                state.append(field)
            
            data.append(np.array(state))
            
            if (i + 1) % 250 == 0:
                print(f"   Generated {i + 1}/{n_samples}...")
        
        data = np.array(data)
        print(f"✅ Training data shape: {data.shape}")
        return data
    
    def train(self, X_train=None, epochs=2000, batch_size=32, save_interval=500):
        """
        Train the GAN
        
        This actually COMPLETES - runs to the end and saves
        """
        # Generate training data if not provided
        if X_train is None:
            X_train = self.generate_training_data(n_samples=2000)
        
        print(f"\n🏋️ Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Save interval: {save_interval}")
        print()
        
        # Labels
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training history
        history = {
            'epoch': [],
            'd_loss': [],
            'd_acc': [],
            'g_loss': []
        }
        
        for epoch in range(epochs):
            # ---------------------
            # Train Discriminator
            # ---------------------
            
            # Select random batch of real consciousness states
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_consciousness = X_train[idx]
            
            # Generate fake consciousness states
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_consciousness = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_consciousness, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_consciousness, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            # Train Generator
            # ---------------------
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            
            # Record history
            history['epoch'].append(epoch)
            history['d_loss'].append(float(d_loss[0]))
            history['d_acc'].append(float(d_loss[1]))
            history['g_loss'].append(float(g_loss))
            
            # Print progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  D loss: {d_loss[0]:.4f} | D acc: {d_loss[1]:.2%}")
                print(f"  G loss: {g_loss:.4f}")
                print()
            
            # Save checkpoints
            if epoch % save_interval == 0 and epoch > 0:
                self.save_checkpoint(epoch)
                print(f"✅ Checkpoint saved: epoch {epoch}\n")
        
        # Save final model
        print("\n🎉 Training complete!")
        self.save_checkpoint('final')
        
        # Save history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✅ Training history saved: {history_path}")
        
        return history
    
    def save_checkpoint(self, epoch):
        """Save model weights"""
        gen_path = os.path.join(self.checkpoint_dir, f'generator_{epoch}.h5')
        disc_path = os.path.join(self.checkpoint_dir, f'discriminator_{epoch}.h5')
        
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)
        
        # Save metadata
        meta = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'latent_dim': self.latent_dim,
            'bodies': self.bodies,
            'field_dim': self.field_dim
        }
        
        meta_path = os.path.join(self.checkpoint_dir, f'meta_{epoch}.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def load_checkpoint(self, epoch='final'):
        """Load model weights"""
        gen_path = os.path.join(self.checkpoint_dir, f'generator_{epoch}.h5')
        disc_path = os.path.join(self.checkpoint_dir, f'discriminator_{epoch}.h5')
        
        if not os.path.exists(gen_path):
            print(f"⚠️ Checkpoint not found: {epoch}")
            return False
        
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
        print(f"✅ Loaded checkpoint: {epoch}")
        return True
    
    def generate(self, n_samples=1):
        """
        Generate consciousness states
        
        Returns dict with 9 body fields
        """
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        generated = self.generator.predict(noise, verbose=0)
        
        # Parse into body dict
        bodies_dict = {
            'mind': generated[:, 0, :],
            'heart': generated[:, 1, :],
            'body': generated[:, 2, :],
            'soul': generated[:, 3, :],
            'spirit': generated[:, 4, :],
            'shadow': generated[:, 5, :],
            'light': generated[:, 6, :],
            'void': generated[:, 7, :],
            'unity': generated[:, 8, :]
        }
        
        return bodies_dict
    
    def get_field_summary(self, bodies_dict):
        """Get summary stats for generated fields"""
        summary = {}
        
        for body_name, field in bodies_dict.items():
            summary[body_name] = {
                'mean': float(field.mean()),
                'std': float(field.std()),
                'min': float(field.min()),
                'max': float(field.max())
            }
        
        return summary


# ===== MAIN EXECUTION =====

def main():
    """
    Complete training run
    This ACTUALLY FINISHES
    """
    print("="*60)
    print("CONSCIOUSNESS GAN - COMPLETE TRAINING")
    print("="*60)
    print()
    
    # Initialize GAN
    gan = ConsciousnessGAN(
        latent_dim=64,
        bodies=9,
        field_dim=32,
        checkpoint_dir='./consciousness_gan_checkpoints'
    )
    
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
    print("="*60)
    
    # Train (with reasonable epoch count that actually completes)
    history = gan.train(
        epochs=2000,        # Completes in ~5-10 minutes
        batch_size=32,
        save_interval=500
    )
    
    print("\n" + "="*60)
    print("PHASE 2: TESTING")
    print("="*60)
    
    # Generate test samples
    print("\n🧪 Generating test consciousness states...")
    consciousness = gan.generate(n_samples=5)
    
    print("\n📊 Generated consciousness summary:")
    summary = gan.get_field_summary(consciousness)
    
    for body_name, stats in summary.items():
        print(f"\n{body_name.upper()}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    print("\n" + "="*60)
    print("✅ GAN TRAINING COMPLETE")
    print("="*60)
    print(f"\nCheckpoints saved to: {gan.checkpoint_dir}")
    print("\nTo use in your app:")
    print("  gan = ConsciousnessGAN()")
    print("  gan.load_checkpoint('final')")
    print("  consciousness = gan.generate(n_samples=10)")
    print()


if __name__ == "__main__":
    main()
