# GUN ASSET FIX GUIDE
## Recovering Missing Texture from Mesh

**Problem**: You have a gun mesh but the texture disappeared

**Symptoms**:
- Gun model renders as gray/white mesh
- No material/texture applied
- Mesh geometry is intact but looks flat

---

## DIAGNOSIS

### Check What You Have:

```javascript
// In Unity/Unreal
GameObject gun = GameObject.Find("GunModel");

// Check mesh
Debug.Log("Has Mesh: " + (gun.GetComponent<MeshFilter>() != null));
Debug.Log("Mesh vertices: " + gun.GetComponent<MeshFilter>().mesh.vertexCount);

// Check material
MeshRenderer renderer = gun.GetComponent<MeshRenderer>();
Debug.Log("Has Material: " + (renderer.material != null));
Debug.Log("Material name: " + renderer.material.name);

// Check texture
if (renderer.material.mainTexture != null) {
    Debug.Log("Texture: " + renderer.material.mainTexture.name);
} else {
    Debug.Log("❌ TEXTURE MISSING");
}
```

---

## RECOVERY OPTIONS

### Option 1: Texture File Still Exists

**If the texture image file is still somewhere:**

```javascript
// Unity
using UnityEngine;

public class GunTextureFix : MonoBehaviour {
    void Start() {
        // Find the gun
        GameObject gun = GameObject.Find("GunModel");
        MeshRenderer renderer = gun.GetComponent<MeshRenderer>();
        
        // Load texture from Resources
        Texture2D gunTexture = Resources.Load<Texture2D>("Textures/GunTexture");
        
        // Apply to material
        if (gunTexture != null) {
            renderer.material.mainTexture = gunTexture;
            Debug.Log("✅ Texture restored");
        } else {
            Debug.Log("❌ Texture file not found");
        }
    }
}
```

**File locations to check:**
- `/Assets/Resources/Textures/`
- `/Assets/Models/Gun/`
- `/Assets/Materials/`
- Your project backup folder
- Original gun asset package

---

### Option 2: Regenerate Texture from UV Map

**If you have the mesh with UVs intact:**

```python
# Python - Generate basic gun texture
from PIL import Image, ImageDraw

def generate_gun_texture(size=1024):
    # Create base texture
    img = Image.new('RGB', (size, size), color=(50, 50, 50))
    draw = ImageDraw.Draw(img)
    
    # Add metallic highlights
    for i in range(0, size, 20):
        draw.line([(i, 0), (i, size)], fill=(70, 70, 70), width=2)
    
    # Add grip texture area
    grip_start = size // 2
    for i in range(grip_start, size, 10):
        draw.ellipse([(i-5, size-100), (i+5, size-80)], fill=(30, 30, 30))
    
    # Add barrel highlights
    draw.rectangle([(0, 0), (size, size//3)], outline=(80, 80, 80), width=3)
    
    # Save
    img.save('gun_texture_regenerated.png')
    print("✅ Generated gun_texture_regenerated.png")

generate_gun_texture()
```

---

### Option 3: Use Procedural Material

**If texture is completely lost, use shader to create metal look:**

```csharp
// Unity - Procedural metal shader
Shader "Custom/GunMetal" {
    Properties {
        _Color ("Base Color", Color) = (0.3, 0.3, 0.3, 1)
        _Metallic ("Metallic", Range(0,1)) = 0.9
        _Smoothness ("Smoothness", Range(0,1)) = 0.8
    }
    
    SubShader {
        Tags { "RenderType"="Opaque" }
        
        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        
        struct Input {
            float3 worldPos;
            float3 worldNormal;
        };
        
        fixed4 _Color;
        float _Metallic;
        float _Smoothness;
        
        void surf (Input IN, inout SurfaceOutputStandard o) {
            // Base color
            o.Albedo = _Color.rgb;
            
            // Metallic properties
            o.Metallic = _Metallic;
            o.Smoothness = _Smoothness;
            
            // Add wear pattern based on world position
            float wear = frac(IN.worldPos.y * 10.0);
            o.Smoothness *= (0.7 + wear * 0.3);
        }
        ENDCG
    }
}
```

Apply this shader:
```csharp
GameObject gun = GameObject.Find("GunModel");
Material metalMaterial = new Material(Shader.Find("Custom/GunMetal"));
gun.GetComponent<MeshRenderer>().material = metalMaterial;
```

---

### Option 4: Extract Texture from Similar Asset

**If you have another gun model with texture:**

```csharp
// Unity - Copy material from another gun
GameObject workingGun = GameObject.Find("WorkingGunModel");
GameObject brokenGun = GameObject.Find("BrokenGunModel");

Material goodMaterial = workingGun.GetComponent<MeshRenderer>().sharedMaterial;
brokenGun.GetComponent<MeshRenderer>().material = goodMaterial;

Debug.Log("✅ Material copied");
```

---

### Option 5: Bake New Texture from Substance

**If you have Substance Painter/Designer:**

1. Export mesh as FBX
2. Import to Substance Painter
3. Use "Metal Base" smart material
4. Add:
   - Scratches layer
   - Dirt layer  
   - Edge wear
5. Export as 4K texture set:
   - BaseColor
   - Normal
   - Metallic
   - Roughness
6. Re-import to Unity/Unreal

---

## QUICK FIX (WORKS 90% OF TIME)

**Most common issue: Material reference broke**

```csharp
// Unity Quick Fix Script
using UnityEngine;
using UnityEditor;

public class GunMaterialFix : MonoBehaviour {
    [MenuItem("Tools/Fix Gun Material")]
    static void FixGunMaterial() {
        // Find all renderers in scene
        MeshRenderer[] renderers = FindObjectsOfType<MeshRenderer>();
        
        foreach (var renderer in renderers) {
            if (renderer.gameObject.name.Contains("Gun")) {
                // Check if material is missing/broken
                if (renderer.sharedMaterial == null || 
                    renderer.sharedMaterial.name.Contains("Default")) {
                    
                    // Try to find gun material in project
                    Material gunMat = AssetDatabase.LoadAssetAtPath<Material>(
                        "Assets/Materials/GunMaterial.mat"
                    );
                    
                    if (gunMat != null) {
                        renderer.material = gunMat;
                        Debug.Log($"✅ Fixed: {renderer.gameObject.name}");
                    }
                }
            }
        }
    }
}
```

Run: **Tools → Fix Gun Material** in Unity menu

---

## PREVENTION

**To avoid this in future:**

### 1. Use Asset Bundles
```csharp
// Pack gun as asset bundle
[MenuItem("Assets/Build Gun Bundle")]
static void BuildGunBundle() {
    BuildPipeline.BuildAssetBundle(
        Selection.activeObject,
        Selection.GetFiltered(typeof(Object), SelectionMode.DeepAssets),
        "Assets/Bundles/gun_complete.unity3d",
        BuildAssetBundleOptions.CollectDependencies,
        BuildTarget.StandaloneWindows
    );
}
```

### 2. Create Prefab with Material
```csharp
// Save as complete prefab
GameObject gun = GameObject.Find("GunModel");
PrefabUtility.SaveAsPrefabAsset(gun, "Assets/Prefabs/GunComplete.prefab");
```

### 3. Version Control Materials
```bash
# In .gitignore, DON'T ignore:
# /Assets/Materials/**
# /Assets/Textures/**

# DO track these files
```

---

## EMERGENCY TEXTURE GENERATION

**If completely stuck, use this AI-generated placeholder:**

```python
# Generate quick gun texture with noise
import numpy as np
from PIL import Image

def emergency_gun_texture(size=512):
    # Create noise-based metal texture
    noise = np.random.randint(40, 70, (size, size, 3), dtype=np.uint8)
    
    # Add vertical scratches
    for x in range(0, size, np.random.randint(30, 50)):
        noise[x:x+2, :] = np.random.randint(80, 100, (2, size, 3))
    
    # Darken bottom (grip area)
    noise[size//2:, :] = noise[size//2:, :] * 0.6
    
    # Convert to image
    img = Image.fromarray(noise.astype('uint8'), 'RGB')
    img.save('gun_emergency_texture.png')
    
    print("✅ Emergency texture generated")
    return img

emergency_gun_texture()
```

---

## DEBUGGING CHECKLIST

When gun mesh loads but has no texture:

- [ ] Check if material exists
- [ ] Check if material has texture slot filled
- [ ] Check if texture file exists in project
- [ ] Check if mesh has UV coordinates
- [ ] Check if shader compiles without errors
- [ ] Check if material is assigned to correct submesh
- [ ] Check import settings on texture (Read/Write enabled?)
- [ ] Check if texture is in Resources folder for runtime loading
- [ ] Check git history for when texture was last present
- [ ] Check if texture path changed after refactoring

---

## CONTACT ME WITH:

If none of this works, I need:

1. **Screenshot** of the broken gun (so I see what mesh you have)
2. **Material inspector** screenshot (what material is assigned)
3. **Console errors** (any shader/texture errors)
4. **When it broke** (did something else change?)
5. **Project structure** (where are Materials/Textures folders)

Then I can give you exact fix for your specific case.

---

**Most likely fix**: Your texture path broke during refactoring.

Try this FIRST:
1. Search project for "gun" or "weapon" texture files
2. If found, reassign to material manually
3. Save material as new asset
4. Reassign to gun mesh

**If texture truly gone**:
Use Option 3 (Procedural Material) - creates metallic gun look without needing texture file.
