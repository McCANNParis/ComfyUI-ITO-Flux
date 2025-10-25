# ComfyUI-ITO-Flux Examples

This document provides practical examples and workflows for using the ITO Flux Sampler.

## Basic Workflow

### Step-by-Step Setup

1. **Load Flux Dev Model**
   - Add "Load Checkpoint" node
   - Select your Flux Dev checkpoint

2. **Setup Text Encoding**
   - Add "CLIP Text Encode (Prompt)" for positive prompt
   - Add "CLIP Text Encode (Prompt)" for negative prompt
   - Connect both to the model's CLIP output

3. **Create Empty Latent**
   - Add "Empty Latent Image" node
   - Set your desired resolution (e.g., 1024x1024)
   - Flux Dev works best with resolutions divisible by 64

4. **Add ITO Flux Sampler**
   - Add "ITO Flux Sampler" node
   - Connect:
     - Model → from Load Checkpoint
     - Positive → from positive CLIP Text Encode
     - Negative → from negative CLIP Text Encode
     - Latent Image → from Empty Latent Image

5. **Decode and Save**
   - Add "VAE Decode" node
   - Add "Save Image" or "Preview Image" node
   - Connect latent output from ITO Flux Sampler

### Basic Settings

```
Prompt: "a beautiful landscape with mountains and a lake, golden hour, highly detailed"
Negative: "blurry, low quality, distorted"

ITO Flux Sampler:
  - steps: 20
  - cfg: 3.5 (base, will be dynamically adjusted)
  - sampler_name: euler
  - scheduler: simple
  - guidance_min: 1.0
  - guidance_max: 3.5
  - divergence_type: l2
  - schedule_type: sigmoid
  - sensitivity: 1.0
  - warmup_steps: 0
  - smoothing_window: 3
  - debug_mode: false
```

## Example Use Cases

### Example 1: Portrait Photography

**Goal**: Sharp details in skin, hair, and eyes with natural colors

**Settings**:
```
Prompt: "portrait of a woman with detailed eyes, professional photography,
         natural lighting, sharp focus"
Negative: "blurry, artificial, oversaturated"

ITO Settings:
  - guidance_min: 1.0
  - guidance_max: 3.5
  - divergence_type: l2
  - schedule_type: sigmoid
  - sensitivity: 1.0
  - warmup_steps: 0
  - smoothing_window: 3
```

**Why these settings?**
- L2 divergence works well for general portraits
- Sigmoid provides smooth, stable transitions
- Default sensitivity (1.0) prevents over-processing skin tones

### Example 2: Detailed Fabric/Materials

**Goal**: Maximum texture detail in clothing, fabrics, or materials

**Settings**:
```
Prompt: "close-up of intricate silk fabric with detailed weave pattern,
         macro photography"
Negative: "smooth, flat, blurry"

ITO Settings:
  - guidance_min: 1.0
  - guidance_max: 4.0
  - divergence_type: frequency
  - schedule_type: sigmoid
  - sensitivity: 1.3
  - warmup_steps: 2
  - smoothing_window: 5
```

**Why these settings?**
- Frequency divergence emphasizes high-frequency details (textures)
- Higher guidance_max (4.0) for stronger detail enhancement
- Increased sensitivity (1.3) for more aggressive guidance adjustment
- Warmup steps help establish structure before fine details

### Example 3: Artistic/Stylized Images

**Goal**: Creative, artistic results with controlled stylization

**Settings**:
```
Prompt: "surreal landscape, painterly style, vibrant colors, artistic"
Negative: "photorealistic, plain, boring"

ITO Settings:
  - guidance_min: 0.5
  - guidance_max: 3.0
  - divergence_type: cosine
  - schedule_type: polynomial
  - sensitivity: 1.5
  - warmup_steps: 0
  - smoothing_window: 3
```

**Why these settings?**
- Lower guidance_min (0.5) allows more creative freedom
- Cosine divergence for directional/stylistic differences
- Polynomial schedule for non-linear creative response
- Higher sensitivity for more dynamic results

### Example 4: Architectural/Technical

**Goal**: Clean lines, precise details, minimal artifacts

**Settings**:
```
Prompt: "modern architecture, glass building, geometric, clean lines,
         architectural photography"
Negative: "distorted, warped, messy"

ITO Settings:
  - guidance_min: 1.5
  - guidance_max: 3.5
  - divergence_type: l2
  - schedule_type: linear
  - sensitivity: 0.8
  - warmup_steps: 3
  - smoothing_window: 5
```

**Why these settings?**
- Higher guidance_min (1.5) maintains structural integrity
- Linear schedule for predictable, consistent behavior
- Lower sensitivity (0.8) for more conservative adjustments
- Warmup steps to establish geometry first

### Example 5: Long Sampling (30+ steps)

**Goal**: High quality with many sampling steps

**Settings**:
```
Prompt: "your detailed prompt here"
Negative: "low quality, blurry"

ITO Settings:
  - steps: 40
  - guidance_min: 1.0
  - guidance_max: 3.0
  - divergence_type: l2
  - schedule_type: sigmoid
  - sensitivity: 0.7
  - warmup_steps: 5
  - smoothing_window: 7
```

**Why these settings?**
- Lower guidance_max (3.0) for many steps
- Lower sensitivity (0.7) prevents over-guidance
- More warmup steps for gradual refinement
- Larger smoothing window for stability

## Comparison Workflow

### Side-by-Side: ITO vs Standard CFG

To compare ITO against standard CFG:

1. **Create two parallel paths** in your workflow
2. **Path 1**: Use standard "KSampler" with fixed CFG (e.g., 3.5)
3. **Path 2**: Use "ITO Flux Sampler" with adaptive guidance
4. **Use identical settings**:
   - Same seed
   - Same steps
   - Same sampler/scheduler
   - Same prompts
5. **Save both outputs** side-by-side

**Recommended comparison settings**:
```
Seed: 42 (or any fixed seed)
Steps: 20
Sampler: euler
Scheduler: simple

Standard KSampler:
  - CFG: 3.5 (fixed)

ITO Flux Sampler:
  - guidance_min: 1.0
  - guidance_max: 3.5
  - divergence_type: l2
  - schedule_type: sigmoid
  - sensitivity: 1.0
```

Look for differences in:
- Texture sharpness
- Color saturation
- Edge quality
- Overall detail

## Debug Mode Examples

### Enabling Debug Visualization

Use the **ITO Flux Sampler (Debug)** node for full visualization:

**Features**:
1. **Console Output**: Real-time metrics printed to console
2. **Debug Plot**: Visual graph of divergence and guidance over time (as IMAGE output)
3. **Summary Statistics**: Printed at the end of sampling

**Example debug workflow**:
```
ITO Flux Sampler (Debug)
  ↓ (latent output)        ↓ (debug_plot output)
  ↓                         ↓
VAE Decode              Preview/Save Image
  ↓                    (metrics visualization)
Save Image
```

**Alternative**: Use regular **ITO Flux Sampler** with `debug_mode=true` for console-only output (no visual plot).

**What to look for in debug plots**:
- **Divergence curve**: Should show variation across steps
  - High early → model predictions differ significantly
  - Low later → predictions converging
- **Guidance curve**: Should adapt based on divergence
  - Should not be flat (that means no adaptation)
  - Should vary smoothly (if using sigmoid)

### Interpreting Debug Output

**Console output example**:
```
=== ITO Debug Info - Step 0 ===
Raw Divergence: 0.234567
Normalized Divergence: 0.523456
Guidance Scale: 2.456
EMA Divergence: 0.234567
Min Seen: 0.234567
Max Seen: 0.234567
========================================
```

**What this tells you**:
- **Raw Divergence**: Actual measured difference
- **Normalized Divergence**: Scaled to [0,1] for schedule mapping
- **Guidance Scale**: The actual guidance being applied this step
- **EMA Divergence**: Smoothed divergence value

## Schedule Visualizer Examples

Use the **ITO Guidance Schedule Visualizer** to preview schedules:

### Example 1: Conservative Schedule
```
guidance_min: 1.5
guidance_max: 3.0
schedule_type: linear
sensitivity: 0.8
```
Result: Gentle slope, limited range

### Example 2: Aggressive Schedule
```
guidance_min: 0.5
guidance_max: 5.0
schedule_type: exponential
sensitivity: 2.0
```
Result: Steep curve, wide range, dramatic response

### Example 3: Balanced Schedule
```
guidance_min: 1.0
guidance_max: 3.5
schedule_type: sigmoid
sensitivity: 1.0
```
Result: S-curve, smooth transitions, stable

## Tips and Tricks

### Finding Optimal Settings

1. **Start with defaults**: Begin with recommended settings
2. **Enable debug mode**: Run once with debug to see behavior
3. **Adjust one parameter at a time**: Don't change everything at once
4. **Use fixed seed**: Keep seed constant while testing
5. **Compare with standard**: Run standard CFG as baseline

### Common Adjustments

**If results are too saturated**:
- ↓ guidance_max
- ↓ sensitivity
- Try polynomial or linear schedule

**If results lack detail**:
- ↑ guidance_max
- ↑ sensitivity
- Try frequency divergence

**If results are unstable/flickering** (when animating):
- ↑ smoothing_window
- ↑ warmup_steps
- ↓ sensitivity

**If no visible difference from standard CFG**:
- ↑ sensitivity
- ↑ divergence range (guidance_max - guidance_min)
- Try different divergence_type

### Advanced Techniques

#### Prompt-Specific Tuning

Different prompts may benefit from different settings:

- **Photorealistic**: l2, sigmoid, sensitivity 1.0
- **Artistic**: cosine, polynomial, sensitivity 1.5
- **Technical**: l2, linear, sensitivity 0.8
- **Texture-heavy**: frequency, sigmoid, sensitivity 1.3

#### Resolution Considerations

Higher resolutions may benefit from:
- Slightly lower sensitivity
- More smoothing
- More warmup steps

#### Batch Processing

When processing multiple images:
1. Use consistent seed range
2. Keep ITO settings constant
3. Monitor console for any anomalies

## Troubleshooting Specific Issues

### Issue: Black or corrupted images

**Solution**:
- Verify model is Flux Dev (not SDXL or SD1.5)
- Check that conditioning is properly connected
- Try reducing guidance_max temporarily
- Check ComfyUI console for errors

### Issue: No visible improvement

**Solution**:
- Increase sensitivity from 1.0 to 1.5
- Widen guidance range (e.g., 0.5 to 4.5)
- Try frequency divergence
- Enable debug mode to verify adaptation is happening

### Issue: Over-processed look

**Solution**:
- Reduce guidance_max (e.g., from 3.5 to 2.5)
- Reduce sensitivity (e.g., from 1.0 to 0.7)
- Add warmup_steps (e.g., 5)
- Try linear schedule instead of sigmoid

### Issue: Inconsistent results with same seed

**Solution**:
- This shouldn't happen; ITO is deterministic
- Check that all other nodes use same seed
- Verify no random elements in workflow
- Report as bug if persists

## Performance Notes

### Speed Comparison

- ITO overhead: ~5-10% slower than standard sampling
- Most time is still in model inference
- Debug mode adds minimal overhead

### Memory Usage

- Minimal additional VRAM required
- Safe to use with large images
- No significant memory leaks

### Batch Size

- Works with batch size > 1
- Each image in batch gets adaptive guidance
- Divergence calculated per-image

---

**Need more help?** Check the main README.md or open an issue on GitHub!
