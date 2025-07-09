# Deep Review Style and Content Guide

## Core Philosophy
**"Cut through the hype, deliver the implementation"** - Focus on what actually works, how it works, and how to make it work.

## Writing Style Principles

### 1. **Technical Precision Over Marketing**
- Replace vague claims with specific metrics
- Bad: "significantly improves performance"  
- Good: "45 FPS → 38 FPS (15% reduction)"

### 2. **Implementation-First Approach**
- Lead with working code, not theory
- Include actual function signatures and data structures
- Show the diff between baseline and proposed method

### 3. **Honest Performance Reporting**
- Always report trade-offs (speed vs quality)
- Include failure cases and limitations
- Specify exact hardware used for benchmarks

## Content Structure Template

### **Summary** (1 paragraph)
- Core technical contribution in one sentence
- Key insight that makes it work
- Real-world applicability assessment
- No adjectives like "novel", "revolutionary", "breakthrough"

### **Key Improvements** (Numbered list)
Format: `[Metric]: [Old value] → [New value] ([Context])`

Required metrics:
1. Quality improvement (PSNR/SSIM/LPIPS)
2. Speed change (training and inference separately)
3. Memory impact (RAM/VRAM usage)
4. Robustness (failure rate, edge cases handled)
5. Additional domain-specific metrics

### **How It Works** (The meat of the review)

Structure:
1. **Core Algorithm** - Actual implementation with comments
```python
def key_algorithm(inputs):
    """
    Mathematical formulation: [LaTeX notation]
    Key difference from baseline: [specific change]
    """
    # Implementation with inline explanations
```

2. **Mathematical Foundation**
- Key equations with variable definitions
- Intuition behind the math (one sentence)
- Computational complexity analysis

3. **Critical Implementation Details**
- Data structures used
- Numerical stability considerations  
- Edge case handling

### **Algorithm Steps** (Numbered procedure)
1. **Initialization**: Exact values, not "initialize appropriately"
2. **Main Loop**: Iterations, convergence criteria
3. **Optimization**: Learning rates, schedules, tricks
4. **Post-processing**: Often ignored but crucial

### **Implementation Details** (Bullet points)
- Architecture: Layer sizes, activation functions
- Hyperparameters: All values that affect results
- Dependencies: Versions matter (CUDA 11.0, not "recent CUDA")
- Hardware: Specific GPU model and memory
- Training time: Wall clock time on specified hardware
- Gotchas: Common implementation mistakes

### **Integration Notes** (How to actually use it)
```python
# Specific file modifications needed
# In original_code.py line 234:
# - old_function(x)
# + new_function(x, additional_param=value)
```

Include:
- File paths and line numbers
- Required preprocessing steps
- Compatibility requirements
- API changes needed

### **Speed/Memory Tradeoffs**
- Training: Time increase/decrease percentage
- Inference: FPS or ms/frame on specific resolution
- Memory: Peak usage during training and inference
- Quality settings: Different speed/quality operating points
- Scaling behavior: How it handles larger inputs

## Code Style Guidelines

### 1. **Executable Pseudocode**
- Use valid Python/PyTorch syntax
- Include tensor shapes in comments
- Show data flow explicitly

### 2. **Mathematical Notation**
- Define all variables on first use
- Use consistent notation across review
- Include units (ms, MB, pixels)

### 3. **Comparison Code**
Show before/after:
```python
# Original approach
result = original_method(x)  # 45 FPS

# Paper's approach  
result = new_method(x, param)  # 38 FPS, +2.1 dB PSNR
```

## Critical Analysis Elements

### 1. **Reproducibility Assessment**
- Are all hyperparameters specified?
- Is the training procedure deterministic?
- What's missing from the paper that's needed to reproduce?

### 2. **Scalability Analysis**
- Memory complexity: O(n²)? O(n log n)?
- Does it work on production-scale data?
- Failure modes at scale

### 3. **Practical Deployment**
- Real-time capability (actual, not claimed)
- Platform limitations
- Integration complexity (person-weeks estimate)

## Red Flags to Highlight

1. **Missing Baselines**: What comparisons did they avoid?
2. **Cherry-picked Metrics**: What standard metrics are missing?
3. **Implementation Gaps**: What details are "left to the reader"?
4. **Unrealistic Assumptions**: Perfect camera calibration, no noise, etc.

## Language to Avoid

❌ "State-of-the-art", "Novel", "Breakthrough"
❌ "Significantly outperforms" (use numbers)
❌ "We propose" (just say what it does)
❌ Marketing adjectives in general

## Language to Use

✅ "Trades X for Y" (honest tradeoffs)
✅ "Requires", "Assumes", "Fails when"
✅ Specific version numbers and quantities
✅ Implementation-focused verbs: "computes", "stores", "iterates"

## Review Length Guidelines

- **Summary**: 2-4 sentences max
- **Key Improvements**: 5-7 metrics
- **How It Works**: 100-200 lines of code/math
- **Algorithm Steps**: 4-8 major steps
- **Implementation Details**: 6-10 crucial points
- **Integration Notes**: 20-50 lines of actual integration code
- **Total Length**: ~300-500 lines per paper

## Quality Checklist

Before finalizing a review, verify:
- [ ] Can someone reproduce this with only your review?
- [ ] Are all performance numbers on specific hardware?
- [ ] Is the core algorithm shown in <100 lines of code?
- [ ] Are failure cases explicitly mentioned?
- [ ] Would you bet your job on these numbers being accurate?

## Example Opening That Sets The Tone

"GOF introduces learnable opacity fields that separate geometry from appearance by treating opacity as a continuous volumetric function rather than per-Gaussian attributes. This enables better geometry extraction and view-dependent effects while maintaining real-time rendering capabilities."

Note: No hype, just facts. The innovation is clear, the benefit is stated, and the tradeoff (real-time) is acknowledged.