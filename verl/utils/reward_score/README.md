# Reward Score Module

A production-ready, multi-domain reward scoring system for reinforcement learning training of reasoning models. This module provides standardized evaluation across math, logic, table reasoning, code, and document QA tasks with support for multiple model formats.

## Table of Contents

- [Overview](#overview)
- [Supported Datasets](#supported-datasets)
- [Architecture & Design](#architecture--design)
- [Usage Examples](#usage-examples)
- [Adding New Datasets](#adding-new-datasets)
- [Adding New Formats](#adding-new-formats)
- [Testing](#testing)
- [Advanced Topics](#advanced-topics)

---

## Overview

The reward_score module implements a **cascade reward system** for evaluating model outputs across multiple reasoning domains. Key features:

- **7 Domain Coverage**: Math, logic, table reasoning, code, tool learning, document QA, and instruction following
- **Dual Format Support**: Compatible with XML (Qwen/Llama) and GPT-OSS formats
- **Cascade Rewards**: Progressive evaluation (thinking → format → correctness)
- **Extensible Architecture**: Plugin-based format handlers and modular scorers

### Why Cascade Rewards?

The 3-stage cascade system separates concerns:
1. **reward_think**: Validates reasoning section formatting (0.0 or 1.0)
2. **reward_fmt**: Validates answer extraction and format (0.0 or 1.0)
3. **score**: Evaluates correctness against ground truth (0.0 to 1.0)

This design enables:
- **Clear training signals**: Models learn proper formatting before correctness
- **Debugging transparency**: Identify exactly where models fail
- **Flexible optimization**: Weight different components during training
- **Cascading failures**: Format errors don't contaminate correctness metrics

---

## Supported Datasets

### Math Domain (`math.py`)

**Scorer**: Mathematical reasoning with \\boxed{} format
**Format**: `\boxed{answer}` with numerical/symbolic verification

| Dataset | Description | Notes |
|---------|-------------|-------|
| `openai/gsm8k` | Grade school math problems | 8K training samples |
| `lighteval/MATH` | Competition-level mathematics | MATH benchmark |
| `HuggingFaceH4/MATH-500` | MATH subset | 500 problems |
| `numina_*` | Numina math series | Various difficulty levels |
| `math_dapo` | DAPO math dataset | Math reasoning |

### Tool Learning (`toolrl.py`)

**Scorer**: Tool invocation and reasoning
**Format**: JSON tool calls with validation

| Dataset | Description | HuggingFace Link |
|---------|-------------|------------------|
| **sungyub/toolrl-verl** | Tool learning for RL | [🔗 HF Hub](https://huggingface.co/datasets/sungyub/toolrl-verl) |
| `rlla` | Real-world tool learning | - |
| `toolace` | Tool ACE benchmark | - |
| `hammer` | HAMMER tool dataset | - |
| `rlla_gpt` | GPT-OSS format tool learning | - |

**Environment Variables (VERL Compatibility):**

The ToolRL scorer supports advanced reward shaping through environment variables:

| Variable | Effect | Default |
|----------|--------|---------|
| `WITHLENGTH=1` | Auto-enable length reward component | Disabled |
| `CORRECTMAX1=1` | Set correctness max to 1 (instead of 3) | 3 |
| `SCHEDULEREWARD=1` | Apply step-based reward scaling | Disabled |
| `SCHEDULELENGTH=1` | Dynamic length threshold scaling (384-640 words) | Disabled |
| `REFINEDREWARD=1` | Strict exact matching (no partial credit) | Disabled |
| `COARSEREWARD=1` | Binary match/no-match scoring | Disabled |
| `INTERMEDIATEREWARD=1` | Simplified intermediate scoring | Disabled |

**Example:**
```bash
export WITHLENGTH=1
export CORRECTMAX1=1
python train_verl.py
```

### Code Execution (`sandbox_fusion/`)

**Scorer**: Multi-language code execution with test case validation
**Format**: Markdown code blocks with automatic language detection
**Requires**: SandboxFusion service

**Supported Datasets (17 total):**

| Dataset | Description | Difficulty | Notes |
|---------|-------------|------------|-------|
| `codecontests` | Google CodeContests | Mixed | Original benchmark |
| `apps` | APPS benchmark | Mixed | Original benchmark |
| `codeforces` | Codeforces problems | Mixed | Original benchmark |
| `taco` | TACO benchmark | Mixed | Original benchmark |
| `code-contests-plus` | Enhanced CodeContests | Mixed | Extended version |
| `kodcode-leetcode` | KodCode LeetCode | Mixed | LeetCode problems |
| `oss` | AceCode dataset | Mixed | Open source |
| `rstar-coder` | R* Coder | Mixed | Advanced reasoning |
| `train-code-leetcode-Easy` | LeetCode Easy | Easy | Training split |
| `train-code-leetcode-Medium` | LeetCode Medium | Medium | Training split |
| `train-code-leetcode-Hard` | LeetCode Hard | Hard | Training split |
| `test-code-leetcode-Medium` | LeetCode Medium Test | Medium | Test split |
| `train-code-taco-easy` | TACO Easy | Easy | Training split |
| `train-code-taco-medium` | TACO Medium | Medium | Training split |
| `train-code-taco-hard` | TACO Hard | Hard | Training split |
| `train-code-taco-medium_hard` | TACO Medium-Hard | Medium-Hard | Training split |
| `train-code-taco-very_hard` | TACO Very Hard | Very Hard | Training split |
| `train-code-taco-unknown_difficulty` | TACO Unknown | Unknown | Training split |

**Supported Languages (30+):**
- **Primary**: python, cpp, java, go, rust, javascript (nodejs), typescript
- **Additional**: kotlin, swift, scala, julia, php, perl, ruby, lua, R, bash
- **Testing Frameworks**: pytest, junit, jest, go_test
- **Specialized**: csharp, sql, cuda, verilog, lean, racket, D_ut, python_gpu

**Language Auto-Detection:**
- Automatically detects language from markdown code blocks (e.g., ````cpp`, ````java`)
- Language mapping: `py3`/`py2`/`python3` → `python`, `c++`/`c++17` → `cpp`
- Fallback to Python for unsupported languages

### Code Verification (`codev.py`)

**Scorer**: Verilog code verification with simulation
**Format**: Code blocks with test validation
**Requires**: SandboxFusion service

| Dataset | Description | HuggingFace Link |
|---------|-------------|------------------|
| **sungyub/codev-r1-verl** | Verilog verification | [🔗 HF Hub](https://huggingface.co/datasets/sungyub/codev-r1-verl) |

### Instruction Following (`ifeval/`)

**Scorer**: Multi-constraint instruction adherence
**Format**: Free text with constraint checking

| Dataset | Description | HuggingFace Link |
|---------|-------------|------------------|
| **sungyub/ifbench-verl** | IFBench for RL training | [🔗 HF Hub](https://huggingface.co/datasets/sungyub/ifbench-verl) |
| `allenai/IF_multi_constraints_upto5` | IFEval benchmark | - |

### Table Reasoning (`table_boxed.py`, `tqa.py`, `tfv.py`, `ff_tqa.py`)

**Scorers**: Multiple table QA formats

| Dataset | Scorer | Format | Description |
|---------|--------|--------|-------------|
| `hitab` | `table_boxed.py` | \\boxed{} | Hierarchical table QA |
| `finqa` | `table_boxed.py` | \\boxed{} | Financial tables |
| `WTQ` | `tqa.py` | JSON list | WikiTableQuestions |
| `TabFact` | `tfv.py` | entailed/refuted | Fact verification |
| `FeTaQA` | `ff_tqa.py` | Free text | Free-form table QA |

### Document QA (`docqa.py`, `docmath.py`, `long.py`)

**Scorers**: Long-context document reasoning

| Dataset | Scorer | Format | Metric |
|---------|--------|--------|--------|
| `multihoprag` | `docqa.py` | "the answer is X" | EM/F1 |
| `musique` | `docqa.py` | "the answer is X" | EM/F1 |
| `docmath` | `docmath.py` | "the answer is X" | Numeric tolerance |
| `long_toc_choices` | `long.py` | "The correct answer is (A)" | Multiple choice |

### Logic Reasoning (`logic.py`)

**Scorer**: Structured logic problems
**Format**: JSON structures (lists, dicts, arrays)

| Dataset | Format | Description |
|---------|--------|-------------|
| `ordering_puzzle` | List | Sequence ordering |
| `zebra_puzzle` | Dict (header+rows) | Constraint satisfaction |
| `graph_logical` | String | Graph reasoning |
| `arcagi1`, `arcagi2` | 2D array | Visual pattern recognition |
| `barc` | 2D array | Abstract reasoning |

---

## Architecture & Design

### 1. Cascade Reward System

The module implements a 3-stage evaluation pipeline with **cascade failures**:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Stage 1:       │     │  Stage 2:       │     │  Stage 3:       │
│  reward_think   │ ──> │  reward_fmt     │ ──> │  score          │
│  (0.0 or 1.0)   │     │  (0.0 or 1.0)   │     │  (0.0 to 1.0)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      ↓                        ↓                        ↓
 Validates              Validates                Evaluates
 thinking section       answer format            correctness
```

**Example from `math.py`:**

```python
def compute_score(model_output: str, ground_truth: str, format_type: str = "auto"):
    reward_think = 0.0
    reward_fmt = 0.0
    reward_correct = 0.0

    # Stage 1: Validate thinking section
    pred, pass_think_parsed = parse_think(model_output, format_type=format_type)
    if pass_think_parsed:
        reward_think = 1.0

        # Stage 2: Validate answer format (only if stage 1 passed)
        pred_parsed, pred_type = parse_answer(pred, format_type=format_type)
        gt_parsed, gt_type = parse_answer(ground_truth, format_type="auto")

        if pred_type == gt_type:  # Format types must match
            reward_fmt = 1.0

            # Stage 3: Verify correctness (only if stage 2 passed)
            if reward_fmt == 1.0:
                is_correct = verify(parse(f"\\boxed{{{pred_parsed}}}"),
                                   parse(f"\\boxed{{{gt_parsed}}}"))
                reward_correct = 1.0 if is_correct else 0.0

    return {
        "score": reward_correct,      # Final correctness score
        "reward_think": reward_think,  # Thinking format reward
        "reward_fmt": reward_fmt,      # Answer format reward
    }
```

**Cascade Failure Example:**
```python
# Missing thinking section → all rewards are 0.0
model_output = "\\boxed{42}"
result = compute_score(model_output, "42")
# Result: {"score": 0.0, "reward_think": 0.0, "reward_fmt": 0.0}
# Note: reward_fmt and score are 0.0 due to cascade failure, not evaluation
```

### 2. Format Handler System

The module uses a **plugin architecture** for handling different response formats:

```
format_handlers/
├── base.py                 # BaseFormatHandler abstract class
├── xml_format.py           # XML format (Qwen/Llama)
└── gpt_oss_format.py       # GPT-OSS format (GPT-OSS-120B)
```

#### XML Format (Default)

**Used by**: Qwen, Llama, most open-source models
**Structure**:
```xml
<think>
Step-by-step reasoning goes here...
</think>
\boxed{42}
```

**Qwen3 Compatibility**: Answer tags are optional (Qwen3 tokenizer doesn't include `<answer>` tags):
```xml
<think>Reasoning...</think>
["A", "B", "C"]  <!-- Plain text answer, no tags -->
```

#### GPT-OSS Format

**Used by**: GPT-OSS-120B model
**Structure**:
```
<|start|>assistant<|channel|>analysis<|message|>
Step-by-step reasoning goes here...
<|end|>

<|start|>assistant<|channel|>final<|message|>
\boxed{42}
<|return|>
```

#### Auto-Detection

The system automatically detects the format:

```python
def detect_format(text: str) -> str:
    # GPT-OSS has higher priority (more specific markers)
    if '<|start|>assistant' in text and '<|channel|>' in text:
        return "gpt_oss"
    # Default to XML format
    return "xml"
```

**Usage**:
```python
# Auto-detect (recommended)
result = compute_score(model_output, ground_truth)

# Explicit format
result = compute_score(model_output, ground_truth, format_type="gpt_oss")
```

### 3. Code Organization

```
reward_score/
├── __init__.py                    # Main routing: default_compute_score()
├── utils.py                       # Shared utilities (parsing, normalization)
│
├── format_handlers/               # Format plugin system
│   ├── __init__.py               # Auto-detection & convenience functions
│   ├── base.py                   # BaseFormatHandler abstract class
│   ├── xml_format.py             # XML format handler
│   └── gpt_oss_format.py         # GPT-OSS format handler
│
├── math.py                        # Math domain scorer
├── logic.py                       # Logic reasoning scorer
├── table_boxed.py                 # Table QA with \boxed{} format
├── tqa.py                         # Table QA with JSON list format
├── tfv.py                         # Table fact verification
├── ff_tqa.py                      # Free-form table QA
├── docqa.py                       # Document QA (EM/F1)
├── docmath.py                     # Document math problems
├── long.py                        # Long-context multiple choice
├── toolrl.py                      # Tool learning scorer
├── codev.py                       # Code verification scorer
│
├── sandbox_fusion/                # Code execution via SandboxFusion
│   ├── __init__.py
│   └── utils.py
│
└── ifeval/                        # Instruction following
    ├── __init__.py
    ├── instructions.py
    ├── instructions_registry.py
    └── instructions_util.py
```

### 4. Routing Logic

The `default_compute_score()` function in `__init__.py` routes requests to appropriate scorers:

```python
def default_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Main entry point for reward scoring.

    Args:
        data_source: Dataset identifier (e.g., "openai/gsm8k", "WTQ")
        solution_str: Model's generated response
        ground_truth: Expected answer
        format_type: "auto", "xml", or "gpt_oss"

    Returns:
        Dict with at minimum: {score, reward_think, reward_fmt}
    """
    # Math domain
    if "gsm8k" in data_source or "math" in data_source.lower():
        from . import math
        return math.compute_score(solution_str, ground_truth, format_type=format_type)

    # Logic domain
    elif "puzzle" in data_source or "arcagi" in data_source:
        from . import logic
        return logic.compute_score(solution_str, ground_truth, data_source, format_type=format_type)

    # Table domain
    elif data_source in ["hitab", "finqa"]:
        from . import table_boxed
        return table_boxed.compute_score(solution_str, ground_truth, data_source, format_type=format_type)

    # ... more routing logic ...
```

### 5. Dataset Routing Patterns

The reward scorer supports two types of routing strategies to match dataset identifiers:

**1. Exact Match Routing:**
Datasets must exactly match the specified string in the routing logic.

```python
# Examples from __init__.py
if data_source in ["codecontests", "apps", "codeforces", "taco"]:
    # Routes to sandbox_fusion scorer

if data_source in ["hitab", "finqa"]:
    # Routes to table_boxed scorer
```

**2. Pattern-Based Routing:**
Datasets are matched using substring patterns, allowing flexible grouping.

```python
# Examples from __init__.py
if "gsm8k" in data_source or "math" in data_source.lower():
    # Routes to math scorer
    # Matches: "openai/gsm8k", "math_dapo", "Big-Math-RL-Verified", etc.

if "puzzle" in data_source or "arcagi" in data_source:
    # Routes to logic scorer
    # Matches: "ordering_puzzle", "zebra_puzzle", "arcagi1", "arcagi2", etc.

if "docmath" in data_source:
    # Routes to docmath scorer
    # Matches: "docmath", "docmath_hard", etc.
```

**Pattern Examples by Domain:**

| Pattern | Matches | Scorer |
|---------|---------|--------|
| `"gsm8k" in data_source` | `openai/gsm8k`, `gsm8k_hard` | `math.py` |
| `"math" in data_source.lower()` | `MATH`, `math_dapo`, `Big-Math-RL-Verified` | `math.py` |
| `"puzzle" in data_source` | `ordering_puzzle`, `zebra_puzzle` | `logic.py` |
| `"arcagi" in data_source` | `arcagi1`, `arcagi2` | `logic.py` |
| `"barc" in data_source` | `barc`, `mini_barc` | `logic.py` |
| `"docmath" in data_source` | `docmath`, `docmath_hard` | `docmath.py` |
| `"long_toc_choices" in data_source` | `long_toc_choices`, `long_toc_choices_hard` | `long.py` |

**Adding New Datasets:**

When adding a new dataset, consider which routing strategy fits best:

- **Use Exact Match** when:
  - Dataset name is unique and unlikely to have variants
  - You want precise control over routing
  - Example: `"my_specific_dataset"`

- **Use Pattern Match** when:
  - Dataset has multiple variants or versions
  - You want to group related datasets
  - Dataset name follows a naming convention
  - Example: `"train-code-*"`, `"*-math-*"`, `"*puzzle*"`

**Example: Adding a New Dataset**

```python
# In __init__.py

# Option 1: Exact match for specific dataset
if data_source in ["my_new_dataset", "another_dataset"]:
    from . import my_scorer
    return my_scorer.compute_score(...)

# Option 2: Pattern match for dataset family
if "my_keyword" in data_source:
    from . import my_scorer
    return my_scorer.compute_score(...)
```

---

## Usage Examples

### Basic Usage

```python
from datatrove.utils.reward_score import default_compute_score

# Example 1: Math problem with XML format (Qwen/Llama)
model_output_xml = """<think>
Let me solve this step by step.
First, I'll identify what we need to find.
2 + 2 = 4
</think>
\\boxed{4}"""

result = default_compute_score(
    data_source="openai/gsm8k",
    solution_str=model_output_xml,
    ground_truth="\\boxed{4}"  # Note: ground truth should match expected format
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

```python
# Example 2: Same problem with GPT-OSS format
model_output_gpt_oss = """<|start|>assistant<|channel|>analysis<|message|>
Let me solve this step by step.
First, I'll identify what we need to find.
2 + 2 = 4
<|end|>

<|start|>assistant<|channel|>final<|message|>
\\boxed{4}
<|return|>"""

result = default_compute_score(
    data_source="openai/gsm8k",
    solution_str=model_output_gpt_oss,
    ground_truth="\\boxed{4}",
    format_type="gpt_oss"  # Explicit format (or use "auto")
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

### Logic Reasoning

```python
# Example 3: Ordering puzzle
model_output = """<think>
Let me analyze the constraints...
Based on the clues, the order must be A, B, C.
</think>
["A", "B", "C"]"""

result = default_compute_score(
    data_source="ordering_puzzle",
    solution_str=model_output,
    ground_truth=["A", "B", "C"]
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

### Table QA

```python
# Example 4: Table question answering
model_output = """<think>
Looking at the table, I need to find cities with population > 1M...
</think>
{"answer": ["Tokyo", "Delhi", "Shanghai"]}"""

result = default_compute_score(
    data_source="WTQ",
    solution_str=model_output,
    ground_truth=["Tokyo", "Delhi", "Shanghai"]
)

print(result)
# {'score': 1.0, ...}
```

### Document QA

```python
# Example 5: Multi-hop document QA
model_output = """<think>
First, I need to find where John lives from Document 1...
Then, I need to check Document 2 for the population...
</think>
the answer is Paris"""

result = default_compute_score(
    data_source="multihoprag",
    solution_str=model_output,
    ground_truth="Paris"
)

print(result)
# {'score': 1.0, 'em': 1.0, 'sub_em': 1.0, 'f1': 1.0, ...}
```

### Code Execution

```python
# Example 6: Python code execution (automatic detection)
model_output = """<think>
I need to write a function that returns the sum.
</think>
```python
def solution(a, b):
    return a + b
```"""

result = default_compute_score(
    data_source="codecontests",
    solution_str=model_output,
    ground_truth={"inputs": ["5\\n3"], "outputs": ["8"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

```python
# Example 7: C++ code execution (automatic language detection)
model_output = """<think>
I'll implement this in C++ for better performance.
</think>
```cpp
#include <iostream>
using namespace std;
int main() {
    int a, b;
    cin >> a >> b;
    cout << a + b << endl;
    return 0;
}
```"""

result = default_compute_score(
    data_source="train-code-leetcode-Medium",
    solution_str=model_output,
    ground_truth={"inputs": ["5 3"], "outputs": ["8"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

```python
# Example 8: Java code execution (automatic language detection)
model_output = """<think>
I'll solve this using Java.
</think>
```java
import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();
        int b = sc.nextInt();
        System.out.println(a + b);
    }
}
```"""

result = default_compute_score(
    data_source="train-code-taco-medium",
    solution_str=model_output,
    ground_truth={"inputs": ["5 3"], "outputs": ["8"]},
    sandbox_fusion_url="http://sandbox-server:5000"
)

print(result)
# {'score': 1.0, 'reward_think': 1.0, 'reward_fmt': 1.0}
```

### Batch Processing

```python
# Example 9: Evaluate multiple samples
samples = [
    {"data_source": "openai/gsm8k", "output": "...", "ground_truth": "42"},
    {"data_source": "WTQ", "output": "...", "ground_truth": ["Paris"]},
    # ... more samples
]

results = []
for sample in samples:
    result = default_compute_score(
        data_source=sample["data_source"],
        solution_str=sample["output"],
        ground_truth=sample["ground_truth"]
    )
    results.append(result)

# Aggregate metrics
avg_score = sum(r["score"] for r in results) / len(results)
avg_think = sum(r["reward_think"] for r in results) / len(results)
avg_fmt = sum(r["reward_fmt"] for r in results) / len(results)

print(f"Average Score: {avg_score:.2f}")
print(f"Average Think Reward: {avg_think:.2f}")
print(f"Average Format Reward: {avg_fmt:.2f}")
```

---

## Adding New Datasets

### Step 1: Determine if You Need a New Scorer

Check if an existing scorer fits your task:

```
┌─────────────────────────────────────────────────────────────┐
│ Does your task involve...                                   │
├─────────────────────────────────────────────────────────────┤
│ → Math problems with numeric/symbolic answers?              │
│   USE: math.py                                              │
│                                                              │
│ → Structured outputs (lists, dicts, arrays)?                │
│   USE: logic.py                                             │
│                                                              │
│ → Table reasoning with boxed format?                        │
│   USE: table_boxed.py                                       │
│                                                              │
│ → Free text answers with EM/F1 scoring?                     │
│   USE: docqa.py                                             │
│                                                              │
│ → Multiple choice questions?                                │
│   USE: long.py                                              │
│                                                              │
│ → Something entirely different?                             │
│   CREATE: new_domain.py                                     │
└─────────────────────────────────────────────────────────────┘
```

### Step 2: Create a New Scorer (If Needed)

**Template**: `new_domain.py`

```python
"""
Reward scorer for [DOMAIN NAME].

Supports both XML and GPT-OSS formats:
- XML: <think>...</think> and optionally <answer>...</answer>
- GPT-OSS: <|channel|>analysis and <|channel|>final

Implements cascade reward system:
1. reward_think: Validates thinking section formatting
2. reward_fmt: Validates answer format and extraction
3. score: Compares with ground truth
"""

from typing import Dict, Any, Union
from .utils import parse_think, normalize_ground_truth

def compute_score(
    model_output: str,
    ground_truth: Union[str, Dict, list],
    data_source: str = "new_domain",
    format_type: str = "auto",
    **kwargs
) -> Dict[str, float]:
    """
    Compute score for [DOMAIN NAME] tasks.

    Args:
        model_output: Model's response text
        ground_truth: Expected answer
        data_source: Dataset identifier
        format_type: "auto", "xml", or "gpt_oss"
        **kwargs: Additional arguments (e.g., timeout_score)

    Returns:
        Dict with keys:
            - score: Correctness score (0.0 to 1.0)
            - reward_think: Thinking format reward (0.0 or 1.0)
            - reward_fmt: Answer format reward (0.0 or 1.0)
    """
    # Initialize rewards
    reward_think = 0.0
    reward_fmt = 0.0
    score = 0.0

    # Stage 1: Validate thinking section
    try:
        text_without_think, think_success = parse_think(model_output, format_type=format_type)
        reward_think = 1.0 if think_success else 0.0
    except Exception as e:
        print(f"[new_domain] Error in parse_think: {e}")
        reward_think = 0.0

    # Cascade failure: Only check format if thinking is valid
    if reward_think == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Stage 2: Extract and validate answer format
    try:
        # TODO: Implement answer extraction logic
        # Example: Extract from <answer> tags or GPT-OSS final channel
        answer_extracted = extract_your_answer(text_without_think)
        reward_fmt = 1.0 if answer_extracted is not None else 0.0
    except Exception as e:
        print(f"[new_domain] Error extracting answer: {e}")
        reward_fmt = 0.0

    # Cascade failure: Only compute score if format is valid
    if reward_fmt == 0.0:
        return {
            "score": score,
            "reward_think": reward_think,
            "reward_fmt": reward_fmt,
        }

    # Stage 3: Compare with ground truth
    try:
        gt_normalized = normalize_ground_truth(ground_truth)

        # TODO: Implement your comparison logic
        # Examples:
        # - String comparison: score = 1.0 if answer == gt_normalized else 0.0
        # - Numeric comparison: score = 1.0 if abs(answer - gt_normalized) < tolerance else 0.0
        # - List comparison: score = jaccard_similarity(answer, gt_normalized)

        score = your_comparison_function(answer_extracted, gt_normalized)
    except Exception as e:
        print(f"[new_domain] Error comparing answer: {e}")
        score = 0.0

    return {
        "score": float(score),
        "reward_think": float(reward_think),
        "reward_fmt": float(reward_fmt),
    }


def extract_your_answer(text: str) -> Any:
    """
    Extract answer from model output.

    Should handle both:
    - XML format: <answer>...</answer> or plain text
    - GPT-OSS format: <|channel|>final content

    Returns:
        Extracted answer or None if extraction fails
    """
    # TODO: Implement extraction logic
    # See logic.py extract_answer_from_tags() for reference
    pass


def your_comparison_function(predicted: Any, ground_truth: Any) -> float:
    """
    Compare predicted answer with ground truth.

    Returns:
        Score between 0.0 and 1.0
    """
    # TODO: Implement comparison logic
    # Examples in logic.py: compare_lists, compare_dicts, compare_strings
    pass
```

### Step 3: Register in `__init__.py`

Add routing logic to `default_compute_score()`:

```python
# In __init__.py

def default_compute_score(data_source, solution_str, ground_truth, **kwargs):
    # ... existing routing logic ...

    # Add your new domain
    elif data_source in ["your_dataset1", "your_dataset2"] or "your_keyword" in data_source:
        from . import new_domain

        res = new_domain.compute_score(
            solution_str,
            ground_truth,
            data_source=data_source,
            **kwargs
        )
        return res

    # ... rest of the function ...
```

### Step 4: Add Tests

**Template**: `tests/utils/reward_score/test_new_domain.py`

```python
"""
Test new_domain scorer with both XML and GPT-OSS formats.
"""

import pytest
from datatrove.utils.reward_score import default_compute_score


class TestNewDomainScorer:
    """Test new_domain.py scorer."""

    def test_correct_answer_xml_format(self):
        """Test correct answer in XML format."""
        model_output = (
            '<think>Reasoning process...</think>\n'
            '<answer>42</answer>'
        )
        ground_truth = '42'
        result = default_compute_score("your_dataset", model_output, ground_truth)

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_correct_answer_gpt_oss_format(self):
        """Test correct answer in GPT-OSS format."""
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>'
            'Reasoning process...'
            '<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>'
            '42'
            '<|return|>'
        )
        ground_truth = '42'
        result = default_compute_score(
            "your_dataset",
            model_output,
            ground_truth,
            format_type="gpt_oss"
        )

        assert result["score"] == 1.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_wrong_answer(self):
        """Test wrong answer."""
        model_output = '<think>Reasoning...</think>\n<answer>99</answer>'
        ground_truth = '42'
        result = default_compute_score("your_dataset", model_output, ground_truth)

        assert result["score"] == 0.0
        assert result["reward_think"] == 1.0
        assert result["reward_fmt"] == 1.0

    def test_cascade_failure_no_thinking(self):
        """Test cascade failure when thinking section is missing."""
        model_output = '<answer>42</answer>'
        ground_truth = '42'
        result = default_compute_score("your_dataset", model_output, ground_truth)

        # Thinking failed, so format and score should also be 0.0 (cascade)
        assert result["score"] == 0.0
        assert result["reward_think"] == 0.0
        assert result["reward_fmt"] == 0.0

    def test_cascade_failure_invalid_format(self):
        """Test cascade failure when answer extraction fails."""
        model_output = '<think>Reasoning...</think>\n[invalid format]'
        ground_truth = '42'
        result = default_compute_score("your_dataset", model_output, ground_truth)

        assert result["reward_think"] == 1.0  # Thinking passed
        assert result["reward_fmt"] == 0.0   # Format failed
        assert result["score"] == 0.0        # Score not evaluated (cascade)

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        # GPT-OSS format should be auto-detected
        model_output = (
            '<|start|>assistant<|channel|>analysis<|message|>Reasoning...<|end|>\n'
            '<|start|>assistant<|channel|>final<|message|>42<|return|>'
        )
        ground_truth = '42'
        result = default_compute_score("your_dataset", model_output, ground_truth)

        assert result["score"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Step 5: Data Preprocessing Integration

If you're creating training data, prepare it in the format expected by the training pipeline:

```python
# data_preprocess/your_domain/preprocess.py

import pandas as pd
from datasets import Dataset

def preprocess_your_dataset():
    """
    Preprocess your dataset into the format expected by the reward scorer.

    Required fields:
    - prompt: List of chat messages (user query)
    - data_source: Dataset identifier for routing
    - reward_model: Dict with 'ground_truth' key
    - extra_info: Optional metadata
    """
    samples = []

    for item in your_raw_data:
        samples.append({
            "prompt": [
                {
                    "role": "user",
                    "content": item["question"]
                }
            ],
            "data_source": "your_dataset",  # Must match routing in __init__.py
            "reward_model": {
                "ground_truth": item["answer"]
            },
            "extra_info": {
                # Optional: any additional metadata
                "difficulty": item.get("difficulty"),
                "category": item.get("category"),
            }
        })

    # Save as parquet
    df = pd.DataFrame(samples)
    df.to_parquet("data/train/your_dataset.parquet")

    return Dataset.from_pandas(df)
```

**Key Points**:
- `data_source` must match the routing pattern in `__init__.py`
- `reward_model.ground_truth` will be passed to your scorer
- Add task-specific instructions to the prompt (e.g., "Please output the final answer within \\boxed{}")
- Do NOT add "think step by step" instructions (handled by chat template)

---

## Adding New Formats

If you need to support a new response format (e.g., Claude format, Gemini format), follow these steps:

### Step 1: Understand the BaseFormatHandler Interface

All format handlers must implement:

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any

class BaseFormatHandler(ABC):
    """Base class for format handlers."""

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the format name (e.g., 'xml', 'gpt_oss', 'claude')."""
        pass

    @abstractmethod
    def detect(self, text: str) -> bool:
        """
        Detect if text is in this format.

        Should check for unique markers that identify this format.
        """
        pass

    @abstractmethod
    def extract_thinking(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract thinking/reasoning content from text.

        Returns:
            (thinking_content, success)
            - thinking_content: Extracted reasoning or None
            - success: True if format is valid (thinking can be None but format OK)
        """
        pass

    @abstractmethod
    def remove_thinking(self, text: str) -> str:
        """
        Remove thinking section from text, returning the remainder.
        """
        pass

    @abstractmethod
    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract final answer/response from text.

        Returns:
            Final response content or None if not found
        """
        pass

    @abstractmethod
    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from text.

        Returns:
            List of dicts with 'name' and 'parameters' keys
        """
        pass

    @abstractmethod
    def check_format(self, text: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Check if text follows the expected format structure.

        Args:
            text: Text to validate
            expected_components: List of expected components (e.g., ['thinking', 'response'])
        """
        pass
```

### Step 2: Implement Your Format Handler

**Example**: `format_handlers/claude_format.py`

```python
"""
Claude format handler.

Supports Claude's XML-like format with specific tags:
- <thinking>...</thinking> for reasoning
- <answer>...</answer> for final answers
"""

import re
from typing import Optional, Tuple, List, Dict, Any
from .base import BaseFormatHandler


class ClaudeFormatHandler(BaseFormatHandler):
    """Handler for Claude response format."""

    @property
    def format_name(self) -> str:
        return "claude"

    def detect(self, text: str) -> bool:
        """Detect Claude format by looking for <thinking> tags."""
        return '<thinking>' in text and '</thinking>' in text

    def extract_thinking(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract content from <thinking>...</thinking> tags.

        Returns:
            (thinking_content, success)
        """
        pattern = r'<thinking>(.*?)</thinking>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip(), True

        # Check if thinking tags are present but malformed
        if '<thinking>' in text:
            return None, False

        # No thinking tags - this is OK (thinking is optional)
        return None, True

    def remove_thinking(self, text: str) -> str:
        """Remove <thinking>...</thinking> sections from text."""
        pattern = r'<thinking>.*?</thinking>'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
        return text.strip()

    def extract_final_response(self, text: str) -> Optional[str]:
        """
        Extract content from <answer>...</answer> tags.
        """
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        # Fallback: return text after </thinking>
        if '</thinking>' in text:
            parts = text.split('</thinking>', 1)
            if len(parts) > 1:
                return parts[1].strip()

        return None

    def extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Claude format.

        Claude uses: <function_calls>...</function_calls>
        """
        tools = []

        # Implement your tool call extraction logic
        # This is a placeholder

        return tools

    def check_format(self, text: str, expected_components: Optional[List[str]] = None) -> bool:
        """
        Check if text follows Claude format structure.
        """
        has_thinking = '<thinking>' in text and '</thinking>' in text
        has_answer = '<answer>' in text and '</answer>' in text

        if expected_components is None:
            # Default: at least one of thinking or answer should be present
            return has_thinking or has_answer

        # Check for specific components
        component_map = {
            'thinking': has_thinking,
            'answer': has_answer,
        }

        return all(component_map.get(comp, False) for comp in expected_components)
```

### Step 3: Register the Format Handler

Add to `format_handlers/__init__.py`:

```python
# Import your handler
from .claude_format import ClaudeFormatHandler

# Add to handlers registry
_HANDLERS = {
    "xml": XMLFormatHandler(),
    "gpt_oss": GPTOSSFormatHandler(),
    "claude": ClaudeFormatHandler(),  # Add here
}

# Update detect_format() if needed
def detect_format(text: str) -> str:
    """
    Auto-detect format from text.

    Checks handlers in order of specificity (most specific first).
    """
    # Check Claude format
    if ClaudeFormatHandler().detect(text):
        return "claude"

    # Check GPT OSS format
    if GPTOSSFormatHandler().detect(text):
        return "gpt_oss"

    # Default to XML format
    return "xml"
```

### Step 4: Update Existing Scorers (If Needed)

Most scorers already use `parse_think()` and `parse_answer()`, which automatically use format handlers. No changes needed unless you have custom extraction logic.

**Verify compatibility**:
```python
from datatrove.utils.reward_score.utils import parse_think, parse_answer

# Test your new format
claude_output = """<thinking>
Reasoning process...
</thinking>
<answer>42</answer>"""

text_without_think, success = parse_think(claude_output, format_type="claude")
print(f"Text: {text_without_think}, Success: {success}")
# Expected: Text: "<answer>42</answer>", Success: True

answer, format_idx = parse_answer(text_without_think, format_type="claude")
print(f"Answer: {answer}, Format: {format_idx}")
# Expected: Answer: "42", Format: 1 (or appropriate index)
```

### Step 5: Add Tests

```python
# tests/utils/reward_score/test_claude_format.py

from datatrove.utils.reward_score.format_handlers.claude_format import ClaudeFormatHandler

class TestClaudeFormatHandler:
    """Test Claude format handler."""

    def test_detect_claude_format(self):
        """Test detection of Claude format."""
        handler = ClaudeFormatHandler()

        claude_text = "<thinking>reasoning</thinking><answer>42</answer>"
        assert handler.detect(claude_text) is True

        xml_text = "<think>reasoning</think>42"
        assert handler.detect(xml_text) is False

    def test_extract_thinking(self):
        """Test extraction of thinking content."""
        handler = ClaudeFormatHandler()

        text = "<thinking>Step 1: Analyze\nStep 2: Solve</thinking><answer>42</answer>"
        thinking, success = handler.extract_thinking(text)

        assert success is True
        assert "Step 1" in thinking
        assert "Step 2" in thinking

    def test_remove_thinking(self):
        """Test removal of thinking section."""
        handler = ClaudeFormatHandler()

        text = "<thinking>reasoning</thinking><answer>42</answer>"
        result = handler.remove_thinking(text)

        assert "<thinking>" not in result
        assert "<answer>42</answer>" in result

    def test_extract_final_response(self):
        """Test extraction of final answer."""
        handler = ClaudeFormatHandler()

        text = "<thinking>reasoning</thinking><answer>42</answer>"
        answer = handler.extract_final_response(text)

        assert answer == "42"
```

### Step 6: Integration Testing

Test with actual scorers:

```python
def test_claude_format_with_math_scorer():
    """Test Claude format with math scorer."""
    from datatrove.utils.reward_score import default_compute_score

    model_output = """<thinking>
    Let me solve 2 + 2.
    2 + 2 = 4
    </thinking>
    <answer>\\boxed{4}</answer>"""

    result = default_compute_score(
        data_source="openai/gsm8k",
        solution_str=model_output,
        ground_truth="\\boxed{4}",
        format_type="claude"
    )

    assert result["score"] == 1.0
    assert result["reward_think"] == 1.0
    assert result["reward_fmt"] == 1.0
```

---

## Testing

### Running Tests

```bash
# Run all reward_score tests
pytest tests/utils/reward_score/ -v

# Run specific domain tests
pytest tests/utils/reward_score/test_math.py -v
pytest tests/utils/reward_score/test_logic.py -v

# Run format handler tests
pytest tests/utils/reward_score/test_format_handlers.py -v

# Run GPT-OSS integration tests
pytest tests/utils/reward_score/test_gpt_oss_all_scorers.py -v

# Run with coverage
pytest tests/utils/reward_score/ --cov=datatrove.utils.reward_score --cov-report=html
```

### Validation Scripts

Two utility scripts are provided for validating dataset coverage and testing scorer routing:

**1. Dataset Coverage Validator** (`scripts/check_dataset_coverage.py`):
- Validates all HuggingFace Hub datasets against the reward scorer router
- Identifies covered/uncovered datasets
- Supports pattern-based matching
- Exits with error if uncovered datasets are found

```bash
# Check default username (sungyub)
python scripts/check_dataset_coverage.py

# Check specific username
python scripts/check_dataset_coverage.py --username your_username

# Check with detailed output
python scripts/check_dataset_coverage.py --verbose
```

**2. Scoring System Tester** (`scripts/test_scoring.py`):
- Smoke tests for reward scoring system
- Tests math, ToolRL, and code domains
- Validates router dispatching works correctly
- Includes dry-run validation for code scoring (checks sandbox requirement)

```bash
# Run all smoke tests
python scripts/test_scoring.py

# Run with verbose output
python scripts/test_scoring.py --verbose
```

**Use Cases:**
- Ensure completeness when adding new datasets to HuggingFace Hub
- Quick validation of scoring system after code changes
- CI/CD integration for automated validation
- Debug routing issues with new dataset identifiers

### Test Organization

```
tests/utils/reward_score/
├── test_math.py                      # Math scorer tests (19 tests)
├── test_logic.py                     # Logic scorer tests (25 tests)
├── test_table_r1_zero_scorers.py    # Table scorers tests (20 tests)
├── test_docqa_rl_verl_scorers.py    # DocQA scorers tests (30 tests)
├── test_format_handlers.py           # Format handler tests (50+ tests)
├── test_gpt_oss_all_scorers.py      # GPT-OSS integration (21 tests)
├── test_utils.py                     # Utility function tests (40+ tests)
├── test_integration.py               # End-to-end integration tests
└── test_refactored_utils.py         # Backward compatibility tests
```

### Writing Good Tests

**Test both formats**:
```python
def test_correct_answer_xml():
    """Test with XML format."""
    output = "<think>reasoning</think>\\boxed{42}"
    result = compute_score(output, "42")
    assert result["score"] == 1.0

def test_correct_answer_gpt_oss():
    """Test with GPT-OSS format."""
    output = "<|start|>assistant<|channel|>analysis<|message|>reasoning<|end|>\n<|start|>assistant<|channel|>final<|message|>42<|return|>"
    result = compute_score(output, "42", format_type="gpt_oss")
    assert result["score"] == 1.0
```

**Test cascade failures**:
```python
def test_cascade_failure_no_thinking():
    """Test that format and score fail when thinking fails."""
    output = "\\boxed{42}"  # Missing thinking
    result = compute_score(output, "42")

    assert result["reward_think"] == 0.0
    assert result["reward_fmt"] == 0.0  # Not evaluated
    assert result["score"] == 0.0       # Not evaluated
```

**Test edge cases**:
```python
def test_empty_thinking():
    """Test empty thinking section."""
    output = "<think></think>\\boxed{42}"
    result = compute_score(output, "42")
    assert result["reward_think"] == 1.0  # Empty is OK

def test_malformed_tags():
    """Test malformed XML tags."""
    output = "<think>reasoning\\boxed{42}"  # Missing </think>
    result = compute_score(output, "42")
    assert result["reward_think"] == 0.0
```

---

## Advanced Topics

### External Dependencies

#### SandboxFusion for Code Execution

Code execution tasks (codecontests, apps) require the SandboxFusion service:

```bash
# Local setup
git clone https://github.com/bytedance/SandboxFusion.git
cd SandboxFusion
poetry install
make run-online

# SLURM deployment
enroot import docker://varad0309/code_sandbox:server
sbatch scripts/sandbox/run_server.sbatch

# Configure
export SANDBOX_FUSION_SERVERS="server1,server2,server3"
```

#### LLM-as-Judge for STEM Verification

STEM tasks may use LLM-as-judge for complex answer verification:

```bash
# Serve general-verifier model (1.5B)
sbatch scripts/tools/serve_llm_as_verifier.sh

# Configure
export STEM_LLM_JUDGE_URL="http://your-verifier-endpoint"
```

### Custom Comparison Logic

For domain-specific comparison needs:

```python
def custom_comparison(predicted: str, ground_truth: str) -> float:
    """
    Implement custom comparison logic.

    Examples:
    - Fuzzy string matching
    - Semantic similarity (embedding-based)
    - Domain-specific equivalence (e.g., chemical formulas)
    """
    # Example: Fuzzy matching with threshold
    from difflib import SequenceMatcher

    similarity = SequenceMatcher(None, predicted, ground_truth).ratio()
    return 1.0 if similarity > 0.8 else 0.0
```

### Timeout Handling

For tasks with expensive verification (code execution, simulation):

```python
from timeout_decorator import timeout

@timeout(30)  # 30 second timeout
def compute_score_with_timeout(model_output, ground_truth, **kwargs):
    """Scorer with timeout protection."""
    try:
        # Your scoring logic here
        return compute_score(model_output, ground_truth, **kwargs)
    except TimeoutError:
        # Return timeout score
        return {
            "score": kwargs.get("timeout_score", 0.0),
            "reward_think": 0.0,
            "reward_fmt": 0.0,
        }
```

See `logic.py` for timeout implementation examples.

### Handling Ground Truth Formats

The `normalize_ground_truth()` utility handles various GT formats:

```python
from datatrove.utils.reward_score.utils import normalize_ground_truth

# Dict format
gt = {"answer": "Paris"}
normalized = normalize_ground_truth(gt)  # Returns: "Paris"

# List format
gt = ["Paris"]
normalized = normalize_ground_truth(gt)  # Returns: "Paris"

# List of dicts
gt = [{"answer": "Paris"}]
normalized = normalize_ground_truth(gt)  # Returns: "Paris"

# Plain value
gt = "Paris"
normalized = normalize_ground_truth(gt)  # Returns: "Paris"

# Custom key
gt = {"city": "Paris"}
normalized = normalize_ground_truth(gt, key="city")  # Returns: "Paris"
```

---

## References

### Papers & Projects

- **Table-R1**: Original table reasoning framework
- **Qwen-Doc**: Document reasoning scorer implementation
- **veRL**: RL training framework (this module integrates with veRL)
- **GPT-OSS-120B**: [HuggingFace Model](https://huggingface.co/openai/gpt-oss-120b)

### Related Datasets

- **sungyub/toolrl-verl**: [🔗 HuggingFace](https://huggingface.co/datasets/sungyub/toolrl-verl)
- **sungyub/codev-r1-verl**: [🔗 HuggingFace](https://huggingface.co/datasets/sungyub/codev-r1-verl)
- **sungyub/ifbench-verl**: [🔗 HuggingFace](https://huggingface.co/datasets/sungyub/ifbench-verl)

### Source Code

- Main module: `src/datatrove/utils/reward_score/`
- Tests: `tests/utils/reward_score/`
- Data preprocessing: `data_preprocess/`

---

## Contributing

When contributing new scorers or format handlers:

1. **Follow the cascade reward pattern**: All scorers must implement the 3-stage system
2. **Support both formats**: Test with XML and GPT-OSS formats
3. **Add comprehensive tests**: Include normal cases, edge cases, and cascade failures
4. **Document your scorer**: Add docstrings with examples
5. **Update this README**: Add your dataset to the supported datasets table

For questions or issues, please refer to the main project documentation.
