# Superposition Stress Test: Code vs. Math Formatting

This project investigates feature superposition in the `Qwen/Qwen3-4B-Instruct-2507` model, for the MATS 9.0 application.

## Hypothesis
Some internal features that support code formatting (e.g., ` ``` `) and math formatting (e.g., `$$`) live in superposition. Steering a discovered "code feature" will causally harm the model's ability to correctly format mathematics in prompts that require both.

## Primary Metrics
1.  **Interference Score**: The average change in the logit margin of math-delimiter tokens when causally steering the code feature on mixed prompts. The margin is defined as `logit(target_math_token) − max_logit(non_math_tokens)`.
2.  **Syntax Validity Rate**: The percentage of generated completions that pass a cheap validator for both Python code blocks (using `ast.parse`) and LaTeX blocks (balanced delimiters).

## Experimental Plan
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Target Layer**: Layer `num_layers // 2`
- **SAE**: 4096 features, trained on activations from diagnostic tokens only.
- **Intervention**: Linear feature steering with `alpha` in `{0, 0.5, 1.0, 1.5, 2.0}`.
- **Controls**: A random feature control and a zero-steer baseline will be used for comparison.