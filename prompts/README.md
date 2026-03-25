# Prompt Design Decisions

## Version Changelog

### v1.0.0 (Initial)
- Initial CoT (Chain of Thought) structured system prompt
- Designed for analyzing ML model evaluation metrics
- Includes reasoning steps for model selection

### v1.1.0 (First Iteration prompt)
- TBD: What changed + why in commit message

## Design Decisions

1. **Chain of Thought**: We use CoT prompting to make the agent's reasoning explicit
2. **Structured Output**: JSON format for machine-readable recommendations
3. **Human Review Flag**: Built-in mechanism for flagging uncertain cases
