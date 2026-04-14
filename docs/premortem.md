# Pre-Mortem: Failure Mode Analysis

## Predicted Failure Modes

| Failure Mode | Likelihood | Detection Method | Mitigation |
|-------------|-----------|------------------|------------|
| Dataset file cannot be read | High | Python tool raises file loading error | Log failure, stop execution |
| Missing or invalid model evaluation metrics | Medium | Input schema validation | Flag for human review |
| Incorrect recommendation due to metric tradeoffs | Medium | Compare against deterministic ranking | Require justification or flag for review |
| Agent produces non-JSON output | Low | Output schema validation | Retry with fallback |
| Network/API timeout | Low | Request timeout | Retry with exponential backoff |
| Invalid input format | Medium | Input schema validation | Return validation error |
| Empty dataset | Low | Dataset analysis | Flag for human review |
| All models have equal metrics | Low | Metric comparison | Flag for human review |

## Session 4: Compare with Actual

*To be filled in after running the agent on evaluation set*
