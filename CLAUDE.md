# Project Preferences

## Git Commit Guidelines for ML Experiments
- Git commit messages should be written in English from a machine learning experiment tracking perspective
- Enable easy experiment tracking and history using git commands
- Commit message format:
  ```
  Experiment: [ML Method] [Stage/Checkpoint]
  
  - Experiment Type: [Specific ML method, e.g., GRPO LoRA, SFT, etc.]
  - Training Stage: [checkpoint number or training state]
  - Model Updates: [relevant model file descriptions]
  - Experiment Output: [main output results]
  ```
- Focus on recording experiment methods, training stages, model weight changes
- Use `git log --grep="Experiment Type"` to query specific experiments

## Git Notes for Performance Metrics
- When training scripts are executed, record experiment performance results using `git notes`
- Attach performance metrics to corresponding commits
- Example: `git notes add -m "Loss: 0.234, Accuracy: 87.5%, Training Time: 2h30m" [commit-hash]`
- View notes with: `git log --show-notes` or `git notes show [commit-hash]`