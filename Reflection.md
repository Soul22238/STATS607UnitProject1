# Reflection Document

## Original State of Analysis
The project started with messy, monolithic Python ipynb files for POS tagging using DistilBERT. Paths and hyperparameters were hard-coded, documentation was minimal, and reproducibility was limited. There was no version control, environment isolation, or structured output management.

## Biggest Challenges in Transformation
- Refactoring code to use a centralized configuration (`config.json`) for all paths and hyperparameters.
- Ensuring compatibility between dependencies (numpy, pandas, torch, transformers).
- Modularizing code for data preprocessing, model definition, training, and evaluation.
- Setting up proper environment isolation with `venv` and automating setup via a shell script.
- Retrofitting documentation and PEP 257-compliant docstrings throughout the codebase.

## Improvements Impacting Reproducibility
- **Centralized Configuration:** All parameters moved to `config.json` for easy modification and repeatability.
- **Version Control:** git and GitHub enabled tracking changes and collaboration.
- **Environment Isolation:** `venv` and `requirements.txt` ensured consistent environments.
- **Automated Pipeline:** The `run` script allowed one-command execution.
- **Structured Outputs:** All models, logs, and results saved in `artifacts/` and `results/`.
- **Documentation:** Markdown README and code docstrings improved usability.

## Lessons for Future Projects
- Start with modular code and configuration files.
- Test setup instructions on a fresh environment early.
- Use descriptive commit messages and commit frequently.
- Ask for feedback during development.
- Keep a running log of challenges and solutions.
- Separate scripts for full-data and result-reproduction workflows.

## Time Estimates for Major Components
- Initial messy code: 2 days
- Refactoring and modularization: 2 days
- Environment setup and automation: 1 day
- Documentation and testing: 1 day
- Debugging and feedback: 1 day

## Best Practices Used
- **Version Control:** git with GitHub, frequent and descriptive commits.
- **Environment:** Python `venv`, requirements managed in `requirements.txt`.
- **Documentation:** Markdown README, PEP 257 docstrings.
- **Testing:** pytest for unit tests.
- **Reproducibility:** All outputs saved in `artifacts/` and `results/`.
