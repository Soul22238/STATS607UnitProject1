# Fine-tune DistilBERT for POS Tagging

## Project Structure

- `src/` : Source code for data preprocessing, model definition, training, and evaluation.
- `data/` : Contains the dataset file `conll2000.csv` for POS tagging.
- `artifacts/` : Stores model checkpoints, training logs, and loss plots.
    - `Test_fit_function_model.pt` : Model weights from test fitting.
    - `Test_fit_function_log.txt` : Training log for test fitting.
    - `Test_fit_function_loss.png` : Loss curve for test fitting.
    - `Train_full_model.pt` : Model weights from full training.
    - `Train_full_log.txt` : Training log for full training.
    - `Train_full_model_loss.png` : Loss curve for full training.
- `results/` : Stores evaluation results.
    - `pos_accuracy.json` : POS tagging accuracy metrics in JSON format.
- `env/` : Python virtual environment (auto-created by the run script).
- `requirements.txt` : Python dependencies for the project.
- `run` : Bash script to set up the environment, install dependencies, and run training/evaluation.
- `README.md` : Project instructions and documentation.

## Quick Start

1. Make the run script executable:
    ```bash
    chmod +x run
    ./run
    ```
2. The script will:
    - Set up the Python environment
    - Install dependencies
    - Train the model and save outputs to `artifacts/`
    - Evaluate the model and save results to `results/`

## Output Folders

- **artifacts/**: Contains all model checkpoints, logs, and training loss plots for both test fitting and full training runs. Use these files to analyze training progress or reload models.
- **results/**: Contains evaluation metrics such as POS tagging accuracy in JSON format for further analysis or reporting.

