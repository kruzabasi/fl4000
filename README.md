# Privacy-Preserving Federated Learning Framework for Portfolio Optimization

## Description

This project implements and evaluates a federated learning system using FedProx and Differential Privacy for portfolio optimization based on non-IID FTSE 100 stock data, as detailed in the research methodology. The goal is to enable collaborative model training among simulated UK financial institutions while preserving data privacy and addressing challenges related to heterogeneous data distributions.

## Setup Instructions

Follow these steps to set up the project environment:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/kruzabasi/fl400.git
    ```

2.  **Navigate to Project Directory:**
    ```bash
    cd fl400
    ```

3.  **Create Python Virtual Environment:**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv .venv
    # Or: python -m venv .venv
    ```

4.  **Activate Virtual Environment:**
    * **Linux/macOS/WSL (bash/zsh):**
        ```bash
        source .venv/bin/activate
        ```
    * **Windows (CMD):**
        ```cmd
        .\.venv\Scripts\activate.bat
        ```
    * **Windows (PowerShell):**
        ```powershell
        .\.venv\Scripts\Activate.ps1
        # If you encounter execution policy issues, you might need to run:
        # Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
        ```
    You should see `(.venv)` at the beginning of your terminal prompt when the environment is active.

    **Troubleshooting:**
    If you see an error like `source: no such file or directory: .venv/bin/activate`, it means the virtual environment has not been created yet. Run `python3 -m venv .venv` in the project directory to create it, then activate as above.

    **Best Practice:**
    For all development and execution, use the virtual environment by running Python scripts with `.venv/bin/python` or after activating the environment with `source .venv/bin/activate` to ensure dependencies are correctly managed and isolated.

5.  **Install Dependencies:**
    Install the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run all tests and validate the integrity of the codebase, use:

```bash
.venv/bin/python -m pytest
```

All tests should pass successfully. The test suite covers:
- Aggregator logic (FedAvg/FedProx)
- Simulation rounds and results output
- Data alignment and loading
- Integration and privacy mechanisms

(Placeholder) Run the main simulation script to start a federated learning experiment:

```bash
.venv/bin/python src/main_simulation.py --config <path_to_config_file>