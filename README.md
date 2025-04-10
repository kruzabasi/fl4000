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

5.  **Install Dependencies:**
    Install the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

(Placeholder) Run the main simulation script to start a federated learning experiment:

```bash
python src/main_simulation.py --config <path_to_config_file>
```