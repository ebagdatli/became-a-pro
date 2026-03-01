import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path, working_dir, kernel_name="python3"):
    print(f"\nRunning notebook: {notebook_path}")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # 2 hours per cell (CNN training can be slow on CPU)
    ep = ExecutePreprocessor(timeout=7200, kernel_name=kernel_name)

    try:
        ep.preprocess(nb, {'metadata': {'path': working_dir}})
        print("Notebook executed successfully.")
    except Exception as e:
        print("ERROR during execution:")
        print(e)
        sys.exit(1)

def validate_model_folder(competition_path):
    models_path = os.path.join(competition_path, "models")

    if not os.path.exists(models_path):
        print("Models folder not found. Creating it.")
        os.makedirs(models_path)

    files = os.listdir(models_path)

    model_files = [f for f in files if f.endswith(".pkl")]

    if not model_files:
        print("WARNING: No model file found in models/ folder.")
    else:
        print("Model files found:")
        for m in model_files:
            print(" -", m)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python run_competition.py CompetitionFolderName")
        sys.exit(1)

    competition_name = sys.argv[1]
    competition_path = os.path.join(os.getcwd(), competition_name)

    if not os.path.exists(competition_path):
        print("Competition folder not found.")
        sys.exit(1)

    notebook_path = os.path.join(
        competition_path,
        "notebooks",
        "main.ipynb"
    )

    if not os.path.exists(notebook_path):
        print("Notebook not found at notebooks/main.ipynb")
        sys.exit(1)

    # ExercisePrediction uses xgboost etc. - use its venv kernel if available
    kernel = "exercise-prediction" if competition_name == "ExercisePrediction" else "python3"
    run_notebook(notebook_path, competition_path, kernel_name=kernel)
    validate_model_folder(competition_path)

    print("\nCompetition execution completed.")

if __name__ == "__main__":
    main()