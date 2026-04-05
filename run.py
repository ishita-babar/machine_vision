import os
import subprocess
import sys


def run_step(command, step_name):
    print(f"\n{step_name}...")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error in: {step_name}")
        sys.exit(1)
    else:
        print(f"{step_name} completed")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    py = sys.executable

    print("\nStarting Adaptive Document Enhancement Project...\n")

    run_step(f'"{py}" generate_labels.py', "Generating labels")
    run_step(f'"{py}" notebook/train.py', "Training model")

    model_path = os.path.join(root, "backend", "model", "model.pkl")
    if not os.path.exists(model_path):
        print("model.pkl not found. Training failed.")
        sys.exit(1)

    print("Model found.")

    print("\nStarting backend server...")
    backend_process = subprocess.Popen(
        [py, "app.py"],
        cwd=os.path.join(root, "backend"),
    )

    print("\nBackend running at http://127.0.0.1:5000")
    print("\nIn another terminal, start the frontend:")
    print("  cd frontend")
    print("  npm install")
    print("  npm start")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping backend...")
        backend_process.terminate()


if __name__ == "__main__":
    main()
