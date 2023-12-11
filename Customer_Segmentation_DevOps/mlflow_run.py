import subprocess

def run_mlflow_commands():

    commands = [
        "mlflow run src/data_preparation -e cleaning -P filepath=../../data/customer_segmentation.csv",  
        "mlflow run src/data_preparation -e encoding",  
        "mlflow run src/data_preparation -e rfm",  

        "mlflow run src/descriptive_stats -e stats",  

        "mlflow run src/dimensionality_reduction -e pca",  


    ]

    for cmd in commands:
        print(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True)

        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
        else:
            print("Command executed successfully")

if __name__ == "__main__":
    run_mlflow_commands()
