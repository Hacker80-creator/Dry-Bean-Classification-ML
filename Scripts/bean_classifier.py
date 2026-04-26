from benchmark_models import benchmark
from config_utils import DEFAULT_CONFIG_PATH


def main():
    print("Initializing training pipeline with model benchmarking...")
    benchmark(config_path=DEFAULT_CONFIG_PATH)
    print("Training pipeline completed. Canonical model saved to models/best_model.joblib")

if __name__ == "__main__":
    main()