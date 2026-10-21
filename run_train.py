from src.train_model import train_models
if __name__ == "__main__":
    results = train_models()
    for name, info in results.items():
        metrics = info["metrics"]
        print(f"{name} metrics: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
