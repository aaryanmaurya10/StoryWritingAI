import os
import json
import math
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, output_dir="outputs/metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": []
        }

    def log_epoch(self, epoch, train_loss, val_loss):
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl = math.exp(min(val_loss, 100))
        
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_perplexity"].append(train_ppl)
        self.history["val_perplexity"].append(val_ppl)
        
        json_path = os.path.join(self.output_dir, "metrics_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
            
        self._generate_plots()

    def _generate_plots(self):
        epochs = self.history["epoch"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", marker="o", linestyle="-")
        ax1.plot(epochs, self.history["val_loss"], label="Val Loss", color="#ff7f0e", marker="o", linestyle="--")
        ax1.set_title("Loss Curves Over Epochs", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        ax2.plot(epochs, self.history["train_perplexity"], label="Train PPL", color="#2ca02c", marker="s", linestyle="-")
        ax2.plot(epochs, self.history["val_perplexity"], label="Val PPL", color="#d62728", marker="s", linestyle="--")
        ax2.set_title("Perplexity Curves (Log Scale)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Perplexity Value")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()