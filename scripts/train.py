from omegaconf import OmegaConf

from pathlib import Path
from semcom.utils.config import load_config
from semcom.utils.io import ensure_dict, save_config
from semcom.utils.seed import set_seed


def main() -> None:
    cfg = load_config()
    
    set_seed(cfg.experiment.seed)

    output_dir = ensure_dict(Path(cfg.experiment.output_dir))
    save_config(cfg, output_dir)

    print(f"Experiment: {cfg.experiment.name}")
    print(f"Output directory: {output_dir}")

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch+1}/{cfg.training.epochs}")

    print("Training completed.")

if __name__ == "__main__":
    main()