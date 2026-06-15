from omegaconf import OmegaConf

from semcom.utils.config import load_config


def main() -> None:
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()