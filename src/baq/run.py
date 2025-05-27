import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from baq.pipelines.training_pipeline import training_pipeline
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training pipeline.
    
    Args:
        config: Hydra configuration
    """
    # Resolve the config
    config = OmegaConf.to_container(config, resolve=True)    
    training_pipeline(config)

if __name__ == "__main__":
    main() 