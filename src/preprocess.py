from pathlib import Path
from PIL import Image
import shutil
import yaml
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[RichHandler(show_time=False)] 
)
logger = logging.getLogger("rich_logger")

def delete_and_copy(root, origin_root):
    shutil.rmtree(root, ignore_errors=True)
    logger.info(f"Deleted existing directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created new directory: {root}")
    for path in origin_root.glob("**/*"):
        if path.is_file():
            new_path = root / path.relative_to(origin_root)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_bytes(path.read_bytes())


def check_hash(root, origin_root):
    for path in root.glob("**/*"):
        if path.is_file():
            origin_path = origin_root / path.relative_to(root)
            if not origin_path.exists():
                logger.warning(f"{origin_path} does not exist")
                return False
            if path.stat().st_size != origin_path.stat().st_size:
                logger.warning(f"{path} and {origin_path} have different sizes")
                return False
    logger.info("All files have matching hashes and sizes.")
    return True


def check_sizes(root):
    files = [file.stem for file in root.glob("**/*.jpeg")]
    bacteria_files = [file for file in files if "bacteria" in file]
    virus_files = [file for file in files if "virus" in file]
    normal_files = [file for file in files if "bacteria" not in file and "virus" not in file]

    logger.info(f"Total files: {len(files)}, Bacteria: {len(bacteria_files)}, Virus: {len(virus_files)}, Normal: {len(normal_files)}")
    return (len(files) == (len(bacteria_files) + len(virus_files) + len(normal_files)))


def move_all_files_to_new_folder(root):
    files = [file for file in root.glob("**/*.jpeg")]
    shutil.rmtree(root / "all", ignore_errors=True)
    logger.info(f"Deleted existing 'all' directory in {root}")

    for file in files:
        new_path = root / "all" / file.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        file.rename(new_path)

    shutil.rmtree(root / "train", ignore_errors=True)
    shutil.rmtree(root / "test", ignore_errors=True)
    shutil.rmtree(root / "val", ignore_errors=True)
    logger.info("Deleted 'train', 'test', and 'val' directories.")


def create_classes_dataset(root):
    bacteria_files = [file for file in root.glob("**/*.jpeg") if "bacteria" in file.stem]
    virus_files = [file for file in root.glob("**/*.jpeg") if "virus" in file.stem]
    normal_files = [file for file in root.glob("**/*.jpeg") if "bacteria" not in file.stem and "virus" not in file.stem]

    for file in bacteria_files:
        dst = root / "BACTERIA" / file.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        file.rename(dst)

    for file in virus_files:
        dst = root / "VIRUS" / file.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        file.rename(dst)

    for file in normal_files:
        dst = root / "NORMAL" / file.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        file.rename(dst)

    for path in root.glob("**/*"):
        if path.is_dir() and "PNEUMONIA" in path.name:
            shutil.rmtree(path)
            logger.info(f"Removed directory {path}")

    shutil.rmtree(root / "all", ignore_errors=True)
    logger.info("Deleted 'all' directory.")


def delete_small_images(root, min_size):
    files = [file for file in root.glob("**/*.jpeg")]
    for file in files:
        image = Image.open(file)
        width, height = image.size
        if width < min_size or height < min_size:
            logger.info(f"Deleting {file} due to small size ({width}x{height})")
            file.unlink()


def resize_images(root, img_dimension=256):
    files = [file for file in root.glob("**/*.jpeg")]
    logger.info(f"Resizing images to {256} x {256}")
    for file in files:
        image = Image.open(file).convert("L")
        width, height = image.size
        image = image.resize((img_dimension, img_dimension), Image.BILINEAR)
        image.save(file, quality=100)
        image.close()


def __main__():
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)
        logger.info("Loaded configuration file.")

    root = Path(config["new_dataset"])
    origin_root = Path(config["original_dataset"])
    image_size = config["image_size"]

    delete_and_copy(root, origin_root)

    assert check_hash(root, origin_root) == True
    assert check_sizes(root) == True

    move_all_files_to_new_folder(root)

    create_classes_dataset(root)

    delete_small_images(root, image_size)

    resize_images(root, image_size)


if __name__ == "__main__":
    __main__()
