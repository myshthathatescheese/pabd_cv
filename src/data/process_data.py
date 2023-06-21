import os
import shutil

from PIL import Image
import click
import tensorflow as tf


@click.command()
@click.option("-i", "--in_dir", default="data/raw/kaggle")
@click.option("-o", "--out_dir", default="data/processed/pet-images")
@click.option("-n", "--n_img", default=20)
@click.option("-s", "--img_size", default=180)
def process_data(in_dir, out_dir, n_img, img_size):
    """Process pet images."""
    make_out_dir(out_dir)
    process_imgs(in_dir, out_dir, n_img, img_size)
    filter_out_corrupted_images(out_dir)


def make_out_dir(out_dir: str) -> None:
    """Create or empty the output directory."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, "Cat"))
    os.mkdir(os.path.join(out_dir, "Dog"))


def process_imgs(in_dir: str, out_dir: str, n_img: int, img_size: int) -> None:
    """Extract labels from images and store them accordingly."""
    all_imgs = os.listdir(in_dir)
    cat_imgs = [img for img in all_imgs if img.startswith("cat")]
    dog_imgs = [img for img in all_imgs if img.startswith("dog")]

    def resize_and_save(imgs_list: list, category_name: str) -> None:
        """Resize the image and store it in the proper directory."""
        for cat_img in imgs_list[:n_img]:
            in_img_path = os.path.join(in_dir, cat_img)
            img = Image.open(in_img_path)
            img_r = img.resize((img_size, img_size))
            out_img_path = os.path.join(out_dir, category_name, cat_img)
            img_r.save(out_img_path)

    resize_and_save(cat_imgs, "Cat")
    resize_and_save(dog_imgs, "Dog")

def filter_out_corrupted_images(out_dir: str) -> None:
    """Delete corrupted images."""
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(out_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
    print(f"Removed {num_skipped} corrupted images.")


if __name__ == "__main__":
    process_data()
