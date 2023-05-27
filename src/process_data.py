import os.path
import shutil

import PIL
import tensorflow as tf
import click


# in_dir = 'data\\raw\\kaggle'
# out_dir = 'data\\processed\\PetImages'
# n_img = 20


@click.command()
@click.option('-i', '--in_dir', default='data/raw/kaggle')
@click.option('-o', '--out_dir', default='data/processed/PetImages')
@click.option('-n', '--n_img', default=20)
@click.option('-s', '--img_size', default=180)
def process_data(in_dir, out_dir, n_img, img_size):
    make_out_dir(out_dir)
    copy_imgs(in_dir, out_dir, n_img, img_size)


def make_out_dir(out_dir):
    if os.path.exists(out_dir):
        os.remove(out_dir)

    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'Cat'))
    os.mkdir(os.path.join(out_dir, 'Dog'))


def copy_imgs(in_dir, out_dir, n_img, img_size):
    all_imgs = os.listdir(in_dir)
    cat_imgs = [img for img in all_imgs if img.startswith('cat')]
    dog_imgs = [img for img in all_imgs if img.startswith('dog')]

    for cat_img in cat_imgs[:n_img]:
        in_img_path = os.path.join(in_dir, cat_img)
        img = PIL.Image.open(in_img_path)
        img_r = img.resize((img_size, img_size))
        out_img_path = os.path.join(out_dir, "Cat", cat_img)
        img_r.save(out_img_path)

    for dog_img in dog_imgs[:n_img]:
        in_img_path = os.path.join(in_dir, dog_img)
        img = PIL.Image.open(in_img_path)
        img_r = img.resize((img_size, img_size))
        out_img_path = os.path.join(out_dir, "Dog", dog_img)
        img_r.save(out_img_path)


if __name__ == '__main__':
    process_data()
