import os
import random
import argparse
from argparse import ArgumentParser

import torch
from lib import dataset
from lib.lightning.lightningmodel import LightningModel


def stylize_image(model, content_file, style_file, content_size=None):
    device = next(model.parameters()).device

    content = dataset.load(content_file)
    style = dataset.load(style_file)

    content = dataset.content_transforms(content_size)(content)
    style = dataset.style_transforms()(style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    output = model(content, style)
    return output[0].detach().cpu()


def parse_args():
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrain_dir', type=str, default='./pretrain')
    parser.add_argument('--content_dir', type=str, default='./train/A')
    parser.add_argument('--style_dir', type=str, default='./train/B')
    parser.add_argument('--output_dir', type=str, default='./output')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args['output_dir'], exist_ok=True)

    content_files = [f for f in os.listdir(args['content_dir']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    style_files = [f for f in os.listdir(args['style_dir']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    model_files = [f for f in os.listdir(args['pretrain_dir']) if f.endswith('.ckpt')]

    if not content_files or not style_files or not model_files:
        raise RuntimeError('内容图、风格图或模型文件缺失')

    content_file = os.path.join(args['content_dir'], random.choice(content_files))
    style_file = os.path.join(args['style_dir'], random.choice(style_files))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model_file in model_files:
        model_path = os.path.join(args['pretrain_dir'], model_file)
        model = LightningModel.load_from_checkpoint(checkpoint_path=model_path)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = stylize_image(model, content_file, style_file)

        output_path = os.path.join(args['output_dir'], f"output_{os.path.splitext(model_file)[0]}.png")
        dataset.save(output, output_path)
