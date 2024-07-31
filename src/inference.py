import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt

import settings
from model import Medigrafi
from utils import fileread, load_test_transforms


def visualize_heatmap(image, heatmap, title, path=None):

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(image, cmap='gray')
    ax.imshow(heatmap, alpha=0.5)

    ax.tick_params(axis='x', labelsize=15, pad=10)
    ax.tick_params(axis='y', labelsize=15, pad=10)
    ax.set_title(title, size=15, pad=12.5, loc='center', wrap=True)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    input_directory = Path(args.input_dir)
    input_directory.mkdir(parents=True, exist_ok=True)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model = Medigrafi('../checkpoints/densenet_weights')
    model.eval()
    transform = load_test_transforms()

    predictions = []
    file_paths = list(input_directory.iterdir())

    for image_path in tqdm(file_paths, total=len(file_paths)):
        if image_path.is_dir() is False:
            if image_path.suffix[1:] in settings.ACCEPTED_FILETYPES:

                image_prediction_directory = output_directory / image_path.name.split('.')[0]
                image_prediction_directory.mkdir(parents=True, exist_ok=True)

                image = fileread(image_path)
                inputs = transform(image).unsqueeze(0)

                with torch.no_grad():
                    probs = model(inputs)

                probs = {key: round(value.item(), 4) for (key, value) in zip(settings.LABELS, probs[0])}

                for label, probability in probs.items():
                    heatmap = cv2.resize(
                        model.create_heatmap(label),
                        (image.shape[1], image.shape[0])
                    )
                    visualize_heatmap(
                        image=image,
                        heatmap=heatmap,
                        title=f'{label} Heatmap - Probability {probability:.4f}',
                        path=image_prediction_directory / f'{label}_{probability:.4f}_prediction.png'
                    )

                probs['image_path'] = str(image_path.absolute())
                predictions.append(probs)

    df = pd.DataFrame(predictions)
    df.to_csv(output_directory / 'predictions.csv', index=False)
