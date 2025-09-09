import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from data import DatasetAdapter, load_dataset
from model import SBIRModel
from utils import calculate_results, compute_distances, expand_file_names, seed_everything


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA.')

    parser.add_argument('--save', action='store_true', default=False,
                        help='Save generated embeddings for sketches and images.')

    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate a HTML file with visualization of results.')

    parser.add_argument('--unseen', action='store_true', default=False,
                        help='Use unseen user train/val split')

    parser.add_argument('--batch_size', type=int, default=60,
                        help='Number of samples in each batch.')

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')

    parser.add_argument('--data_dir', type=str, default='fscoco',
                        help='Directory for the dataset.')

    parser.add_argument('--models', type=str, nargs='+', help="Paths to the model files")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    seed_everything(args.seed)

    for model_path in expand_file_names(args.models):
        if os.path.exists(model_path):
            model = SBIRModel.load_module(model_path, strict=False, cuda=args.cuda)
        else:
            raise ValueError(f"Model path {model_path} does not exist. Please provide a valid model path.")

        print(f"Evaluating model: {model_path}")

        sketch_transforms = model.create_sketch_transforms()
        image_transforms = model.create_image_transforms()

        mode = "unseen" if args.unseen else "normal"

        dataset = load_dataset(args.data_dir, mode=mode)
        _, dataset_test, _ = dataset.split()

        dataset_test = DatasetAdapter(dataset_test, transforms_sketch=sketch_transforms,
                                      transforms_image=image_transforms)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size * 3, shuffle=False)

        output_file = os.path.splitext(model_path)[0] + "_embeddings.npy"

        if args.cuda:
            model.cuda()

        with torch.no_grad():
            model.eval()

            sketch_output = []
            image_output = []
            for data in tqdm.tqdm(dataloader_test):
                if args.cuda:
                    data = [d.cuda() for d in data]

                output1 = model(data[0])
                output2 = model(data[1])

                sketch_output.append(output1.cpu())
                image_output.append(output2.cpu())

            sketch_output = torch.concatenate(sketch_output).numpy()
            image_output = torch.concatenate(image_output).numpy()

            distances = compute_distances(sketch_output, image_output)

            if args.visualize:
                top1, top5, top10, num = calculate_results(distances, file_names=dataset_test.collection.file_names())
            else:
                top1, top5, top10, num = calculate_results(distances)

            print(f' top1: {top1 / float(num):.4f} ({top1})')
            print(f' top5: {top5 / float(num):.4f} ({top5})')
            print(f' top10: {top10 / float(num):.4f} ({top10})')

            if args.save:
                # Stack embeddings and save them
                embeddings = np.stack([sketch_output, image_output], axis=0)
                np.save(output_file, embeddings)


if __name__ == "__main__":
    main()
