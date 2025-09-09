import torch
import os
import time

from torch.optim import Adam
from torch.utils.data import DataLoader

from model import SBIRModel
from utils import seed_everything, calculate_results, compute_distances
from data import load_dataset, DatasetAdapter


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--save', type=str, default='model_XXX',
                        help='Path to save the trained model.')

    parser.add_argument('--unseen', action='store_true', default=False,
                        help='Use unseen user train/val split when using FSCOCO dataset')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')

    parser.add_argument('--batch_size', type=int, default=60,
                        help='Number of samples in each batch.')

    # <type of model (CLIP | Siamese)>;<type of encoder (convnext_base(>;<Initialization Imagenet (only for Siamese)>
    parser.add_argument("--model_path", type=str, default='CLIP;convnext_base',
                        help="path to saved model to load or specifications of a model to load")

    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')

    parser.add_argument('--data_dir', type=str, default='fscoco',
                        help='Directory for the dataset.')

    parser.add_argument('--loss', type=str, default='ICon',
                        help='Loss function to use. Options: InfoNCE, ICon, Triplet. Add ; to separate loss parameters.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    seed_everything(args.seed)

    model_path = args.model_path

    if os.path.exists(model_path):
        model = SBIRModel.load_module(model_path, strict=False, cuda=args.cuda)
    else:
        tokens = model_path.split(";")

        if tokens[0] == "CLIP":
            from model import CLIP_SBIRModel
            model_base = tokens[1] if len(tokens) > 1 else "convnext_base"
            model = CLIP_SBIRModel(model=model_base)
        elif tokens[0] == "Siamese":
            from model import SiameseSBIRModel
            arch = tokens[1] if len(tokens) > 1 else "convnext_base"
            init = tokens[2] if len(tokens) > 2 else "imagenet"
            model = SiameseSBIRModel(arch=arch, init=init)
        else:
            raise ValueError(
                f"Unknown model base: {tokens[0]}. Please specify a valid model base (CLIP, Siamese).")

    sketch_transforms = model.create_sketch_transforms()
    image_transforms = model.create_image_transforms()

    mode = "unseen" if args.unseen else "normal"

    dataset = load_dataset(args.data_dir, mode=mode)
    dataset_train, dataset_test, _ = dataset.split()

    dataset_train = DatasetAdapter(dataset_train, transforms_image=image_transforms,
                                   transforms_sketch=sketch_transforms)
    dataset_test = DatasetAdapter(dataset_test, transforms_image=image_transforms, transforms_sketch=sketch_transforms)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size * 3, shuffle=False)

    # Print configurations
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    if args.cuda:
        model.cuda()
        scaler = None
        if model.use_amp():
            print("Using automatic mixed precision (AMP) for training.")
            scaler = torch.amp.GradScaler()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    lossargs = args.loss.split(";")
    loss = lossargs[0]

    if loss.lower() == "infonce":
        from loss import InfoNCE
        temp = float(lossargs[1]) if len(lossargs) > 1 else 0.05
        loss_fn = InfoNCE(temperature=temp)
    elif loss.lower() == "icon":
        from loss import ICon
        temp = float(lossargs[1]) if len(lossargs) > 1 else 0.07
        alpha = float(lossargs[2]) if len(lossargs) > 2 else 0.2
        loss_fn = ICon(temperature=temp, alpha=alpha)
    elif loss.lower() == "triplet":
        from loss import TripletLoss
        margin = float(lossargs[1]) if len(lossargs) > 1 else 0.2
        loss_fn = TripletLoss(margin=margin)
    else:
        raise ValueError(f"Unknown loss function: {args.loss}. Please specify a valid loss function.")

    tic = time.time()

    if args.save:
        save_path = os.path.abspath(args.save)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader_train):
            if args.cuda:
                data = [d.cuda() for d in data]

            input1, input2 = data[0], data[1]

            if scaler is not None:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output1 = model(input1)
                    output2 = model(input2)

                    loss = loss_fn(output1, output2)
                    running_loss += loss.item()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output1 = model(input1)
                output2 = model(input2)

                loss = loss_fn(output1, output2)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

            if i % 5 == 4:
                print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 5  :0.5f}')
                running_loss = 0.0

        with torch.no_grad():
            model.eval()

            sketch_output = []
            image_output = []


            for data in dataloader_test:
                if args.cuda:
                    data = [d.cuda() for d in data]

                input1, input2 = data[0], data[1]

                output1 = model(input1)
                output2 = model(input2)

                sketch_output.append(output1.cpu())
                image_output.append(output2.cpu())

            sketch_output = torch.concatenate(sketch_output)
            image_output = torch.concatenate(image_output)

            dis = compute_distances(sketch_output.numpy(), image_output.numpy())
            top1, top5, top10, num = calculate_results(dis)

            print(
                f"EPOCH {str(epoch)}: top1: {top1 / float(num):.4f} ({top1}), top5: {top5 / float(num):.4f} ({top5}), top10: {top10 / float(num):.4f} ({top10})")

        if args.save:
            save_path = args.save.replace("XXX", str(epoch))
            # Append suffix to the save path if not already present
            if not save_path.endswith(".pth"):
                save_path += ".pth"

            model.save_module(save_path, epoch=epoch, lr=args.lr, unseen=args.unseen, dataset=dataset_train.name(),
                              elapsed_time=time.time() - tic,
                              timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(tic)))


if __name__ == "__main__":
    main()
