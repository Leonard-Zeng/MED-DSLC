import torch
import torchvision.transforms as transforms
import clip
import open_clip
import pandas as pd
import os
from clip_lora_datasets import build_dataset
from clip_lora_datasets.class_filter import load_class_filter, filter_dataset_to_classnames
from clip_lora_datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora
from device_utils import resolve_device


def main():

    # Load config file
    args = get_arguments()
    device = resolve_device(args.device)
    args.device = str(device)
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone, device=device)
    # clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/16", pretrained='openai')
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
    shots = -1 if args.eval_only else args.shots
    dataset = build_dataset(args.dataset, args.root_path, shots, preprocess, subsample=args.subsample)
    class_filter = load_class_filter(getattr(args, "class_filter_path", None))
    if class_filter and args.dataset in class_filter:
        dataset = filter_dataset_to_classnames(dataset, class_filter[args.dataset], domain=args.dataset)
    # print(dataset.test[0])
    
    if 'imagenet' in args.dataset:
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

    orig_dataset = args.dataset
    zs_acc, lora_test_acc = run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
    args.dataset = orig_dataset
    
    # Update CSV file with results
    csv_filename = "imagenet_clip_results.csv"
    
    # Create DataFrame if file doesn't exist
    if not os.path.exists(csv_filename):
        df = pd.DataFrame(columns=['method', 'subsample'])
    else:
        df = pd.read_csv(csv_filename)
    
    # Ensure 'method' and 'subsample' columns exist
    if 'method' not in df.columns:
        df['method'] = None
    if 'subsample' not in df.columns:
        df['subsample'] = None
    
    # Update or create row for "clip+lora" with matching subsample
    mask_clip_lora = (df['method'] == 'clip+lora') & (df['subsample'] == args.subsample)
    if mask_clip_lora.any():
        idx = df[mask_clip_lora].index[0]
        df.at[idx, args.dataset] = lora_test_acc
    else:
        # Create new row for clip+lora
        new_row = {'method': 'clip+lora', 'subsample': args.subsample}
        new_row[args.dataset] = lora_test_acc
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Update or create row for "base clip zero-shots" with matching subsample
    mask_zero_shot = (df['method'] == 'base clip zero-shots') & (df['subsample'] == args.subsample)
    if mask_zero_shot.any():
        idx = df[mask_zero_shot].index[0]
        df.at[idx, args.dataset] = zs_acc
    else:
        # Create new row for base clip zero-shots
        new_row = {'method': 'base clip zero-shots', 'subsample': args.subsample}
        new_row[args.dataset] = zs_acc
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

if __name__ == '__main__':
    main()
