from tqdm import tqdm
import torch
import clip
from device_utils import resolve_device


def _infer_device(model, device=None):
    if device is not None:
        return resolve_device(device)
    return next(model.parameters()).device

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc

def correct_flg(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_flg = correct[: topk].float()
    return correct_flg


def clip_classifier(classnames, template, clip_model, device=None):
    device = _infer_device(clip_model, device)
    template = template if isinstance(template, list) else [template]
    print("preparing text embeddings")
    with torch.no_grad():
        clip_weights = []
        for classname in tqdm(classnames):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            # print(texts)
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    print("done preparing text embeddings")
    return clip_weights


def pre_load_features(clip_model, loader, device=None):
    device = _infer_device(clip_model, device)
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.to(device), target.to(device)
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels
