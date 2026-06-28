import os
import random
from collections import defaultdict
import json
import numpy as np

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, subsample_classes, build_domain_template

"""
template = ['a photo of a {}, a type of pet.']
"""
template = ['a photo of a {}.']

class OxfordPets(DatasetBase):

    dataset_dir = 'OxfordPets'

    def __init__(self, root, num_shots, subsample="all", load_pseudolabel=False):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')
        self.pseudolabel_dict = None
        self.load_pseudolabel = load_pseudolabel
        if load_pseudolabel:
            pseudolabel_path = os.path.join(self.dataset_dir, 'grounding_dino_pseudolabel.json')
            with open(pseudolabel_path, 'r') as f:
                self.pseudolabel_dict = json.load(f)

        # self.template = template
        self.template = build_domain_template("oxford_pets")

        train, val, test = self.read_split(self.split_path, self.image_dir, self.pseudolabel_dict)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        train, val, test = subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(' ')
                breed = imname.split('_')[:-1]
                breed = '_'.join(breed)
                breed = breed.lower()
                imname += '.jpg'
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1 # convert to 0-based index
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=breed
                )
                items.append(item)
        
        return items
    
    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)
        
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        
        return train, val
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        
        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val,
            'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
    
    @staticmethod
    def read_split(filepath, path_prefix, pseudolabel_dict=None):
        # print(len(pseudolabel_dict.keys()))
        # print(pseudolabel_dict)
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                if pseudolabel_dict is not None:
                    img_fname = impath.split("/")[-1]
                    img_fname = img_fname.split(".")[0]
                    bboxes = np.array(pseudolabel_dict[img_fname]['boxes']) if pseudolabel_dict.__contains__(img_fname) else np.zeros((0,4))
                    scores = np.array(pseudolabel_dict[img_fname]['scores']) if pseudolabel_dict.__contains__(img_fname) else np.zeros((0,1))
                else:
                    bboxes = np.zeros((0,4))
                    scores = np.zeros((0,1))
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname,
                    bboxes=bboxes,
                    scores=scores
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test
