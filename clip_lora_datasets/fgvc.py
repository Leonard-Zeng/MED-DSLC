import os
import json
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, subsample_classes, build_domain_template
import numpy as np

"""
template = ['a photo of a {}, a type of aircraft.']
"""
template = ['a photo of a {}.']
class FGVCAircraft(DatasetBase):

    dataset_dir = 'fgvc_aircraft'

    def __init__(self, root, num_shots, post_fix="", subsample="all", load_pseudolabel=False):
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.pseudolabel_dict = None
        self.load_pseudolabel = load_pseudolabel
        if load_pseudolabel:
            pseudolabel_path = os.path.join(self.dataset_dir, 'grounding_dino_pseudolabel.json')
            with open(pseudolabel_path, 'r') as f:
                self.pseudolabel_dict = json.load(f)

        # self.template = template
        self.template = build_domain_template("fgvc")

        classnames = []
        with open(os.path.join(self.dataset_dir, 'variants.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, 'images_variant_train.txt', self.pseudolabel_dict)
        val = self.read_data(cname2lab, 'images_variant_val.txt', self.pseudolabel_dict)
        test = self.read_data(cname2lab, 'images_variant_test.txt', self.pseudolabel_dict)
        classnames = [name+post_fix for name in classnames]
        
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        train, val, test = subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, split_file, pseudolabel_dict=None):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                imname = line[0] + '.jpg'
                classname = ' '.join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                if pseudolabel_dict is not None:
                    img_fname = impath.split("/")[-1]
                    img_fname = img_fname.split(".")[0]
                    bboxes = pseudolabel_dict[img_fname]['boxes'] if pseudolabel_dict.__contains__(img_fname) else np.zeros((0,4))
                    scores = pseudolabel_dict[img_fname]['scores'] if pseudolabel_dict.__contains__(img_fname) else np.zeros((0,4))
                else:
                    bboxes = np.zeros((0,4))
                    scores = np.zeros((0,4))
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname,
                    bboxes=bboxes,
                    scores=scores
                )
                items.append(item)
        
        return items
