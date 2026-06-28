import os
import json
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, subsample_classes, build_domain_template

from .oxford_pets import OxfordPets


template = ['a photo of a {}.']


class SUN397(DatasetBase):

    dataset_dir = 'SUN397'

    def __init__(self, root, num_shots, subsample="all", load_pseudolabel=False):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'SUN397')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_SUN397.json')
        self.pseudolabel_dict = None
        self.load_pseudolabel = load_pseudolabel
        if load_pseudolabel:
            pseudolabel_path = os.path.join(self.dataset_dir, 'grounding_dino_pseudolabel.json')
            with open(pseudolabel_path, 'r') as f:
                self.pseudolabel_dict = json.load(f)
        # self.template = template
        self.template = build_domain_template("sun397")

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir, self.pseudolabel_dict)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        train, val, test = subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:] # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split('/')[1:] # remove 1st letter
                names = names[::-1] # put words like indoor/outdoor at first
                classname = ' '.join(names)
                
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=classname
                )
                items.append(item)
        
        return items
