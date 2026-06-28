import os
import json
from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader, subsample_classes, build_domain_template
from .oxford_pets import OxfordPets


template = ['a centered satellite photo of {}.']

# template = ['a photo of a {}.']

NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots, subsample="all", load_pseudolabel=False):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        self.pseudolabel_dict = None
        self.load_pseudolabel = load_pseudolabel
        if load_pseudolabel:
            pseudolabel_path = os.path.join(self.dataset_dir, 'grounding_dino_pseudolabel.json')
            with open(pseudolabel_path, 'r') as f:
                self.pseudolabel_dict = json.load(f)
        # self.template = template
        self.template = build_domain_template("eurosat")

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir, self.pseudolabel_dict)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        train, val, test = subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
    
    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CNAMES[cname_old]
            item_new = Datum(
                impath=item_old.impath,
                label=item_old.label,
                classname=cname_new
            )
            dataset_new.append(item_new)
        return dataset_new
