import os
import json
from .utils import Datum, DatasetBase, subsample_classes, build_domain_template
from .oxford_pets import OxfordPets


template = ['a photo of a {}.']
# templates = [
#     "a photo of a {}.",
#     "a picture of a {}.",
#     "a drawing of a {}.",
#     "a sculpture of a {}.",
#     "a black and white photo of a {}.",
#     "a close-up photo of a {}."
# ]


class Caltech101(DatasetBase):

    dataset_dir = 'Caltech101'

    def __init__(self, root, num_shots, subsample="all", load_pseudolabel=False):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '101_ObjectCategories')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_Caltech101.json')

        self.pseudolabel_dict = None
        self.load_pseudolabel = load_pseudolabel
        if load_pseudolabel:
            pseudolabel_path = os.path.join(self.dataset_dir, 'grounding_dino_pseudolabel.json')
            with open(pseudolabel_path, 'r') as f:
                self.pseudolabel_dict = json.load(f)

        # self.template = template
        self.template = build_domain_template("caltech101")

        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir, self.pseudolabel_dict)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        train, val, test = subsample_classes(train, val, test, subsample=subsample)
        super().__init__(train_x=train, val=val, test=test)
