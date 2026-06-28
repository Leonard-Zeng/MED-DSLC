import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image

dir_name2domain_name ={
    "Caltech101": "caltech101",
    "DTD": "dtd",
    "eurosat": "eurosat",
    "fgvc_aircraft": "fgvc",
    "Food101": "food101",
    "imagenet": "imagenet",
    "Flower102": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "CUB200": "cub200",
    "RESISC45": "resisc45",
    "AIBD_Cars": "aibd_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}

# Domain templates for consistent prompt generation
# Base format: "a {adj_keyword} photo of a {noun_keyword} {} {post_keyword}"
# Use "overwrite" to fully override the base template.
DOMAIN_TEMPLATES = {
    "caltech101": {"adj_keyword": "", "noun_keyword": "common object", "post_keyword": ""},
    "eurosat": {"adj_keyword": "remote", "noun_keyword": "", "post_keyword": "from satellite"},
    "stanford_cars": {"adj_keyword": "", "noun_keyword": "car model", "post_keyword": ""},
    "aibd_cars": {"adj_keyword": "", "noun_keyword": "car model", "post_keyword": ""},
    "cub200": {"adj_keyword": "", "noun_keyword": "bird species", "post_keyword": ""},
    "resisc45": {"overwrite": "a remote sensing image of {}."},
    "food101": {"adj_keyword": "", "noun_keyword": "", "post_keyword": ", a type of food"},
    "oxford_pets": {"adj_keyword": "", "noun_keyword": "", "post_keyword": ", a type of pet"},
    "oxford_flowers": {"adj_keyword": "", "noun_keyword": "", "post_keyword": ", a type of flower"},
    "dtd": {"adj_keyword": "close-up", "noun_keyword": "", "post_keyword": "texture"},
    "sun397": {"adj_keyword": "", "noun_keyword": "scene", "post_keyword": "", "overwrite": "a photo taken in {}"},
    "ucf101": {"adj_keyword": "", "noun_keyword": "action", "post_keyword": ""},
    "fgvc": {"adj_keyword": "", "noun_keyword": "", "post_keyword": ", a type of aircraft"},
    "imagenet": {"adj_keyword": "", "noun_keyword": "", "post_keyword": ""}
}


def build_domain_template(domain: str) -> str:
    """Build a template string with a single {} placeholder."""
    template = DOMAIN_TEMPLATES.get(domain, {})
    overwrite = template.get("overwrite")
    if overwrite:
        return overwrite

    adj_keyword = template.get("adj_keyword") or ""
    noun_keyword = template.get("noun_keyword") or ""
    post_keyword = template.get("post_keyword") or ""

    parts = ["a", adj_keyword, "photo", "of", "a", noun_keyword, "{}", post_keyword]
    cleaned = [part for part in parts if part]
    return " ".join(cleaned)


def create_domain_prompt(classname: str, domain: str) -> str:
    """
    Create a domain-specific prompt using the template format.

    Args:
        classname: The class name (e.g., "dog", "BMW")
        domain: The domain name (e.g., "caltech101", "eurosat")

    Returns:
        Formatted prompt string
    """
    template = build_domain_template(domain)
    return template.format(classname)

dataset2prompt = {
    "caltech101": [
        "A photo of a musical instrument, animal, vehicle, plant, tool, household object, or architectural structure.",
        "A photo of a miscellaneous object."
    ],
    "dtd": [
        "“A close-up image of a surface texture—patterned, fibrous, porous, or irregular.”",
        "A close-up image of an unspecified surface texture."
    ],
    "eurosat": [
        "A satellite image of agricultural land, forest, grassland, built-up areas, roads, or water bodies",
        "A satellite image of a miscellaneous landscape."
    ],
    "fgvc": [
        "A photo of a commercial airliner, regional turboprop or jet, business jet, military aircraft, or general-aviation airplane.",
        "A photo of a miscellaneous fixed-wing aircraft."
    ],
    "food101": [
        "A photo of a breakfast item, salad, appetizer, soup, sandwich, pasta dish, pizza, burger, grilled meat, seafood dish, Asian street food, or dessert.",
        "A photo of a miscellaneous food item."
    ],
    # "imagenet": "imagenet",
    "oxford_flowers": [
        "A photo of a flowering plant—an annual garden flower, wildflower, bulb flower, orchid, lily, daisy, or tropical bloom.",
        "A photo of a miscellaneous plant."
    ],
    "oxford_pets": [
        "A photo of a domestic cat breed or a domestic dog breed.",
        "A photo of a miscellaneous animal."
    ],
    "stanford_cars": [
        "A photo of a car with a specific make, model, and body style.",
        "A photo of a miscellaneous vehicle."
    ],
    "aibd_cars": [
        "A photo of a car with a specific make, model, and body style.",
        "A photo of a miscellaneous vehicle.",
    ],
    "cub200": [
        "A photo of a wild bird species.",
        "A photo of a miscellaneous bird.",
    ],
    "resisc45": [
        "An overhead or satellite image of terrain, buildings, roads, water, or land cover.",
        "A remote sensing image of a miscellaneous scene.",
    ],
    "sun397": [
        "A photo of an indoor space, an outdoor urban environment, a transportation setting, a natural landscape, a cultural landmark, a recreational facility, an industrial or agricultural site, or a water feature.",
        "A photo of a miscellaneous environment or scene."
    ],
    "ucf101": [
        "A photo of a person engaged in personal grooming, household chores, sports & fitness activities, or playing a musical instrument.",
        "A photo of a person performing a miscellaneous activity."
    ],
}

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(
            self, impath='', label=0, domain=-1,
            classname='', dataset_name = '',
            real_classname='', real_dataset_name='',
            bboxes=None, scores=None
    ):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname
        self._dataset_name = dataset_name
        self.real_classname = real_classname
        self.real_dataset_name = real_dataset_name
        self._bboxes = bboxes
        self._scores = scores

    @property
    def bboxes(self):
        return self._bboxes
    
    @property
    def scores(self):
        return self._scores

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname
    
    @property
    def dataset_name(self):
        return self._dataset_name


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains
    in_domains = [] # string names of all in-domains
    out_domains = [] # strign names of all out-domains
    domain2classes = {} # domain name -> classes

    def __init__(self, train_x=None, train_u=None, val=None, test=None, remap_labels=True):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data

        if remap_labels:
            self._num_classes = self.get_num_classes(train_x)
            self._lab2cname, self._classnames = self.get_lab2cname(train_x)
            # self._lab2domain, self.domains = self.get_lab2domain(train_x)
            # self.classname2domain = self.get_classname2domain()

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def lab2domain(self):
        return self._lab2domain

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels = (labels + np.max(labels) + 1) % (np.max(labels) + 1) # make -1 the largest
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def get_lab2domain(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.domain))
        mapping = {label: domain for label, domain in container}
        labels = list(mapping.keys())
        labels = (labels + np.max(labels) + 1) % (np.max(labels) + 1) # make -1 the largest
        labels.sort()
        domains = [mapping[label] for label in labels]
        return mapping, domains

    def get_classname2domain(self):
        mapping = {}
        for lbl in self.lab2cname:
            classname = self.lab2cname[lbl]
            domain = self.lab2domain[lbl]
            mapping[classname] = domain
        return mapping

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size=224, transform=None, tokenizer=None, is_train=False, classnames=None,
                 return_img0=False, k_tfm=1, return_domain=False, return_domain_name=False, templates=None, domain=None,
                 random_template=False, load_pseudolabel=False):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0
        self.classnames = classnames
        self.tokenizer = tokenizer
        if type(templates) is not list:
            self.templates = [templates]
        else:
            self.templates = templates
        self.domain = domain
        self.random_template = random_template
        self.load_pseudolabel = load_pseudolabel
        # print(f"**********************************{self.random_template}**********************************")

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)
        self.return_domain = return_domain
        self.return_domain_name = return_domain_name

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath,
            'bboxes': item.bboxes,
            'scores': item.scores
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img
        else:
            output['img'] = img0

        if self.return_img0:
            output['img'] = self.to_tensor(img0)

        if self.return_domain_name:
            return output['img'], output['label'], output['domain'], item.real_dataset_name, item.real_classname
        elif self.return_domain:
            return output['img'], output['label'], output['domain']
        elif self.load_pseudolabel:
            # for this case, we need to return the bboxes and scores and domain
            return output['img'], output['label'], output['domain'], output['bboxes'], output['scores']
        else:
            if self.tokenizer:
                if self.templates is not None:
                    if self.random_template:
                        # Randomly select one template
                        template = random.choice(self.templates)
                        prompt = template.format(self.classnames[output['label']])
                        # print(f"prompt: {prompt}")
                        return output['img'], self.tokenizer(prompt).squeeze()
                    else:
                        # Use all templates (default behavior)
                        prompts = [template.format(self.classnames[output['label']]) for template in self.templates]
                        return output['img'], self.tokenizer(prompts).squeeze()
                else:
                    # Use domain-specific prompt if domain is provided, otherwise use default
                    if self.domain:
                        prompt = create_domain_prompt(self.classnames[output['label']], self.domain)
                    else:
                        prompt = f"a photo of a {self.classnames[output['label']]}"
                    return output['img'], self.tokenizer(prompt).squeeze()
            else:
                return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

class CombinedDataset(DatasetBase):
    template = ["a photo of a {}."]
    def __init__(
            self, datasets: list[DatasetBase],
            remap_labels: bool = True,
            ood_datasets_inds=None,
            ood_prompt: str = None
    ):
        if ood_datasets_inds is None:
            ood_datasets_inds = []

        self.datasets = datasets
        self.ood_datasets_inds = ood_datasets_inds
        self.ood_prompt = ood_prompt
        if len(ood_datasets_inds) > 0:
            assert ood_prompt is not None

        train_x, train_u, val, test = self._merge_splits(
            ['train_x', 'train_u', 'val', 'test'], remap_labels
        )
        super().__init__(
            train_x=train_x,
            train_u=train_u if train_u else None,
            val=val if val else None,
            test=test
        )

    def _merge_splits(self, split_names, remap_labels):
        merged = {name: [] for name in split_names}
        label_offset = 0
        print(self.datasets)

        d_idx = 0
        for idx, ds in enumerate(self.datasets):
            # print(f"{ds}: {d_idx}")
            n_cls = ds.num_classes
            dataset_dir = ds.dataset_dir.split("/")[-1]
            domain = dir_name2domain_name[dataset_dir]
            is_ood = ( idx in self.ood_datasets_inds )
            if not is_ood:
                self.domains.append(domain)
            if idx in self.ood_datasets_inds:
                self.out_domains.append(domain)
            else:
                self.in_domains.append(domain)
            self.domain2classes[domain] = ds.classnames

            for name in split_names:
                data_list = getattr(ds, name) or []
                if remap_labels:
                    new_list = [
                        Datum(
                            impath=d.impath,
                            label=-1 if is_ood else (d.label + label_offset),
                            domain=-1 if is_ood else d_idx,
                            dataset_name="other" if is_ood else domain,
                            real_dataset_name=domain,
                            classname=self.ood_prompt if is_ood else d.classname,
                            real_classname= d.classname
                        )
                        for d in data_list
                    ]
                else:
                    if is_ood:
                        new_list = [
                            Datum(
                                impath=d.impath,
                                label=-1 if is_ood else d.label,
                                domain=-1 if is_ood else d_idx,
                                dataset_name="other" if is_ood else domain,
                                real_dataset_name=domain,
                                classname=self.ood_prompt if is_ood else d.classname,
                                real_classname=d.classname
                            )
                            for d in data_list
                        ]
                    else:
                        new_list = list(data_list)
                merged[name].extend(new_list)

            if idx in self.ood_datasets_inds:
                continue
            else:
                if remap_labels:
                    label_offset += n_cls
                d_idx += 1

        return (merged['train_x'],
                merged['train_u'],
                merged['val'],
                merged['test'])

    def create_cross_labels_from_selected_classnames(self, selected_classnames: list):
        domain_mapping = {}

        if selected_classnames and isinstance(selected_classnames[0], (tuple, list)):
            selected_keys = [(d, c) for d, c in selected_classnames]
            key_to_label = {key: idx for idx, key in enumerate(selected_keys)}

            def _get_domain_key(datnum: Datum):
                domain = datnum.dataset_name
                if not domain or domain == "other":
                    domain = getattr(datnum, "real_dataset_name", None) or domain
                return (domain, datnum.classname)

            def filter(datnum_list: list[Datum]) -> list[Datum]:
                rslt = []
                for datnum in datnum_list:
                    key = _get_domain_key(datnum)
                    if key in key_to_label:
                        datnum._label = key_to_label[key]
                        rslt.append(datnum)
                        domain_mapping[key] = datnum.domain
                return rslt
        else:
            selected_keys = list(selected_classnames)
            key_to_label = {key: idx for idx, key in enumerate(selected_keys)}

            def filter(datnum_list: list[Datum]) -> list[Datum]:
                rslt = []
                for datnum in datnum_list:
                    if datnum.classname in key_to_label:
                        datnum._label = key_to_label[datnum.classname]
                        rslt.append(datnum)
                        domain_mapping[datnum.classname] = datnum.domain
                return rslt

        train_x = filter(self.train_x) if self.train_x is not None else None
        train_u = filter(self.train_u) if self.train_u is not None else None
        val = filter(self.val) if self.val is not None else None
        test = filter(self.test) if self.test is not None else None

        selected_domain_targets = []
        for key in selected_keys:
            selected_domain_targets.append(domain_mapping[key])

        return DatasetBase(train_x, train_u, val, test), selected_domain_targets


def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
    return_domain=False,
    return_domain_name=False,
    tokenizer=None,
    classnames=None,
    templates=None,
    num_workers=8
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "drop_last": False,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 4,
                "multiprocessing_context": "fork",
            }
        )

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(
            data_source, input_size=input_size, transform=tfm, is_train=is_train, templates=templates,
            return_domain=return_domain, return_domain_name=return_domain_name, tokenizer=tokenizer, classnames=classnames,
        ),
        **loader_kwargs,
    )
    assert len(data_loader) > 0

    return data_loader

import math
def subsample_classes(*args, subsample="all", all_labels=None):
    """Divide classes into two groups. The first group
    represents base classes while the second group represents
    new classes.

    Args:
        args: a list of datasets, e.g. train, val and test.
        subsample (str): what classes to subsample.
    """
    assert subsample in ["all", "base", "new"]

    # print(f"**********************************{subsample}**********************************")

    if subsample == "all":
        return args
    
    dataset = args[0]
    # print(f"**********************************{dataset}**********************************")
    if all_labels is None:
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
    else:
        labels = all_labels
    n = len(labels)
    # Divide classes into two halves
    m = math.ceil(n / 2)

    print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
    if subsample == "base":
        selected = labels[:m]  # take the first half
    else:
        selected = labels[m:]  # take the second half
    relabeler = {y: y_new for y_new, y in enumerate(selected)}

    # print(f"selected: {selected}")
    
    output = []
    for dataset in args:
        dataset_new = []
        for item in dataset:
            if item.label not in selected:
                continue
            item_new = Datum(
                impath=item.impath,
                label=relabeler[item.label],
                classname=item.classname
            )
            dataset_new.append(item_new)
        output.append(dataset_new)
    
    # print(output)
    return output

def create_benchmark(data_sources: list[DatasetBase], firstk_cnt):
    selected_classnames = []
    for data_source in data_sources:
        selected_classnames += data_source.classnames[:firstk_cnt]
    combine_ds = CombinedDataset(
        datasets=data_sources,
    )
    final_ds, selected_domain_targets = combine_ds.create_cross_labels_from_selected_classnames(
        selected_classnames
    )
    return final_ds, selected_domain_targets
