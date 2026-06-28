shared_prompts = [
    "a photo of a {}",
    "a photo of the {}",
    "a picture of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a blurry photo of a {}",
    "a black and white photo of a {}",
    "a low resolution photo of a {}",
    "a good photo of a {}",
    "a bright photo of a {}",
    "a dark photo of a {}",
    "a rendering of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a bad photo of a {}",
    "a weird photo of a {}"
]


caltech101_prompts = [
    "an image of a {}",
    "a detailed photo of a {}",
    "a high-quality image of a {}",
    "a professional photo of a {}",
    "a studio shot of a {}"
]

eurosat_prompts = [
    "a satellite photo of {}",
    "a satellite image of {}",
    "an aerial view of {}",
    "a top-down view of {}",
    "a high-resolution satellite photo of {}"
]

stanford_cars_prompts = [
    "a photo of a {} car",
    "a picture of the {} automobile",
    "a photo of a {} vehicle",
    "a photo of a {} car on the road",
    "a side view of a {} car"
]

aibd_cars_prompts = stanford_cars_prompts

cub200_prompts = [
    "a photo of a {} bird",
    "a picture of the {} bird species",
    "a close-up photo of a {} bird",
    "a photo of a {} bird in the wild",
    "a detailed photo of a {} bird"
]

resisc45_prompts = [
    "a remote sensing image of {}",
    "a satellite image of {}",
    "an aerial view of {}",
    "a top-down remote sensing photo of {}",
    "an overhead image of {}"
]

food101_prompts = [
    "a photo of {} food",
    "a picture of a plate of {}",
    "a close-up of {}",
    "a delicious serving of {}",
    "a dish of {} on a table"
]

oxford_pets_prompts = [
    "a cute photo of a {}",
    "a pet photo of a {}",
    "a close-up of a {}",
    "a picture of a {} indoors",
    "a photo of a {} animal"
]

oxford_flowers_prompts = [
    "a photo of a {} flower",
    "a close-up of a {} blossom",
    "a macro shot of a {} bloom",
    "a picture of a {} in the garden",
    "a colorful photo of a {} flower"
]

dtd_prompts = [
    "a photo of {} texture",
    "a close-up of {} pattern",
    "a detailed shot of {} surface",
    "a macro photo of {}",
    "a picture of a {} material"
]

sun397_prompts = [
    "a photo of a {} scene",
    "a picture of a {} environment",
    "an image of a {} location",
    "a wide shot of a {} place",
    "a photo showing {} surroundings"
]

ucf101_prompts = [
    "a photo of a person {}",
    "a picture of someone {}",
    "an image of a person performing {}",
    "a photo of people {}",
    "a still frame of someone {}"
]

fgvc_prompts = [
    "a photo of a {}",
    "a picture of a {}",
    "a close-up of a {}",
    "a side view of a {}",
    "a detailed photo of {}",
]

prompt_groups = {
    "shared": shared_prompts,
    "caltech101": caltech101_prompts,
    "eurosat": eurosat_prompts,
    "stanford_cars": stanford_cars_prompts,
    "aibd_cars": aibd_cars_prompts,
    "cub200": cub200_prompts,
    "resisc45": resisc45_prompts,
    "food101": food101_prompts,
    "oxford_pets": oxford_pets_prompts,
    "oxford_flowers": oxford_flowers_prompts,
    "dtd": dtd_prompts,
    "sun397": sun397_prompts,
    "ucf101": ucf101_prompts,
    "fgvc": fgvc_prompts
}

