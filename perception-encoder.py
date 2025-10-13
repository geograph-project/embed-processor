## !!NOTE!!
##  ...this script isnt fully functional, its a extract of code from a functional workbook, to demonstrate working with perception encode model
##  https://colab.research.google.com/drive/1dveSFylrG8EHS8P1xgKm_yD0X-JPTqaK

import sys
import os

# Geograph expects the encoded embeddings in little endian format. If using a diffente system please contact us about how to provide data
assert(sys.byteorder == 'little')

if 'google.colab' in sys.modules:
    print('Running in Colab.')
    !git clone https://github.com/facebookresearch/perception_models.git
    !pip install decord
    !pip install ftfy
    sys.path.append('./perception_models')
    os.chdir('./perception_models')
else:
    sys.path.append('../../../')
import decord

import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

perception_model_name = 'PE-Core-B16-224' # Using the model name from the example


import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

model = pe.CLIP.from_config(perception_model_name, pretrained=True)  # Downloads from HF
model = model.to(device)

preprocess = transforms.get_image_transform(model.image_size)
tokenizer = transforms.get_text_tokenizer(model.context_length)


if True:
    # data_json fetched from an API

    current_image_batch = []   # will be holding preprocessed image
    current_text_batch = []    # will be holding the tokenized text
    img_embeds = []
    txt_embeds = []

    for image in tqdm(data_json['rows']):

        img_id = image['gridimage_id']
        filename = image_dir + '/' + os.path.basename(image["fullpath"])

        if not os.path.isfile(filename):
            fullurl = data_json['prefix'] + image['fullpath']

            logging.info(f'Fetching {fullurl}')
            try:
                with requests.get(fullurl) as file:
                    file.raise_for_status()
                    with open(filename, 'wb') as fd:
                        for chunk in file.iter_content(chunk_size=2048):
                            fd.write(chunk)

        with Image.open(filename) as pil_img:
            # Attempt to preprocess the image
           img = preprocess(pil_img)
        current_image_batch.append(img)

        txt_tokens = tokenizer([image['title']]).squeeze(0) # Use the tokenizer from the Perception Encoder example and remove the extra dimension

        current_text_batch.append(txt_tokens)


    with torch.no_grad():
        start_time = time.time()
        img_embeds = model.encode_image(images_tensor).cpu().detach() # This now returns a batch of features
        image_encoding_time = time.time() - start_time

        start_time = time.time()
        txt_embeds = model.encode_text(text_tensor).cpu().detach() # This now returns a batch of features
        text_encoding_time = time.time() - start_time

    # Print timing information
    print(f"Image encoding took {image_encoding_time:.2f} seconds, text encoding took {text_encoding_time:.2f} seconds, for {len(data_json['rows'])} images.")

    # Geograph expect float32 (if running on GPU clip might produce float16)
    assert img_embeds.dtype == torch.float32, "Image tensor must be torch.float32"


    for idx, embedding in enumerate(img_embeds):
        binary_embedding = embedding.numpy().tobytes()
        #....

    for idx, embedding in enumerate(txt_embeds):
        binary_embedding = embedding.numpy().tobytes()
        #....



