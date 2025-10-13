import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
from PIL import Image
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,  # New import for LLM tokenizer
    BitsAndBytesConfig  # New import for quantization
)
import uvicorn

app = FastAPI()

# Your existing model loading
#clip_model = CLIPModel.from_pretrained("wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
#clip_processor = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
smallcap_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
smallcap_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# New model loading for the LLM
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
llm_model_name = "google/gemma-2b-it"  # The 2B version is smaller and easier to run
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)


class ImageRequest(BaseModel):
	image: str

class TextRequest(BaseModel):
	text: str

# New Pydantic class for LLM requests
class LLMRequest(BaseModel):
    text: str
    image: str = None  # Optional image field

# Rest of your existing classes and code...

# ... (your existing endpoints)

@app.post("/generate")
async def generate_llm_response(request: LLMRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text prompt provided.")

    try:
        full_prompt = request.text
        if request.image:
            # Decode the image and generate a caption
            image_bytes = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(image_bytes))

            # Use the existing smallcap model to describe the image
            inputs = smallcap_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                generated_ids = smallcap_model.generate(pixel_values=inputs.pixel_values, max_length=50)
            image_caption = smallcap_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Combine the image caption with the user's text prompt
            full_prompt = f"The image shows: {image_caption}. Based on the image, {request.text}"

        # Format the prompt for the Gemma model
        chat_prompt = f"<start_of_turn>user\n{full_prompt}<end_of_turn>\n<start_of_turn>model"

        # Tokenize the prompt
        inputs = llm_tokenizer(chat_prompt, return_tensors="pt")

        # Generate the response
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, max_new_tokens=100, do_sample=True)

        # Decode the output and extract the model's response
        generated_text = llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Extract the model's reply from the full conversation
        response = generated_text.split("<start_of_turn>model\n")[-1].strip()

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating LLM response: {str(e)}")

if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000)


