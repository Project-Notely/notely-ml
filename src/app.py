from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from models import HandwritingRecognitionModel
from models import Generator
from utils import preprocess_image, combine_images

app = FastAPI()

# Load pre-trained models
recognition_model = HandwritingRecognitionModel(num_classes=100)  # Adjust num_classes
recognition_model.load_state_dict(torch.load("models/recognition_model.pth"))
recognition_model.eval()

generator_model = Generator()
generator_model.load_state_dict(torch.load("models/generator_model.pth"))
generator_model.eval()


@app.get("/")
async def root():
    return {"message": "Handwriting recognition and generation API"}


# Handwriting recognition route
@app.post("/recognize")
async def recognize_handwriting(file: UploadFile = File(...)):
    try:
        # Load and preprocess the image
        image = Image.open(file.file).convert("L")
        processed_image = preprocess_image(image)

        # Perform prediction
        with torch.no_grad():
            output = recognition_model(processed_image.unsqueeze(0))
            prediction = torch.argmax(output, dim=2).squeeze().tolist()

        # Convert prediction to characters
        recognized_text = "".join([decode_character(p) for p in prediction if p != 0])
        return {"recognized_text": recognized_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Fine-tuning route
@app.post("/fine-tune")
async def fine_tune(user_samples: list[UploadFile] = File(...)):
    try:
        # Load user images and preprocess
        user_data = []
        for file in user_samples:
            image = Image.open(file.file).convert("L")
            processed_image = preprocess_image(image)
            user_data.append(processed_image)

        # Convert to DataLoader
        user_dataloader = create_user_dataloader(user_data)

        # Fine-tune the model
        optimizer = torch.optim.Adam(recognition_model.parameters(), lr=0.0001)
        fine_tune_model(recognition_model, user_dataloader, optimizer, criterion)

        # Save fine-tuned model
        torch.save(recognition_model.state_dict(), "models/fine_tuned_model.pth")
        return {"message": "Model fine-tuned successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Handwriting generation route
@app.post("/generate")
async def generate_handwriting(text: str):
    try:
        # Generate handwriting from text
        noise = torch.randn(len(text), 100)  # Generate noise for each character
        generated_images = generator_model(noise)

        # Save generated images as a single image
        result_image = combine_images(generated_images)
        result_image.save("generated_handwriting.png")

        return {
            "message": "Handwriting generated successfully",
            "file": "generated_handwriting.png",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
