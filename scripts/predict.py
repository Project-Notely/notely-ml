import argparse
import torch
import matplotlib.pyplot as plt
from models.cnn import EMNISTCNN
from utils.preprocess import ImagePreprocessor, EMNISTPreprocessor

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to model to use')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image to predict')
    
    args = parser.parse_args()
    
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EMNISTCNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f'Model loaded from {args.model_path}')
    
    # predict image
    preprocessor = ImagePreprocessor()
    image = preprocessor.load_image(image_path=args.image_path)
    image = image.to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        
    emnist_preprocessor = EMNISTPreprocessor()
    mapping = emnist_preprocessor.mapping
    print(f'Predicted String: {mapping[str(predicted.item())]}')
    
    # display image
    image = image.squeeze(0).squeeze(0).cpu().numpy()
    plt.imshow(image, cmap='gray')
    plt.show()
    
    
if __name__ == '__main__':
    main()
    
    