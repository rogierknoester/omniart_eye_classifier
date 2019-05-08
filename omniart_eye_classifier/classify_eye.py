from typing import Tuple
from torchvision.transforms import transforms

import torch
from PIL import Image
import os
from .classifier_model import Classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = Classifier()
state_path = os.path.join(os.path.dirname(__file__), 'classifier_model_state.pth')
classifier.load_state_dict(torch.load(state_path))

# Mark as evaluation
classifier.eval()
classifier.to(device)

# Model is trained on 224 by 224 images, resize anything to that
transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# The classes the model can predict
classes = ('amber', 'blue', 'brown', 'gray', 'grayscale', 'green', 'hazel', 'irisless', 'negative', 'red')


def classify_eye(eye_image: Image) -> Tuple[str, float]:
    """
    Classify the eye colour in an image of an eye
    The classes that are used are: amber, blue, brown, gray, grayscale, green, hazel, and red

    Returns the class as a string and the confidence of the prediction
    """
    tensor = transforms(eye_image).unsqueeze(0).to(device)
    class_outputs = classifier(tensor)
    class_prediction = torch.argmax(class_outputs).item()

    confidence = torch.nn.functional.softmax(class_outputs.squeeze(), dim=0).data.cpu().numpy()

    return classes[class_prediction], confidence[class_prediction]
