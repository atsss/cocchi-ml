import torch
from torchvision import transforms
import urllib
import requests
from PIL import Image
import urls
import csv

torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.eval()

response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

rows = []

for url in urls.all:
    url, filename = (url, "tmp.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        prediction = torch.nn.functional.softmax(output[0], dim=0)

    row = [url]

    for i in range(1000):
        if prediction[i] > 0.05:
            row.append(labels[i])
            row.append(float(prediction[i] * 100))

    rows.append(row)

with open('output.csv', 'w') as file:
    w = csv.writer(file)
    w.writerows(rows)
