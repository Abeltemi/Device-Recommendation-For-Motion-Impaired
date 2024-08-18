from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
from glob import glob

######################## Dependencies #########################
import torch
from torch import nn
from pytorchvideo.models.hub import slowfast_r50
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    UniformTemporalSubsample, 
    ShortSideScale,
    ApplyTransformToKey,
    UniformCropVideo
)
from torchvision.transforms import Compose, Lambda

######################## Dependencies ends #########################


# Set up Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

ALLOWED_VIDEO_FORMATS = {'mp4', 'avi'}

# Load model
device = 'cpu'
model = slowfast_r50(pretrained=False)
model.blocks[-1].proj = nn.Linear(2304, 2)
model.load_state_dict(torch.load('model/best_model.pth', map_location=torch.device(device)))
# model.to(device)
model.eval()

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10

class PackPathway(torch.nn.Module):
    def init(self):
        super().init()
    
    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1]-1, frames.shape[1] // slowfast_alpha).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transformer = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x/255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=side_size),
        CenterCropVideo(crop_size),
        PackPathway()
    ])
)

num_clips = 10
clip_duration = (num_frames * sampling_rate) /frames_per_second

video_path = ".mp4"
start_sec = 0
end_sec = start_sec + clip_duration


def devices():
    device_images = glob('./static/images/*.jpeg')
    device_images += glob('./static/images/*.png')
    device_images += glob('./static/images/*.jpg')

    devices = []
    device_and_descriptions = {
        "lower limb orthoses": "Orthosis that encompasses the knee and ankle joints and the foot.",
        "walkers": "Frame that helps a person to maintain stability and balance while walking or standing, with either four tips (ferrules) or two tips and two castors",
        "crutch": "Devices providing support when walking that have a horizontal padded support that is placed against the upper body next to the armpit",
        "walking stick": "Device providing support when walking that has a single shaft that branches into three or four shaft, each of which ends with a non-slip tip (ferrule)",
        "spinal orthoses":"Orthosis that encompasses the whole or part of the thoracic, lumbar and sacro-iliac regions of the truck.",
        "upper limb orthoses": "Used to stabilize (immobilize) the wrist and hand in the desired position to rest the joint, tendons, ligaments or maintain a certain bone alignment.",
        "walker_Row": "Frame that help a person to maintain stability and balance while walking, that has hand grips and three or more wheels (with or without a platform)",
        "wheelchairs": "Intended to be self-propelled by the users by pushing rims or wheels. Can be used indoor/outdoor and on various types of terrain"

    }
    for device_image in device_images:
        device_name = device_image.split('.')[1].split('\\')[-1]
        filename = device_image.split("\\")[-1]
        devices.append({
            "filename": filename,
            "name": device_name,
            "description": device_and_descriptions[device_name],
        })
    
    return devices

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_FORMATS

def process_and_predict(video_path):
    video = EncodedVideo.from_path(video_path)

    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    video_data = transformer(video_data)

    raw_inputs = video_data['video']
    inputs = [i[None, ...] for i in raw_inputs]

    with torch.no_grad():
        preds = model(inputs)
    _, prediction = torch.max(preds, 1)
    # post_act = nn.Softmax(dim=1)
    # print(f"post_act softmax: {post_act}")
    # preds = post_act(preds)
    # print(preds)
    print(prediction)
    idx_to_class = {0: 'abnormal', 1: 'normal'}
    predicted_label = idx_to_class[prediction.item()]
    print(predicted_label)
    return predicted_label

@app.route('/')
def index():
    return render_template('index.html')

# Upload video route
@app.route('/upload', methods=['POST'])
def upload_video_file():
    print(request.files)
    devices_images = devices()
    if 'file' not in request.files:
        print("No file found...")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print("filename is empty")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # prediction
        result = process_and_predict(filepath)
        result = str(result).title()
        os.remove(filepath) # this cleans up the saved file

        return render_template('result.html', result=result, devices=devices_images)
    else:
        print("Cannot perform request!")
        return redirect(request.url)
    
# Main
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)