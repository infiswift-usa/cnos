import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import cv2
import numpy as np
from matplotlib import pyplot as plt

from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
from hydra.utils import instantiate
import argparse
import glob
from .src.utils.bbox_utils import CropResizePad
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
from .src.model.utils import Detections, convert_npz_to_json
from .src.model.loss import Similarity
from .src.utils.inout import save_json_bop23
from .src.poses.pyrender import render
import cv2
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import subprocess
import pyrender
from .src.utils.trimesh_utils import as_mesh
from .src.utils.trimesh_utils import get_obj_diameter
import trimesh
import importlib.resources as resources
from cnos import __name__ as pkg_name

class Segmenter:
    def __init__(self, model = 'cnos'):
        self.load_model(model = model)

    def to_grayscale(self):
        # Convert the image to grayscale
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_threshold(self):
        # Apply a binary threshold to the image to create a binary image
        _, self.binary_image = cv2.threshold(self.gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    def load_model(self, model, stability_score_thresh=0.97):
        with initialize(version_base=None, config_path="./configs"):
            cfg = compose(config_name='run_inference.yaml', overrides=[f"model={model}"])
        cfg_segmentor = cfg.model.segmentor_model

        print (cfg.model)
        if "fast_sam" in cfg_segmentor._target_:
            logging.info("Using FastSAM, ignore stability_score_thresh!")
        else:
            cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
        metric = Similarity()
        logging.info("Initializing model")
        model = instantiate(cfg.model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.descriptor_model.model = model.descriptor_model.model.to(device)
        model.descriptor_model.model.device = device
        # if there is predictor in the model, move it to device
        if hasattr(model.segmentor_model, "predictor"):
            model.segmentor_model.predictor.model = (
                model.segmentor_model.predictor.model.to(device)
            )
        else:
            model.segmentor_model.model.setup_model(device=device, verbose=True)
        logging.info(f"Moving models to {device} done!")

        self.model = model

    def load_templates(self, template_dir):
        self.template_dir = template_dir
        logging.info("Initializing template")
        template_paths = glob.glob(f"{template_dir}/*.png")
        boxes, templates = [], []
        for path in template_paths:
            image = Image.open(path)
            boxes.append(image.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        if torch.cuda.is_available():
            templates = proposal_processor(images=templates, boxes=boxes).cuda()
        else:
            templates = proposal_processor(images=templates, boxes=boxes)

        if not os.path.exists(f"{template_dir}/cnos_results"):
            # If the directory does not exist, create it
            os.makedirs(f"{template_dir}/cnos_results")
            
        save_image(templates, f"{template_dir}/cnos_results/templates.png", nrow=7)
        self.ref_feats = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    )
        logging.info(f"Ref feats: {self.ref_feats.shape}")

    def run_inference(self, rgb, num_max_dets=1, conf_threshold=0.4):
        # run inference
        rgb = Image.open(rgb).convert("RGB")
        detections = self.model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        decriptors = self.model.descriptor_model.forward(np.array(rgb), detections)

        print (decriptors.shape)

        metric = Similarity()
         # get scores per proposal
        scores = metric(decriptors[:, None, :], self.ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(
            score_per_detection, dim=-1
        )

        # get top-k detections
        scores, index = torch.topk(score_per_detection, k=num_max_dets, dim=-1)
        detections.filter(index)
        
        # keep only detections with score > conf_threshold
        detections.filter(scores>conf_threshold)
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
            
        detections.to_numpy()
        print (detections)
        save_path = f"{self.template_dir}/cnos_results/detection"
        detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections)
        vis_img = self.visualize(rgb, detections, self.template_dir)
        vis_img.save(f"{self.template_dir}/cnos_results/vis.png")

    def visualize(self, rgb, detections, save_path):
        img = rgb.copy()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # img = (255*img).astype(np.uint8)
        colors = distinctipy.get_colors(len(detections))
        alpha = 0.33

        for mask_idx, det in enumerate(detections):
            mask = rle_to_mask(det["segmentation"])
            edge = canny(mask)
            edge = binary_dilation(edge, np.ones((2, 2)))
            obj_id = det["category_id"]
            temp_id = obj_id - 1

            r = int(255*colors[temp_id][0])
            g = int(255*colors[temp_id][1])
            b = int(255*colors[temp_id][2])
            img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
            img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
            img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
            img[edge, :] = 255

            mask_img = Image.fromarray(np.uint8(mask*255))
            mask_img.save(f"{save_path}/cnos_results/mask_{mask_idx}.png")
        
        
        img = Image.fromarray(np.uint8(img))
        img.save(f'{save_path}/cnos_results/vis.png')
        prediction = Image.open(f'{save_path}/cnos_results/vis.png')
        
        # concat side by side in PIL
        img = np.array(img)
        concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
        concat.paste(rgb, (0, 0))
        concat.paste(prediction, (img.shape[1], 0))
        return concat
    
    def load_detections(self, rgb, detections):
        pass
        

    def segment_screw(self):
        # Find contours in the binary image
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask to draw the segmented screw
        self.segmented_image = np.zeros_like(self.gray_image)
        
        # Draw the contours on the mask
        cv2.drawContours(self.segmented_image, contours, -1, (255), thickness=cv2.FILLED)

    def display_images(self):
        # Display the original, grayscale, binary, and segmented images
        images = [self.image, self.gray_image, self.binary_image, self.segmented_image]
        titles = ['Original Image', 'Grayscale Image', 'Binary Image', 'Segmented Screw']

        for i in range(4):
            plt.subplot(2, 2, i+1)
            if i == 0:
                plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    def render_templates(self, cad_path='./data/Chrome_screw.obj', output_dir='./data/templates', obj_pose=None,  lighting_intensity=0.5, radius=60):
        # Set environment variables

        disable_output = False
        light_itensity = lighting_intensity

        if obj_pose is None:
            with resources.path('cnos.src.poses.predefined_poses', 'obj_poses_level0.npy') as path:
                obj_pose = str(path)
            print (f'Loading pose file from cnos - {obj_pose}')
        poses = np.load(obj_pose)
        # we can increase high energy for lightning but it's simpler to change just scale of the object to meter
        # poses[:, :3, :3] = poses[:, :3, :3] / 1000.0
        poses[:, :3, 3] = poses[:, :3, 3] / 1000.0
        if radius != 1:
            poses[:, :3, 3] = poses[:, :3, 3] * radius
        intrinsic = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]]
        )
        img_size = [480, 640]
        is_tless = False

        # load mesh to meter
        mesh = trimesh.load_mesh(cad_path)
        diameter = get_obj_diameter(mesh)
        if diameter > 100: # object is in mm
            mesh.apply_scale(0.001)
        if is_tless:
            # setting uniform colors for mesh
            color = 0.4
            mesh.visual.face_colors = np.ones((len(mesh.faces), 3)) * color
            mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 3)) * color
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            mesh = pyrender.Mesh.from_trimesh(as_mesh(mesh))
        os.makedirs(output_dir, exist_ok=True)
        render(
            output_dir=output_dir,
            mesh=mesh,
            obj_poses=poses,
            intrinsic=intrinsic,
            img_size=(480, 640),
            light_itensity=light_itensity,
        )




if __name__ == '__main__':

    # Inputs

    # 1. Path to get or generate Templates - template_dir : str
    # 2. CAD PATH 
    # 3. Input RGB Image Path
    # 4. Max Detections : default 10
    # 5. Confidence Threshold : 0.4

    # Outputs

    # 1. Templates saved in template_dir
    # 2. Model Results in {path to template_dir}/cnos_results directiory
    #    - detections.json --> Segmentation of detected objects
    #    - vis.png --> Segmentation Visualization

    template_dir ='./data/templates'

    # Render Templates (Only needed once per object)
    render_templates(cad_path='./data/Chrome_screw.obj', output_dir=template_dir)

    segmenter = Segmenter(template_dir = template_dir)
    segmenter.run_inference(rgb = './data/screws_d405.png', num_max_dets=10, conf_threshold=0.4)