"""Runs installed model pipelines; does not download models automatically."""
import argparse, os, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['instantid','text-image'],required=True);p.add_argument('--photo');p.add_argument('--root');p.add_argument('--output',required=True);p.add_argument('--prompt',required=True);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
import torch
from diffusers import AutoPipelineForText2Image, ControlNetModel
from PIL import Image
steps=int(os.environ.get('MORPH_STEPS','30'));model=os.environ['SDXL_MODEL'];device=os.environ.get('MORPH_DEVICE','cpu');dtype=torch.float16 if device.startswith('cuda') else torch.float32
generator=torch.Generator(device=device).manual_seed(a.seed)
if a.mode=='text-image':
 pipe=AutoPipelineForText2Image.from_pretrained(model,torch_dtype=dtype,local_files_only=True).to(device)
 image=pipe(prompt=a.prompt,num_inference_steps=steps,generator=generator).images[0]
else:
 import cv2,numpy as np
 from insightface.app import FaceAnalysis
 sys.path.insert(0,a.root)
 from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline,draw_kps
 from infer_img2img import resize_img
 root=Path(a.root);face=resize_img(Image.open(a.photo).convert('RGB'))
 analysis=FaceAnalysis(name='antelopev2',root=str(root),providers=['CUDAExecutionProvider','CPUExecutionProvider'] if device.startswith('cuda') else ['CPUExecutionProvider'])
 analysis.prepare(ctx_id=0 if device.startswith('cuda') else -1,det_size=(640,640));faces=analysis.get(cv2.cvtColor(np.array(face),cv2.COLOR_RGB2BGR))
 if len(faces)!=1:raise ValueError('Identity generation requires one visible face.')
 control=ControlNetModel.from_pretrained(str(root/'checkpoints/ControlNetModel'),torch_dtype=dtype,local_files_only=True)
 pipe=StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained(model,controlnet=control,torch_dtype=dtype,local_files_only=True).to(device)
 pipe.load_ip_adapter_instantid(str(root/'checkpoints/ip-adapter.bin'))
 image=pipe(prompt=a.prompt,image=face,image_embeds=faces[0]['embedding'],control_image=draw_kps(face,faces[0]['kps']),controlnet_conditioning_scale=.8,ip_adapter_scale=.8,num_inference_steps=steps,guidance_scale=5,strength=.85,generator=generator).images[0]
image.save(a.output)
