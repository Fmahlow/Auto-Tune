from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
import os
import subprocess
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from diffusers import AutoPipelineForText2Image
import torch
import os
from urllib.parse import urlparse, parse_qs
import gradio as gr
import time
from transformers import CLIPTextModel
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
import shutil
from diffusers import AutoencoderKL
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline


def scrape_images(concept, num_images=10, count=0, output_folder='images'):
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    op.add_argument('--no-sandbox')
    op.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=op)

    busca = concept
    link = f"https://www.google.com/search?q={busca}&tbm=isch"
    driver.get(link)

    img = []
    link_img = []

    n_img = num_images
    count = 0
    for i in range(1, n_img+1):
        if n_img <= count:
            break
        if count % 10 == 0:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            img_element = driver.find_element(By.XPATH, f"//*[@id=\"islrg\"]/div[1]/div[{str(i)}]/a[1]/div[1]/img")
            img_link = img_element.get_attribute("src")
            if img_link is None:
                img_link = img_element.get_attribute('data-src')
                if img_link is None:
                    print("continua none")
            link_img.append(img_link)

            if "base64" in img_link:
                img_link = img_link.split(",")[1]
                img_data = base64.b64decode(img_link)
                img_pil = Image.open(BytesIO(img_data))
                img.append(img_pil)
                count += 1
            else:
                img_response = requests.get(img_link)
                img_pil = Image.open(BytesIO(img_response.content))
                img.append(img_pil)
                count += 1
        except:
            for k in range(1, n_img-i+1):
                try:
                    if count % 10 == 0:
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    if n_img <= count:
                        break
                    img_element = driver.find_element(By.XPATH,f"//*[@id=\"islrg\"]/div[1]/div[{str(i)}]/div[{str(k)}]/a[1]/div[1]/img")
                    img_link = img_element.get_attribute("src")
                    link_img.append(img_link)
                    if img_link is None:
                        img_link = img_element.get_attribute('data-src')
                        if img_link is None:
                            print("continua none")
                    if "base64" in img_link:
                        img_link = img_link.split(",")[1]
                        img_data = base64.b64decode(img_link)
                        img_pil = Image.open(BytesIO(img_data))
                        img.append(img_pil)
                        count += 1
                    else:
                        img_response = requests.get(img_link)
                        img_pil = Image.open(BytesIO(img_response.content))
                        img.append(img_pil)
                        count += 1
                except:
                    break

    print(len(img))
    count = 0
    for i in img:
        if i.mode != 'RGB':
            i = i.convert('RGB')
        count += 1
        nome_arquivo = str(concept)+str(count)+".jpeg"
        i.save(os.path.join(str(output_folder), nome_arquivo))


def get_file_content(file):
    images_pil = []
    for i in file:
        imgs_pil = Image.open(i)
        images_pil.append(imgs_pil)
    return images_pil


def update_files1(input_images):
    return gr.update(root_dir="/home/mahlow/")

def update_files2(input_images):
    return gr.update(root_dir="/home/mahlow/auto_tune/images")

def delete_unselected_items(selected_items, root_dir):
    # List all files in the root directory
    all_files = os.listdir(root_dir)

    # Iterate through all files in the root directory
    for file_name in all_files:
        file_path = os.path.join(root_dir, file_name)

        # Check if the file is not selected and it's a file (not a directory)
        if file_name not in selected_items and os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

def update_count(count, num_imgs):
    return gr.update(value=num_imgs+count)

def return_count(count):
    return count


def run_training(instance_prompt):
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    instance_dir = "/home/mahlow/auto_tune/images"
    output_dir = f"/home/mahlow/auto_tune/gen_imgs/{str(time.time()).replace('.', '')}"
    vae_path = "madebyollin/sdxl-vae-fp16-fix"
    os.mkdir(output_dir)
                
                
    command = f"accelerate launch /home/mahlow/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \
                --pretrained_model_name_or_path={model_name}  \
                --mixed_precision='fp16' \
                --instance_data_dir={instance_dir} \
                --pretrained_vae_model_name_or_path={vae_path} \
                --output_dir={output_dir} \
                --instance_prompt='{instance_prompt}' \
                --resolution=1024 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=4 \
                --learning_rate=1e-4 \
                --lr_scheduler='constant' \
                --lr_warmup_steps=0 \
                --push_to_hub \
                --enable_xformers_memory_efficient_attention \
                --gradient_checkpointing \
                --use_8bit_adam \
                --report_to='wandb' \
                --validation_prompt='A photo of @fmahlow man' \
                --validation_epochs=25 \
                --max_train_steps=3000"

    try:
        subprocess.run(command, shell=True, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


def gerar_imagem(prompt):
    unet = UNet2DConditionModel.from_pretrained(
    "latent-consistency/lcm-sdxl",
    torch_dtype=torch.float16,
    variant="fp16",
    )
    pipeline_text2image = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16
    ).to("cuda")
    pipeline_text2image.scheduler = LCMScheduler.from_config(pipeline_text2image.scheduler.config)
    return pipeline_text2image(prompt=prompt, num_inference_steps=4, guidance_scale=8.0).images[0]       

def gerar_imagem_train(prompt):
    # lora_model_id = "FelipeMahlow/17122680270484388"
    # card = RepoCard.load(lora_model_id)
    # base_model_id = card.data.to_dict()["base_model"]
    # pipeline = DiffusionPipeline.from_pretrained(
    #     base_model_id).to("cuda")
    # pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    # pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    # pipeline.load_lora_weights(lora_model_id, adapter_name="finetuning")
    # pipeline.set_adapters(["lcm", "finetuning"], adapter_weights=[1.0, 1.0])

    
    return pipeline(prompt, num_inference_steps=4, guidance_scale=1).images[0]
    
def mover_arquivos(pasta_temporaria, pasta_permanente = "images/"):
    if not os.path.exists(pasta_permanente):
        os.makedirs(pasta_permanente)

    aux = 0
    for arquivo in pasta_temporaria:
        destino = os.path.join(pasta_permanente, f'imagem_{aux}.jpg')
        shutil.move(arquivo, destino)
        aux += 1
        print(f"Arquivo '{arquivo}' movido para '{pasta_permanente}'.")

output_folder = "images"
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

lora_model_id = "FelipeMahlow/17122680270484388"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]
pipeline = DiffusionPipeline.from_pretrained(
    base_model_id).to("cuda")
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
pipeline.load_lora_weights(lora_model_id, adapter_name="finetuning")
pipeline.set_adapters(["lcm", "finetuning"], adapter_weights=[1.0, 1.0])

# Limpando a pasta de saída antes de começar
for file_name in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file_name)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        # Se você quiser excluir diretórios também, descomente a linha abaixo
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Falha ao excluir {file_path}: {str(e)}")

with gr.Blocks() as demo:
    count = gr.Number(value=0, visible=False)
    gr.Markdown("Pipeline para Finetuning automático do DreamBooth")
    with gr.Row():
        with gr.Column():
            conceito = gr.Textbox(label="Digite o conceito para realizar o Fine-Tuning", placeholder="Digite aqui o conceito")
            num_imgs = gr.Slider(1, 100, value=10, step=1, label="Número de imagens para webscrapping", info="Escolha entre 10 and 100", interactive=True)
            btn = gr.Button(value="Buscar Imagens")
            file = gr.File(interactive=True, file_count='multiple')
            btn_submit_files = gr.Button(value="Submeter novos arquivos")
            files = gr.FileExplorer(interactive=True, root_dir= output_folder, height=300)#(file_types=["image"], file_count="multiple", label="Input Images", interactive=True)
        with gr.Column():
            input_images = gr.Gallery(type="filepath", interactive=False)
            btn.click(scrape_images, inputs=[conceito, num_imgs, count]).then(update_files1, input_images, files).then(update_files2, input_images, files).then(update_count, [count, num_imgs], count)
            btn_submit_files.click(mover_arquivos, file).then(update_files1, input_images, files).then(update_files2, input_images, files).then(update_count, [count, num_imgs], count)
            instance_prompt = gr.Textbox(label="Digite o Instance Prompt para o treinamento", placeholder="Digite aqui o conceito")
            btn_train_model = gr.Button(value="Treinar o Modelo")
            btn_train_model.click(run_training, instance_prompt)
            out = gr.Textbox(label="Digite o prompt da imagem que quer gerar", placeholder="Prompt da imagem")
            btn_test_model = gr.Button(value="Testar o Modelo")
            with gr.Row():
                with gr.Column():
                    final_img_inicial = gr.Image(label="Imagem Antes do Treinamento", width=400, interactive=False)
                with gr.Column():
                    final_img_treinada = gr.Image(label="Imagem Após Treinamento", width=400, interactive=False)
            #btn_test_model.click(gerar_imagem, out, final_img_inicial).then(gerar_imagem_train, out, final_img_treinada)
            btn_test_model.click(gerar_imagem_train, out, final_img_treinada)
    files.change(get_file_content, files, input_images)

demo.launch(debug=True, share=True)