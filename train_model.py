def run_training(concept):
    model_name = "runwayml/stable-diffusion-v1-5"
    instance_dir = "/home/mahlow/auto_tune/images"
    output_dir = "/home/mahlow/auto_tune/gen_imgs"

    command = f"accelerate launch /home/mahlow/diffusers/examples/dreambooth/train_dreambooth.py \
                --pretrained_model_name_or_path={model_name}  \
                --instance_data_dir={instance_dir} \
                --output_dir={output_dir} \
                --instance_prompt='a photo of {concept}' \
                --resolution=512 \
                --train_batch_size=1 \
                --gradient_accumulation_steps=1 \
                --learning_rate=5e-6 \
                --lr_scheduler='constant' \
                --lr_warmup_steps=0 \
                --max_train_steps=400 \
                --push_to_hub"
    try:
        subprocess.run(command, shell=True, check=True)
        print("Training completed successfully.")
        pipeline_text2image_train = AutoPipelineForText2Image.from_pretrained(output_dir, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


run_training("saci perere")