import timm

model = timm.create_model("hf_hub:paige-ai/Virchow", pretrained=True)
print(model)
