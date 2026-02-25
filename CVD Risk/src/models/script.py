from transformers import AutoModel

model = AutoModel.from_pretrained("YukunZhou/RETFound_dinov2_meh")
print(model.config.hidden_size)  # should print 1024