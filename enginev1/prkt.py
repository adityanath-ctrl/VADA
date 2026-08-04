from fastapi import FastAPI, HTTPException
from nemo.collections.asr.models import ASRModel
from omegaconf import open_dict
import torch
import numpy as np

class BatchedTranscriptionEngine:

	def __init__(self):
		model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

		if torch.cuda.is_available():
			print("is cuda avilable inhere woow ")
			self.model = model.to(device="cuda")
			print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
			print(f"Reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")

		self.model.eval()

	def transcribe_file(self, path: str):

		with torch.inference_mode():

			results = self.model.transcribe(
				audio=[path],
				batch_size=1,
				timestamps=True,
				num_workers=0,
			)

		return results[0]

	def transcribe_stream(self, audios,batch_size: int):
		with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			return self.model.transcribe(audios, batch_size=batch_size)