from nemo.collections.asr.models import ASRModel
import torch
import numpy as np


class BatchedTranscriptionEngine:

	def __init__(self):
		model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

		if torch.cuda.is_available():
			print("is cuda available inhere woow ")
			self.model = model.to(device="cuda")
			print(f"Allocated : {torch.cuda.memory_allocated()/1024**3:.2f} GB")
			print(f"Reserved  : {torch.cuda.memory_reserved()/1024**3:.2f} GB")
		else:
			self.model = model

		self.model.eval()

		# Grab the decoding_computer reference so we can toggle CUDA graphs
		# on/off per call. CUDA Graphs require fixed tensor shapes to replay
		# correctly. transcribe_file gets variable-length files → shapes change
		# every call → graph replays to wrong/stale memory pointers → crash.
		# transcribe_stream always sends fixed 5-second chunks → same shape
		# every time → CUDA graphs are safe and give full speed benefit.
		self._decoding_computer = None
		if hasattr(self.model, "decoding") and hasattr(self.model.decoding, "decoding"):
			dc = self.model.decoding.decoding
			if hasattr(dc, "decoding_computer"):
				computer = dc.decoding_computer
				if hasattr(computer, "disable_cuda_graphs") and hasattr(computer, "enable_cuda_graphs"):
					self._decoding_computer = computer
					print("[prkt] CUDA graph toggle available — graphs ON for streaming, OFF for file jobs")

	def transcribe_file(self, path: str):
		# File audio is variable-length → tensor shapes differ each call.
		# CUDA Graphs capture static shapes, so replaying against a different
		# shape hits stale/freed memory → cudaErrorIllegalAddress on 2nd+ call.
		# Disable graphs for this call only, then re-enable for streaming.
		if self._decoding_computer:
			self._decoding_computer.disable_cuda_graphs()
		try:
			with torch.inference_mode():
				results = self.model.transcribe(
					audio=[path],
					batch_size=1,
					timestamps=True,
					num_workers=0,
				)
			return results[0]
		finally:
			# Always re-enable so streaming gets CUDA graph speed back.
			if self._decoding_computer:
				self._decoding_computer.enable_cuda_graphs()

	def transcribe_stream(self, audios, batch_size: int):
		# Fixed 5-second chunks → always the same shape → CUDA graphs are
		# safe here and give the full latency/throughput benefit.
		with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			return self.model.transcribe(audios, batch_size=batch_size)