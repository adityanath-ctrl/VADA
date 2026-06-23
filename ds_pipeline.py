"""
	Diarization pipeline uses the something
	like the SortFormer for performance critical 
	issue in here we are using SortFormer in here 
"""
from nemo.collections.asr.models import SortformerEncLabelModel

class DiarizationPipeLine:
	def __init__(self, variant="nvidia/diar_sortformer_4spk-v1"):
		self.model = SortformerEncLabelModel.from_pretrained(variant)
		# self.model = model.cuda()

	def process_segments(self,segments):
		output = []

		for seg in segments:
			start, end, speaker = seg.split()
			output.append({
				"start": float(start),
				"end": float(end),
				"speaker": speaker,
				"duration": float(end) - float(start)
			})

		return {"segments": output}

	def sort_segments(self, segments):
		return sorted(segments, key=lambda x: x["start"])

	def diarize(self,path: list[str]):
		segments = self.model.diarize(path)[0]
		json_ = self.process_segments(segments)
		return self.sort_segments(json_["segments"])

