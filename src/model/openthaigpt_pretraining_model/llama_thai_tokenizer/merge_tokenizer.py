from transformers import LlamaTokenizer  # type: ignore
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model  # type: ignore
import sentencepiece as spm  # type: ignore
import os

LLAMA_TOKENIZER_DIR = "tokenizer.model"
THAI_SP_MODEL_DIR = "sentencepiece.bpe.model"


llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR)
thai_sp_model = spm.SentencePieceProcessor()
thai_sp_model.Load(THAI_SP_MODEL_DIR)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
thai_spm = sp_pb2_model.ModelProto()
thai_spm.ParseFromString(thai_sp_model.serialized_model_proto())

llama_spm_tokens = {p.piece for p in llama_spm.pieces}
print(f"Before: {len(llama_spm_tokens)}")

for p in thai_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 1.0
        llama_spm.pieces.append(new_p)

merged_pieces = list(llama_spm.pieces)
merged_pieces.sort(key=lambda p: p.score, reverse=True)
llama_spm.ClearField("pieces")
llama_spm.pieces.extend(merged_pieces)

llama_spm_tokens2 = {p.piece for p in llama_spm.pieces}
print(f"After: {len(llama_spm_tokens2)}")

output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/english_thai_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/english_thai_llama.model")
tokenizer.save_pretrained(output_hf_dir)
