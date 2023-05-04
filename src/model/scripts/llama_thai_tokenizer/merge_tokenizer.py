from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import os
from openthaigpt_pretraining_model.llama_thai_tokenizer.constants import (
    LLAMA_TOKENIZER_DIR,
    THAI_SP_MODEL_DIR,
    OUTPUT_SP_DIR,
    OUTPUT_HF_DIR,
)


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
        new_p.score = 0.0
        llama_spm.pieces.append(new_p)

llama_spm_tokens2 = {p.piece for p in llama_spm.pieces}
print(f"After: {len(llama_spm_tokens2)}")

SAVE_TOKENIZER_NAME = "english_thai_llama.model"
os.makedirs(OUTPUT_SP_DIR, exist_ok=True)
with open(OUTPUT_SP_DIR + "/" + SAVE_TOKENIZER_NAME, "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=OUTPUT_SP_DIR + "/" + SAVE_TOKENIZER_NAME)
tokenizer.save_pretrained(OUTPUT_HF_DIR)
