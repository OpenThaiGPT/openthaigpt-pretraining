from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm


def merge(llama_tokenizer_dir, thai_sp_model_dir, get_spm_tokenizer=False):
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    thai_sp_model = spm.SentencePieceProcessor()
    thai_sp_model.Load(thai_sp_model_dir)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    thai_spm = sp_pb2_model.ModelProto()
    thai_spm.ParseFromString(thai_sp_model.serialized_model_proto())

    llama_spm_tokens = {p.piece for p in llama_spm.pieces}

    for p in thai_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0.0
            llama_spm.pieces.append(new_p)

    if get_spm_tokenizer:
        return llama_spm

    llama_tokenizer.sp_model = spm.SentencePieceProcessor(
        model_proto=llama_spm.SerializeToString()
    )
    return llama_tokenizer
