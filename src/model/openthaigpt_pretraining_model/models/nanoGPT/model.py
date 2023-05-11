import torch.nn.functional as F
import torch.backends.cuda as cuda
from transformers import AutoConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)  # noqa: F401

from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import (
    add_code_sample_docstrings,
    logging,
)
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from ...optimizers.lion.constants import (  # type: ignore
    ROTARY_EMB_BASE,
    ROTARY_PCT,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention as OriginalGPT2Attention,
    GPT2Block as OriginalGPT2Block,
    GPT2Model as OriginalGPT2Model,
    GPT2LMHeadModel as OriginalGPT2LMHeadModel,
)  # noqa: E501

logger = logging.get_logger(__name__)
_attn_orig = OriginalGPT2Attention._attn

_CHECKPOINT_FOR_DOC = "gpt2"
_CONFIG_FOR_DOC = "GPT2Config"


# patch GPT2Attention to use flash_sdp, disable it when doing the inference
def _attn_wrapper(self, query, key, value, attention_mask=None, head_mask=None):
    if head_mask is not None:
        raise NotImplementedError("head_mask is not implemented for flash_sdp")
    is_causal = attention_mask is None
    with cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    ):
        attn_out = F.scaled_dot_product_attention(
            query=query.half(),
            key=key.half(),
            value=value.half(),
            is_causal=is_causal,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p,
        ).float()
    return attn_out, None


class GPT2AttentionWithRotary(OriginalGPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config)

        self.rotary_pct = ROTARY_PCT
        self.rotary_emb_base = ROTARY_EMB_BASE
        self.use_rotary = config.use_rotary
        self.rotary_ndims = int(self.head_dim * self.rotary_pct)
        if self.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_ndims,
                config.max_position_embeddings,
                base=self.rotary_emb_base,
            )

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError("If class is used as cross attention")

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        ###############################
        # START USE ROTARY
        ###############################
        if self.use_rotary:
            # Compute rotary embeddings on rotary_ndims
            query_rot = query[
                ..., : self.rotary_ndims
            ]  # [batch, num_attention_heads, seq_len, head_size]
            query_pass = query[..., self.rotary_ndims :]
            key_rot = key[..., : self.rotary_ndims]
            key_pass = key[..., self.rotary_ndims :]
            # Compute token offset for rotary embeddings (when decoding)
            seq_len = key.shape[-2]
            if layer_past:
                seq_len += layer_past[0].shape[-2]
            cos, sin = self.rotary_emb(value, seq_len=seq_len)
            query, key = apply_rotary_pos_emb(
                query_rot, key_rot, cos, sin, position_ids
            )
            query = torch.cat((query, query_pass), dim=-1)
            key = torch.cat((key, key_pass), dim=-1)
        ###############################
        # END USE ROTARY
        ###############################
        if layer_past is not None:
            past_key, past_value = layer_past  # type: ignore
            key = torch.cat((past_key, key), dim=-2)  # type: ignore
            value = torch.cat((past_value, value), dim=-2)  # type: ignore

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)  # type: ignore

        return outputs  # type: ignore


class EditGPT2Block(OriginalGPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config)
        self.attn = GPT2AttentionWithRotary(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            position_ids=position_ids,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(f"If `encoder_hidden_states` are passed, {self}")
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = (
                outputs + cross_attn_outputs[2:]
            )  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class EditGPT2Model(OriginalGPT2Model):
    def __init__(self, config):
        super().__init__(config)
        ###############################
        # START USE ROTARY BY Remove original self.wpe
        ###############################
        self.use_rotary = config.use_rotary
        if not self.use_rotary:
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        else:
            del self.wpe
            print("Let's use Rotary Positional Encoding")
        ###############################
        # END USE ROTARY
        ###############################
        self.h = nn.ModuleList(
            [
                EditGPT2Block(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])  # type: ignore
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device  # type: ignore  # noqa: E501

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])  # type: ignore
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])  # type: ignore

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))  # type: ignore
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(  # type: ignore
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])  # type: ignore  # noqa: E501

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)  # type: ignore
            attention_mask = attention_mask[:, None, None, :]  # type: ignore

            attention_mask = attention_mask.to(dtype=self.dtype)  # type: ignore
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)  # type: ignore  # noqa: E501
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        ###############################
        # PREPARE USE ROTARY BY Remove position_embeds
        ###############################
        if not self.use_rotary:
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        ###############################
        # END USE ROTARY BY Remove position_embeds
        ###############################
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("use_cache=True is incompatible")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):  # type: ignore  # noqa: E501
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(  # type: ignore
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)  # type: ignore  # noqa: E501
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)  # type: ignore
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            use_cache,
                            output_attentions,
                            position_ids=position_ids,
                        )

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],  # type: ignore
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],  # type: ignore
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    position_ids=position_ids,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)  # type: ignore

            if output_attentions:
                all_self_attentions = all_self_attentions + (  # type: ignore
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (  # type: ignore
                        outputs[3 if use_cache else 2],
                    )

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class EditGPT2LMHeadModel(OriginalGPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = EditGPT2Model(config)

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)  # type: ignore
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


def make_model(
    pretrained_name,
    max_tokens,
    tokenizer,
    use_flash,
    use_checkpointing,
    device,
    use_rotary,
):
    config = AutoConfig.from_pretrained(
        pretrained_name,
        vocab_size=len(tokenizer),
        n_ctx=max_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        optimize_cuda_cache=True,
    )
    ###############################
    # PREPARE USE ROTARY BY CONFIG
    ###############################
    config.use_rotary = use_rotary
    ###############################
    # END PREPARE USE ROTARY BY CONFIG
    ###############################
    model = EditGPT2LMHeadModel(config).to(device)

    GPT2AttentionWithRotary._attn = _attn_orig
    if use_flash:
        print("Use Flash Attention")
        GPT2AttentionWithRotary._attn = _attn_wrapper

    model.resize_token_embeddings(len(tokenizer))

    # https://www.kaggle.com/code/vad13irt/optimization-approaches-for-transformers
    if use_checkpointing:
        print("Use Gradient Checkpointing")
        model.gradient_checkpointing_enable()
    print(model)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPT-2 size requires_grad: {model_size/1000**2:.1f}M parameters")

    return model
