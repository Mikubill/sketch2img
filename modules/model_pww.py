import math
import torch.nn.functional as F

import torch
from diffusers import UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttention

def get_attention_scores(attn, query, key, attention_mask=None):

    if attn.upcast_attention:
        query = query.float()
        key = key.float()

    attention_scores = torch.baddbmm(
        torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        ),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=attn.scale,
    )

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if attn.upcast_softmax:
        attention_scores = attention_scores.float()

    return attention_scores


def CrossAttnProcessor(
    attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None
):
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

    query = attn.to_q(hidden_states)
    query = attn.head_to_batch_dim(query)

    encoder_states = hidden_states
    is_xattn = False
    if encoder_hidden_states is not None:
        is_xattn = True
        img_state = encoder_hidden_states["img_state"]
        input_ids = encoder_hidden_states["input_ids"]
        encoder_states = encoder_hidden_states["states"]
        uncond_chunk = encoder_hidden_states["uncond"]
        weight_func = encoder_hidden_states["weight_func"]
        sigma = encoder_hidden_states["sigma"]

    key = attn.to_k(encoder_states)
    value = attn.to_v(encoder_states)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)

    attention_scores = get_attention_scores(attn, query, key, attention_mask)

    if is_xattn:
        b, z, c = attention_scores.shape
        dim = attention_scores.shape[0]
        w = generate_attention_weight(img_state, input_ids, z, c).to(query.device)
        if uncond_chunk:
            cross_attention_weight = weight_func(w, sigma, attention_scores[dim // 2 :])
            attention_scores[dim // 2 :] = (
                attention_scores[dim // 2 :] + cross_attention_weight * attn.scale
            )
        else:
            cross_attention_weight = weight_func(w, sigma, attention_scores)
            attention_scores = attention_scores + cross_attention_weight * attn.scale

    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(query.dtype)

    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    return hidden_states


def generate_attention_weight(img_state, text_ids, z, c):
    if img_state is None:
        return 0

    token_lis = text_ids.tolist()
    ret_tensor = torch.zeros((z, c), dtype=torch.float32)

    for v_as_tokens, img_where_color in img_state:
        is_in = 0
        z0 = img_where_color.shape[0] * img_where_color.shape[1]

        for idx, tok in enumerate(token_lis):
            if token_lis[idx : idx + len(v_as_tokens)] == v_as_tokens:
                is_in = 1

                # print(token_lis[idx : idx + len(v_as_tokens)], v_as_tokens)
                ret_tensor[:, idx : idx + len(v_as_tokens)] += (
                    F.interpolate(
                        img_where_color.unsqueeze(0).unsqueeze(1),
                        scale_factor=math.sqrt(z / z0),
                        mode="bilinear",
                        align_corners=True,
                    )
                    .squeeze()
                    .reshape(-1, 1)
                    .repeat(1, len(v_as_tokens))
                )

        if not is_in == 1:
            print(f"tokens {v_as_tokens} not found in text")

    return ret_tensor


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device="cpu"):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t**2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def encode_sketchs(tokenizer, state):
    img_state = []
    for k, v in state.items():
        if v["map"] is None:
            continue

        v_input = tokenizer(
            k,
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False,
        ).input_ids
        dotmap = v["map"] < 255
        arr = torch.from_numpy(dotmap.astype(int) * v["weight"])
        img_state.append((v_input, arr))

    return img_state


def load_learned_embed_in_clip(
    learned_embeds_path, text_encoder, tokenizer, token=None
):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        print(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        print(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def hook_unet(tokenizer, unet):
    ode_sigmas = get_sigmas_vp(1000)
    org_forward = unet.forward

    def forward(self, sample, t, encoder_hidden_states, *args, **kwargs):
        if encoder_hidden_states is not None:

            text_input = tokenizer(
                [self.st_prompt],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]

            weight_func = lambda w, sigma, qk: self.st_global_weight * w  * math.log(1 + sigma) * qk.max()

            encoder_hidden_states = {
                "img_state": encode_sketchs(tokenizer, self.st_state),
                "input_ids": text_input,
                "states": encoder_hidden_states,
                "uncond": self.st_cfg,
                "sigma": ode_sigmas[1000 - t],
                "weight_func":weight_func,
            }
        return org_forward(sample, t, encoder_hidden_states, *args, **kwargs)

    unet.forward = forward.__get__(unet, UNet2DConditionModel)
    unet.org_forward = org_forward


def set_state(unet, prompt, state, global_weight, do_classifier_free_guidance):
    unet.st_prompt = prompt
    unet.st_state = state
    unet.st_global_weight = global_weight
    unet.st_cfg = do_classifier_free_guidance