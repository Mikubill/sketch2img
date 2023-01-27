import re
import math
import numpy as np
import torch

# Code from https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/8e2aeee4a127b295bfc880800e4a312e0f049b85, modified.


class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, text_encoder, enable_emphasis=True):
        super().__init__()

        self.device = lambda: text_encoder.device
        self.enable_emphasis = enable_emphasis
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""

        self.chunk_length = 75

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """

        if self.enable_emphasis:
            parsed = parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        comma_padding_backtrack = 20  # default value in https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/6cff4401824299a983c8e13424018efc347b4a2b/modules/shared.py#L410
        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)

                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                elif (
                    comma_padding_backtrack != 0
                    and len(chunk.tokens) == self.chunk_length
                    and last_comma != -1
                    and len(chunk.tokens) - last_comma <= comma_padding_backtrack
                ):
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """

        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, and for SD2 it's 1024.
        An example shape returned by this function can be: (2, 77, 768).
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """

        batch_chunks, token_count = self.process_texts(texts)
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        ts = []
        for i in range(chunk_count):
            batch_chunk = [
                chunks[i] if i < len(chunks) else self.empty_chunk()
                for chunks in batch_chunks
            ]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            # self.embeddings.fixes = [x.fixes for x in batch_chunk]

            # for fixes in self.embeddings.fixes:
            #     for position, embedding in fixes:
            #         used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)
            ts.append(tokens)

        return np.hstack(ts), torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = torch.asarray(remade_batch_tokens).to(self.device())

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index + 1 : tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to(self.device())
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(
            z.shape
        )
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        return z


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    CLIP_stop_at_last_layers = 1

    def __init__(self, tokenizer, text_encoder):
        super().__init__(text_encoder)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(",</w>", None)

        self.token_mults = {}
        tokens_with_parens = [
            (k, v)
            for k, v in vocab.items()
            if "(" in k or ")" in k or "[" in k or "]" in k
        ]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == "[":
                    mult /= 1.1
                if c == "]":
                    mult *= 1.1
                if c == "(":
                    mult *= 1.1
                if c == ")":
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.tokenizer.bos_token_id
        self.id_end = self.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)[
            "input_ids"
        ]

        return tokenized

    def encode_with_transformers(self, tokens):
        tokens = tokens.to(self.text_encoder.device)
        outputs = self.text_encoder(tokens, output_hidden_states=True)

        if self.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-self.CLIP_stop_at_last_layers]
            z = self.text_encoder.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        return z


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res
