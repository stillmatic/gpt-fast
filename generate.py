# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True
# torch._functorch.config.enable_autograd_cache = True

default_device = "cuda" if torch.cuda.is_available() else "cpu"

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

ENTROPY_BOUNDS = (0.1, 3.5)
VARENTROPY_THRESHOLD = 1.5
INJECTION_COOLDOWN = 15
BRANCH_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 1.0
NOISE_SCALE = 0.05
# hardcoded, represents "..." in 1B model
EOS_TOKEN_ID = 128009
COT_TOKENS = [1131, 14524, 81122, 11748]
MAX_INJECTIONS = 3
PUNCTUATION_TOKENS = [13, 11, 220, 198, 271, 30]


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
):
    # Apply temperature scaling
    logits = logits / max(temperature, 1e-5)

    # Apply top-k filtering if specified
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    if min_p is not None:
        max_prob = probs.max()
        p_threshold = min_p * max_prob
        mask = probs >= p_threshold
        probs = probs * mask.float()
        probs = probs / probs.sum()

    return probs


def beam_search(
    logits: List[torch.Tensor],
    beam_width: int,
    max_beam_steps: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None
) -> List[Tuple[List[int], float]]:
    """
    Performs beam search decoding.

    Args:
        logits (List[torch.Tensor]): 
            A list where each element is a tensor of shape (1, vocab_size) 
            representing the logits at each decoding step.
        beam_width (int): 
            The number of beams to keep at each step.
        max_beam_steps (int): 
            The maximum number of decoding steps.
        temperature (float, optional): 
            Temperature parameter for scaling logits. Default is 1.0.
        top_k (Optional[int], optional): 
            If specified, only the top_k tokens with highest probabilities are considered 
            at each step. Default is None.
        min_p (Optional[float], optional): 
            If specified, filters out tokens with probabilities below min_p * max_prob 
            at each step. Default is None.

    Returns:
        List[Tuple[List[int], float]]: 
            A list of tuples where each tuple contains a generated token sequence 
            and its corresponding cumulative log probability.
    """
    # Initialize the beam with an empty sequence and a score of 0
    beams: List[Tuple[List[int], float]] = [([], 0.0)]

    for step in range(max_beam_steps):
        if step >= len(logits):
            # print(f"Step {step}: No more logits provided. Stopping early.")
            break

        current_logits = logits[step].cpu()  # Shape: (1, vocab_size)
        
        # Convert logits to probabilities using the provided helper function
        probs = logits_to_probs(
            current_logits, 
            temperature=temperature, 
            top_k=top_k, 
            min_p=min_p
        ).squeeze(0)  # Shape: (vocab_size,)

        # Convert probabilities to log probabilities to prevent underflow
        log_probs = torch.log(probs + 1e-10)  # Adding epsilon for numerical stability

        all_candidates: List[Tuple[List[int], float]] = []

        # Iterate over each beam to expand
        for seq, cum_log_prob in beams:
            # Get the top `beam_width` tokens and their log probabilities
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                token = topk_indices[i].item()
                token_log_prob = topk_log_probs[i].item()
                
                # Create a new sequence by appending the selected token
                new_seq = seq + [token]
                
                # Update the cumulative log probability
                new_cum_log_prob = cum_log_prob + token_log_prob
                
                # Add the new candidate to the list
                all_candidates.append((new_seq, new_cum_log_prob))
        
        # Sort all candidates by their cumulative log probability in descending order
        all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        
        # Select the top `beam_width` candidates to form the new beam
        beams = all_candidates[:beam_width]
        
        # print(f"Step {step + 1}/{max_beam_steps}:")
        # for i, (seq, score) in enumerate(beams):
        #     print(f"  Beam {i + 1}: Sequence={seq}, Score={score:.4f}")

    # find top beam, return tokens, probs
    top_beam = max(beams, key=lambda x: x[1])
    tokens, score = top_beam
    # print(f"Top Beam: Sequence={tokens}, Score={score:.4f}")
    return torch.tensor(tokens).cuda(), score

def compute_entropy_and_varentropy(probs):
    # Ensure probs sum to 1 and handle zero probabilities
    probs = probs.clamp(min=1e-12)
    probs = probs / probs.sum()

    # Compute entropy
    log_probs = torch.log2(probs)  # Using log base 2
    entropy = -torch.sum(probs * log_probs)

    # Compute variance of entropy
    neg_log_probs_squared = (-log_probs) ** 2
    expected_neg_log_probs_squared = torch.sum(probs * neg_log_probs_squared)
    varentropy = expected_neg_log_probs_squared - entropy**2

    return entropy, varentropy


def should_branch(probs, entropy, varentropy):
    top_two_probs = torch.topk(probs, 2).values[0]
    branch_value = (top_two_probs[0] - top_two_probs[1]) < BRANCH_THRESHOLD
    # entropy_lower_bound = entropy < ENTROPY_BOUNDS[0]
    entropy_lower_bound = False
    entropy_upper_bound = entropy > ENTROPY_BOUNDS[1]
    varentropy_low = varentropy < VARENTROPY_THRESHOLD

    should_branch = (branch_value | entropy_lower_bound | entropy_upper_bound) & varentropy_low
    if should_branch:
        print(f"Top two: {top_two_probs[1].cpu().item(), top_two_probs[0].cpu().item()}, Entropy: {entropy}, Branch: {branch_value}, Entropy Lower: {entropy_lower_bound}, Entropy Upper: {entropy_upper_bound}")
    return should_branch


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None, min_p: Optional[float] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k, min_p)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def sample_with_entropy(
    logits,
    tokens_since_last_injection: int = 0,
    num_injections: int = 0,
    previous_token: Optional[int] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
):
    probs = logits_to_probs(logits[:, -1], temperature, top_k, min_p)
    entropy, varentropy = compute_entropy_and_varentropy(probs)
    # print(f"Entropy: {entropy.item():.02f}, Varentropy: {varentropy.item():.02f}")
    if should_branch(
        probs, entropy, varentropy
    ) :
        print("===== Branching", entropy, varentropy)
        # adjust temperature and add noise by random multiplier between -0.1 and 0.1
        branching_temperature = temperature * (1 + (torch.rand(1) - 0.5) * 0.2)
        noise = torch.randn_like(probs) * NOISE_SCALE
        noisy_logits = logits[:, -1] + noise
        # beam search
        idx_next, _ = beam_search(noisy_logits, 5, 5, temperature=branching_temperature, top_k=top_k, min_p=min_p)
        idx_next = idx_next.unsqueeze(0)
    else:
        idx_next = multinomial_sample_one_no_sync(probs)

    if num_injections < MAX_INJECTIONS and tokens_since_last_injection > INJECTION_COOLDOWN and varentropy < VARENTROPY_THRESHOLD and entropy > ENTROPY_THRESHOLD and previous_token in PUNCTUATION_TOKENS:
        # randomly select from COT tokens
        print("===== Injecting", entropy, varentropy)
        idx_next = torch.tensor(COT_TOKENS, device="cuda")[torch.randint(len(COT_TOKENS), (1,))].unsqueeze(0)
        tokens_since_last_injection = 0
        num_injections += 1
    else:
        tokens_since_last_injection += 1

    return idx_next, probs, tokens_since_last_injection, num_injections


def prefill(
    model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    tokens_since_last_injection: int,
    num_injections: int,
    previous_token: Optional[int] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample_with_entropy(logits, tokens_since_last_injection, num_injections, previous_token, **sampling_kwargs)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    tokenizer,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []
    tokens_since_last_injection, num_injections = 0, 0
    found_eos = False
    previous_token = None
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob, tokens_since_last_injection, num_injections = decode_one_token(
                model,
                cur_token,
                input_pos,
                tokens_since_last_injection,
                num_injections,
                previous_token,
                **sampling_kwargs,
            )
            if next_token.item() == EOS_TOKEN_ID:
                found_eos = True
            if not found_eos:
                print(tokenizer.decode(next_token.view(-1).tolist()), next_token.item())
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()
            previous_token = next_token.item()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs,
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor(
        [input_pos], dtype=torch.int64, device=cur_token.device
    )
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model,
        cur_token.view(1, -1),
        orig_input_pos.clone(),
        speculate_k,
        **sampling_kwargs,
    )
    draft_tokens = torch.cat(draft_tokens).view(-1)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device),
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs).squeeze(1)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k] / p)
    rejected_locations = (
        torch.rand_like(accept_draft_prob) > accept_draft_prob
    ).nonzero()
    final_tokens = None
    if rejected_locations.shape[0] == 0:  # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        final_tokens = torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        final_tokens = torch.cat([draft_tokens[:accept_length], next_token])
    return final_tokens


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    tokenizer,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = (
        max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    )
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(
                max_batch_size=batch_size, max_seq_length=max_seq_length
            )

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(
        model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs
    ).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[:, input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
            for i in next_tokens[:num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(
            model,
            next_token.view(batch_size, -1),
            input_pos,
            max_new_tokens - 1,
            tokenizer,
            callback=callback,
            **sampling_kwargs,
        )
        seq[:, T + 1 :] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {"accept_counts": accept_counts}
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    # consider not disallowing all tokens
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = "cuda" in device
    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device=default_device,
    min_p: Optional[float] = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    from tp import maybe_init_dist

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    device_sync(device=device)  # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    print("COT TOKEN", tokenizer.encode("Wait"))
    print("PUNC TOKEN", tokenizer.encode("."))

    # print("EOS TOKEN", tokenizer.eos_id())

    if isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # generate a fully synthetic prompt
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    if compile:
        if is_speculative and use_tp:  # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = (
                False  # Bug with cudagraph trees in this case
            )

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(
                model_forward, mode="reduce-overhead", fullgraph=True
            )

        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
        "accept_counts": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device)  # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                tokenizer=tokenizer,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                min_p=min_p,
            )
            aggregate_metrics["accept_counts"].append(metrics["accept_counts"])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device)  # MKG
        t = time.perf_counter() - t0

        if not interactive:
            # Just displaying the first generation
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(generated_tokens_sec)
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec"
        )
        print(
            f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s"
        )
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(
            f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}"
        )

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


default_prompt = """<|start_header_id|>system<|end_header_id|>

You are a world-class AI system, capable of complex reasoning. Begin your response with <thinking> tags and think step by step through the query, and then provide your final response inside <output> tags.<|eot_id|><|start_header_id|>user<|end_header_id|>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<thinking>"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument(
        "--prompt",
        type=int_or_str,
        default=default_prompt,
        help="Input prompt. If it's an integer, will instead generate a synthetic prompt.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=2, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=200, help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size to benchmark with"
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--compile_prefill",
        action="store_true",
        help="Whether to compile the prefill (improves prefill perf, but higher compile times)",
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")
    parser.add_argument(
        "--speculate_k", type=int, default=5, help="Speculative execution depth."
    )
    parser.add_argument(
        "--draft_checkpoint_path",
        type=Path,
        default=None,
        help="Draft checkpoint path.",
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="Device to use"
    )

    args = parser.parse_args()
    main(
        args.prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.batch_size,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.compile_prefill,
        args.profile,
        args.draft_checkpoint_path,
        args.speculate_k,
        args.device,
        args.min_p,
    )
