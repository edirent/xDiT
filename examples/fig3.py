import sys, os
# add xDiT root to PYTHONPATH so that 'xfuser' module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import time
import torch
import torch.distributed
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from datetime import datetime
from dateutil.parser import isoparse
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
)
from xfuser.model_executor.cache.diffusers_adapters import apply_cache_on_transformer
from datasets import load_dataset
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank

    text_encoder_2 = T5EncoderModel.from_pretrained(
        engine_config.model_config.model,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16
    )

    if args.use_fp8_t5_encoder:
        from optimum.quanto import freeze, qfloat8, quantize
        logging.info(f"rank {local_rank} quantizing text encoder 2")
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

    cache_args = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch.bfloat16,
        text_encoder_2=text_encoder_2,
    )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    else:
        pipe = pipe.to(f"cuda:{local_rank}")

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe.prepare_run(input_config, steps=input_config.num_inference_steps)

    # Load first 20 unique prompts with timestamps
    ds = load_dataset(
        'poloclub/diffusiondb',
        'large_text_only',
        split='train',
        streaming=True
    )

    last_prompt = None
    prompt_ts = []
    for rec in ds:
        p = rec['prompt']
        if p == last_prompt:
            continue
        last_prompt = p

        ts = rec['timestamp']
        if not isinstance(ts, datetime):
            ts = isoparse(ts)
        prompt_ts.append((p, ts))

        if len(prompt_ts) >= 20:
            break
    prompt_ts.sort(key=lambda x: x[1])

    total = len(prompt_ts)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # Generate and save images; profile last iteration
    for idx, (prompt, _) in enumerate(prompt_ts):
        is_last = (idx == total - 1)
        if is_last:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                output = pipe(
                    height=input_config.height,
                    width=input_config.width,
                    prompt=prompt,
                    num_inference_steps=input_config.num_inference_steps,
                    output_type=input_config.output_type,
                    max_sequence_length=256,
                    guidance_scale=input_config.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(input_config.seed + idx),
                )
            # Optionally save profile: prof.export_chrome_trace("./profiler_logs/trace.json")
        else:
            output = pipe(
                height=input_config.height,
                width=input_config.width,
                prompt=prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                max_sequence_length=256,
                guidance_scale=input_config.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed + idx),
            )

        if input_config.output_type == "pil" and pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                image_rank = idx * len(output.images) + i
                image_name = f"flux_result_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                image.save(f"./results/{image_name}")
                print(f"image {image_rank} saved to ./results/{image_name}")

    # Print overall time and memory
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(
            f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
        )

if __name__ == "__main__":
    main()
