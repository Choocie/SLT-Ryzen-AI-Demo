#
# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved. 
#
import zmq
import torch
import logging
import time
import argparse
import os
import sys
import psutil
from transformers import set_seed
from transformers import LlamaTokenizer

import qlinear
from utils import Utils
from model_utils import (
    warmup, 
    decode_prompt,
    decode_prompts,
    get_wikitext2,
    perplexity,
)
from profiler import ProfileAIE
import gc

from modeling_llama_amd import LlamaForCausalLM, LlamaAttention

from pre_quant import run_awq, apply_awq
from quantizer import real_quantize_model_weight
from qmodule import WQLinear

set_seed(123)

def signal_handler(signal, frame):
    print('You pressed Ctrl+C! Exit...')
    sys.exit(0)

def load_model(args):
    tokenizer = LlamaTokenizer.from_pretrained("./llama-2-wts-hf/7B_chat")

    ckpt = "pytorch_llama27b_w_bit_{}_awq{}_{}amd.pt".format(args.w_bit, "_fa" if args.flash_attention else "", "lm_" if args.lm_head else "")
    print(f"Loading from ckpt: {ckpt}")
    if not os.path.exists(ckpt):
        print(f"\n\n ***** Run --task quantize (with/without lm_head) first to save quantized model ...!!! \n\n")
        raise SystemExit 
    model = torch.load(ckpt)

    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer 


if __name__ == "__main__":
    
    config = {
        'use_translation':True,
        'gloss_prompt_port': '1237',
        'use_translation': '1238'
    }
    
    ctx = zmq.Context()

    rcv_sock = ctx.socket(zmq.SUB)
    #rcv_sock.setsockopt(zmq.CONFLATE, 1)
    #rcv_sock.setsockopt(zmq.RCVTIMEO, 5000)
    rcv_sock.connect("tcp://127.0.0.1:" + config['gloss_port'])
    rcv_sock.subscribe("")
    
    send_sock = ctx.socket(zmq.PUB)
    send_sock.bind("tcp://*:" + config['translate_port'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="Dataset - wikitext2-raw-v1, wikitext2-v1", type=str, default="raw", choices=["non-raw", "raw"])
    parser.add_argument('--w_bit', help="weight bit size", type=int, default=4, choices=[3, 4])
    parser.add_argument('--awq', help="load awq scales, clips from pt or run awq", type=str, default="load", choices=["load", "run", "none"]) 
    parser.add_argument("--target", help="cpu, aie, aie_emu", type=str, default="aie", choices=["cpu", "aie_emu", "aie"])
    parser.add_argument('--task', help="quantize: Apply AWQ and save ckpt; perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode set of prompts;", type=str, default="decode", choices=["quantize", "decode", "benchmark", "benchmark_long", "perplexity"] )
    parser.add_argument('--flash_attention', help="Enable flash attention", action='store_true')
    parser.add_argument('--lm_head', help="Enable PerGrp quantization of lm_head layer", action='store_true')
    parser.add_argument('--num_torch_threads', help="Number of torch threads", type=int, default=4, choices=[1, 2, 3, 4, 5, 6, 7, 8])
    args = parser.parse_args()
    
    dev = os.getenv("DEVICE")

    if dev == "stx":
        p = psutil.Process()
        p.cpu_affinity([0, 1, 2, 3])
    torch.set_num_threads(args.num_torch_threads)
    
    log_dir = "./logs_awq_7B"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/log_awq_7B.log"

    logging.basicConfig(filename=log_file,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.CRITICAL)

    model, tokenizer = load_model(args)

    if args.awq != "none":
        for n, m in model.named_modules():
            if isinstance(m, qlinear.QLinearPerGrp):
                print(f"Preparing weights of layer : {n}")
                m.device = "aie"
                m.quantize_weights()
    
    warmup(model, tokenizer)
    
    while True:
        text = rcv_sock.recv().decode("utf-8")
        response = decode_prompt(model,tokenizer,text,max_new_tokens=30)
        print(response)
        send_sock.send((response).encode('ascii'))

    logging.shutdown()
    out_file = log_file.replace(".log", "_profile.csv")
    out_file = open(out_file, "w")
    ProfileAIE.analyze_profiling(False, True, log_file, out_file)
    out_file.close()
    
    rcv_sock.close()
    send_sock.close()
    ctx.term()
