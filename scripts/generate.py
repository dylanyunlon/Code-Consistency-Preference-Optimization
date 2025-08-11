from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk
from vllm import LLM, SamplingParams

import argparse
import torch
import json
import os
from pathlib import Path
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="/data/jiacheng/dylan/aaai/Code-Consistency-Preference-Optimization/cppo/mistralai/Mistral-7B-Instruct-v0.2"
    )
    parser.add_argument("--output_dir", type=str, default="generated/iter1")
    parser.add_argument("--prompts", type=str, default="dylansss/ccpo_math_dataset")
    parser.add_argument("--maxlen", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--frac_len", type=int, default=0)
    parser.add_argument("--data_frac", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    # 添加限制参数
    parser.add_argument("--limit_samples", type=int, default=None, 
                       help="限制处理的样本数量，用于测试")
    return parser.parse_args()


def apply_template(text, tokenizer):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, {"role": "assistant", "content": "None"}],
        tokenize=False, add_generate_prompt=True
    ).split("None")[0]


def split_prompts(data_items, frac_len, data_frac):
    """修改分片函数，处理完整的数据项而不仅仅是prompts"""
    if frac_len > 0:
        split_len = frac_len
        if split_len * (data_frac + 1) > len(data_items):
            return data_items[split_len * data_frac:]
        else:
            return data_items[split_len * data_frac: split_len * (data_frac + 1)]
    else:
        return data_items[:]


def main():
    args = parse_arguments()
    model_path = args.model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载完整数据集
    print(f"📊 加载数据集: {args.prompts}")
    data = load_dataset(args.prompts, split="train")
    print(f"✅ 原始数据集大小: {len(data)}")

    # 构建完整的数据项列表（保持原始索引）
    full_data_items = []
    for idx in range(len(data)):
        full_data_items.append({
            'original_index': idx,
            'prompt': data[idx]["prompt"],
            'answer': data[idx]["answer"],
            'source': data[idx].get("source", "unknown")
        })

    # 样本数量限制（保持原始索引信息）
    if args.limit_samples is not None:
        print(f"🧪 测试模式：限制处理样本数量为 {args.limit_samples}")
        # 使用固定种子确保可重现
        random.seed(42)
        selected_items = random.sample(full_data_items, min(args.limit_samples, len(full_data_items)))
        # 按原始索引排序，保持一定的顺序
        selected_items.sort(key=lambda x: x['original_index'])
        full_data_items = selected_items
        print(f"✅ 已选择 {len(full_data_items)} 个样本进行处理")
        print(f"   原始索引范围: {full_data_items[0]['original_index']} - {full_data_items[-1]['original_index']}")

    # 初始化分词器
    if "mistral" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("/data/jiacheng/dylan/aaai/Code-Consistency-Preference-Optimization/cppo/mistralai/Mistral-7B-Instruct-v0.2")
    elif "llama-3" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    elif "gemma-2" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    else:
        raise ValueError("Model not supported")
    tokenizer.pad_token = tokenizer.eos_token

    # 应用模板到prompts
    for item in full_data_items:
        item['formatted_prompt'] = apply_template(item['prompt'], tokenizer)

    print(f"📊 格式化后的数据项数量: {len(full_data_items)}")
    if full_data_items:
        print(f"示例格式化prompt: {full_data_items[0]['formatted_prompt'][:100]}...")
    
    # 分片处理
    data_frac, frac_len = args.data_frac, args.frac_len
    processed_data_items = split_prompts(full_data_items, frac_len, data_frac)
    print(f"📊 分片后数据项数量: {len(processed_data_items)}")

    # 提取prompts用于生成
    prompts_for_generation = [item['formatted_prompt'] for item in processed_data_items]

    # 初始化LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.world_size,
    )

    pairs = args.pairs
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存数据元信息（关键改进！）
    metadata = {
        'total_original_samples': len(data),
        'selected_samples': len(full_data_items),
        'processed_samples': len(processed_data_items),
        'data_frac': data_frac,
        'frac_len': frac_len,
        'limit_samples': args.limit_samples,
        'pairs': pairs,
        'data_items': [
            {
                'index': i,
                'original_index': item['original_index'],
                'prompt': item['prompt'],
                'answer': item['answer'],
                'source': item['source']
            }
            for i, item in enumerate(processed_data_items)
        ]
    }
    
    metadata_file = f"{args.output_dir}/metadata_{data_frac}.json"
    with open(metadata_file, "w", encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 数据元信息已保存: {metadata_file}")

    # 生成responses
    for p in range(pairs):
        print(f"🚀 生成第 {p+1}/{pairs} 对响应...")
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=args.maxlen,
            seed=p * 50,
        )
        response = llm.generate(prompts_for_generation, sampling_params)
        output = list(map(lambda x: x.outputs[0].text, response))
        
        # 保存responses（保持与metadata的索引一致）
        response_file = f"{args.output_dir}/responses_{p}.json"
        with open(response_file, "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"✅ 第 {p+1} 对响应已保存: {response_file}")

    print(f"\n🎉 所有响应生成完成！输出目录: {args.output_dir}")
    print(f"📋 生成的文件:")
    print(f"   - metadata_{data_frac}.json (数据元信息)")
    for p in range(pairs):
        print(f"   - responses_{p}.json (第{p+1}对响应)")
    
    print(f"\n📊 数据统计:")
    print(f"   - 原始数据集大小: {metadata['total_original_samples']}")
    print(f"   - 选择处理样本: {metadata['selected_samples']}")
    print(f"   - 最终处理样本: {metadata['processed_samples']}")
    print(f"   - 生成响应对数: {pairs}")


if __name__ == "__main__":
    main()