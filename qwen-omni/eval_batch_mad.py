import sys
import json
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, PartialState, InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm
import time
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import torch.nn.functional as F
from utils import mm_contrast_decode_qwen, MODALITY_QUERY_PROMPT, ANSWER_QUERY_PROMPT
from qwen_omni_utils import process_mm_info
import warnings
warnings.filterwarnings('ignore')

class AVHBenchDataset(Dataset):
    """Dataset class for AVHBench evaluation with Qwen Omni"""

    def __init__(self, qa_data, video_dir, modal_type):
        self.qa_data = qa_data
        self.video_dir = video_dir
        self.modal_type = modal_type

        # Filter out samples with missing videos
        self.valid_samples = []
        for qa_item in qa_data:
            if 'AV' in qa_item['task'] :
                continue
            if modal_type == "a":
                video_path = os.path.join(video_dir, f"{qa_item['video_id']}.wav")
            else:
                video_path = os.path.join(video_dir, f"{qa_item['video_id']}.mp4")

            # Check for audio file as well
            audio_path = video_path.replace("videos", "audios").replace(".mp4", ".wav")

            if os.path.exists(video_path):
                self.valid_samples.append({
                    **qa_item,
                    'video_path': video_path,
                    'audio_path': audio_path if os.path.exists(audio_path) else None
                })
            else:
                if PartialState().is_main_process:
                    print(f"Warning: File {video_path} not found, skipping...")
        # self.valid_samples = self.valid_samples[:20]

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        return self.valid_samples[idx]

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return batch

def load_dataset(dataset_path):
    """Load QA annotations from JSON file"""
    with open(os.path.join(dataset_path, 'QA.json'), 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    return qa_data


def run_inference(model, processor, sample, modal_type, accelerator, args):
    """Run inference on a single sample with Qwen Omni using contrast decoding"""
    video_path = sample['video_path']
    audio_path = sample['audio_path']
    question = sample['text']
    ground_truth = sample['label']
    task = sample['task']
    video_id = sample['video_id']

    # try:
    start_time = time.time()

    # Add task-specific prompt modifications
    if task != "AV Captioning":
        question = question + ANSWER_QUERY_PROMPT

    add_prompt = MODALITY_QUERY_PROMPT
    

    response = mm_contrast_decode_qwen(
        model, processor, question, video_path, audio_path, modal_type, args, gamma=args.gamma, add_prompt=add_prompt
    )

    inference_time = time.time() - start_time

    result = {
        'video_id': video_id,
        'task': task,
        'question': question,
        'ground_truth': ground_truth,
        'prediction': response,
        'inference_time': inference_time,
        'device': str(accelerator.device),
        'contrast_decode': args.use_contrast_decode
    }

    return result

    # except Exception as e:
    #     if accelerator.is_main_process:
    #         print(f"Error processing {video_id} on {accelerator.device}: {str(e)}")

    #     result = {
    #         'video_id': video_id,
    #         'task': task,
    #         'question': question,
    #         'ground_truth': ground_truth,
    #         'prediction': f"ERROR: {str(e)}",
    #         'inference_time': 0,
    #         'device': str(accelerator.device),
    #         'contrast_decode': args.use_contrast_decode
    #     }
    #     return result

def evaluate_dataset_qwen_omni_avcd(args):
    """Main evaluation function using Qwen Omni with contrast decoding and accelerate"""

    # Initialize accelerator
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])

    # Check available GPUs
    if accelerator.is_main_process:
        print(f"Using accelerate with {accelerator.num_processes} processes")
        if torch.cuda.is_available():
            print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        print(f"Current process device: {accelerator.device}")
        print(f"Contrast decoding enabled: {args.use_contrast_decode}")

    # Load model and processor
    if accelerator.is_main_process:
        print(f"Loading Qwen Omni model from {args.model_path}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=accelerator.device,
        attn_implementation="flash_attention_2"
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    # Disable audio generation to save memory if not needed
    if args.disable_talker:
        model.disable_talker()
        if accelerator.is_main_process:
            print("Disabled audio generation to save ~2GB GPU memory")

    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_path}")

    qa_data = load_dataset(args.dataset_path)

    # Determine directory based on modal type
    if args.modal_type == "a":
        media_dir = os.path.join(args.dataset_path, 'audios')
    else:
        media_dir = os.path.join(args.dataset_path, 'videos')

    dataset = AVHBenchDataset(qa_data, media_dir, args.modal_type)

    if accelerator.is_main_process:
        print(f"Total valid samples: {len(dataset)}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Prepare dataloader with accelerate
    dataloader = accelerator.prepare(dataloader)

    # Evaluation loop
    all_results = []
    start_time = time.time()

    if accelerator.is_main_process:
        print(f"Starting evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader,
                                             desc=f"GPU {accelerator.process_index}",
                                             disable=not accelerator.is_local_main_process)):

            batch_results = []

            for sample in batch:
                result = run_inference(model, processor, sample, args.modal_type, accelerator, args)
                batch_results.append(result)

                if args.verbose and accelerator.is_main_process:
                    print(f"\nProcessed: {sample['video_id']}")
                    print(f"Task: {sample['task']}")
                    print(f"Question: {sample['text']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Time: {result['inference_time']:.2f}s")

            all_results.extend(batch_results)

            # Periodic saving and memory cleanup
            if (batch_idx + 1) % args.save_interval == 0:
                if accelerator.is_main_process:
                    print(f"Processed {(batch_idx + 1) * args.batch_size * accelerator.num_processes} samples")

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Gather results from all processes
    all_results = accelerator.gather_for_metrics(all_results)

    total_time = time.time() - start_time

    # Save results (only on main process)
    if accelerator.is_main_process:
        # Sort results by video_id for consistency
        all_results.sort(key=lambda x: x['video_id'])
        
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        # Add contrast decode suffix to filename
        cd_suffix = "_mad" if args.use_contrast_decode else ""
        output_file = args.output_path.replace('.json', f'_qwen_omni_{args.modal_type}{cd_suffix}_{args.gamma}.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {output_file}")
        print(f"Total samples processed: {len(all_results)}")

        # Print basic statistics
        successful_samples = [r for r in all_results if not r['prediction'].startswith('ERROR:')]
        print(f"Successful inferences: {len(successful_samples)}")
        print(f"Failed inferences: {len(all_results) - len(successful_samples)}")

        if successful_samples:
            avg_time = sum(r['inference_time'] for r in successful_samples) / len(successful_samples)
            print(f"Average inference time per sample: {avg_time:.2f}s")
            print(f"Throughput: {len(successful_samples) / total_time:.2f} samples/s")

        # Print per-device statistics
        device_stats = {}
        for result in all_results:
            device = result.get('device', 'unknown')
            if device not in device_stats:
                device_stats[device] = {'total': 0, 'successful': 0, 'time': 0}

            device_stats[device]['total'] += 1
            if not result['prediction'].startswith('ERROR:'):
                device_stats[device]['successful'] += 1
                device_stats[device]['time'] += result['inference_time']

        print(f"\nPer-Device Statistics:")
        for device, stats in device_stats.items():
            if stats['successful'] > 0:
                avg_time = stats['time'] / stats['successful']
                print(f"  {device}: {stats['successful']}/{stats['total']} successful, avg time: {avg_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on AVHBench dataset with contrast decoding and accelerate")

    parser.add_argument('--model-path', type=str, required=False,
                       default='Qwen/Qwen2.5-Omni-7B',
                       help='Path to the Qwen Omni model')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], required=True,
                       help='Modal type: a=audio, v=video, av=audio-video')
    parser.add_argument('--dataset-path', type=str, required=False,
                       default='/mnt/T5/dataset/AVHBench',
                       help='Path to AVHBench dataset')
    parser.add_argument('--output-path', type=str, required=False,
                       default='./results/evaluation_results_qwen_omni_avcd.json',
                       help='Path to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N batches')
    parser.add_argument('--max-new-tokens', type=int, default=1,
                       help='Maximum new tokens to generate')
    parser.add_argument('--use-contrast-decode', action='store_true',
                       help='Enable multi-modal contrast decoding for better performance')
    parser.add_argument('--load-8bit', action='store_true',
                       help='Load model in 8-bit precision')
    parser.add_argument('--load-4bit', action='store_true',
                       help='Load model in 4-bit precision')
    parser.add_argument('--disable-talker', action='store_true',
                       help='Disable audio generation to save GPU memory (~2GB)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each sample')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma for contrast decoding')
    args = parser.parse_args()

    evaluate_dataset_qwen_omni_avcd(args)