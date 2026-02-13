import json
import os
import argparse
import time

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import warnings
warnings.filterwarnings('ignore')
from utils import mm_contrast_decode_qwen, MODALITY_QUERY_PROMPT, ANSWER_QUERY_PROMPT
from accelerate import InitProcessGroupKwargs
from datetime import timedelta
try:
    from qwen_omni_utils import process_mm_info
except ImportError:
    print("Warning: qwen_omni_utils not found. Please install: pip install qwen-omni-utils[decord] -U")

    def process_mm_info(conversation, use_audio_in_video=False):
        print("using custom process_mm_info function")
        audios = []
        images = []
        videos = []
        for msg in conversation:
            if 'content' in msg:
                for content in msg['content']:
                    if content['type'] == 'video':
                        videos.append(content['video'])
                    elif content['type'] == 'audio':
                        audios.append(content['audio'])
                    elif content['type'] == 'image':
                        images.append(content['image'])
        return audios, images, videos


class CMMDataset(Dataset):
    """Dataset class for CMM benchmark evaluation with Qwen Omni."""

    def __init__(self, qa_data, media_base_dir, modal_type):
        self.qa_data = qa_data
        self.media_base_dir = media_base_dir
        self.modal_type = modal_type
        print(modal_type)
        self.valid_samples = []
        for qa_item in qa_data:
            audio_rel = qa_item.get('audio_path')
            video_rel = qa_item.get('video_path')

            audio_full = None
            if audio_rel:
                cleaned_audio_rel = audio_rel[2:] if audio_rel.startswith('./') else audio_rel
                audio_full = os.path.join(media_base_dir, cleaned_audio_rel)

            video_full = None
            if video_rel:
                cleaned_video_rel = video_rel[2:] if video_rel.startswith('./') else video_rel
                video_full = os.path.join(media_base_dir, cleaned_video_rel)

            has_audio = audio_full and os.path.exists(audio_full)
            has_video = video_full and os.path.exists(video_full)

            include_sample = False
            if modal_type == "a":
                include_sample = has_audio
                missing_type = "audio"
            elif modal_type == "v":
                include_sample = has_video
                missing_type = "video"
            else:  # modal_type == "av"
                include_sample = has_video
                missing_type = "video"

            if include_sample:
                sample = {
                    **qa_item,
                    'audio_full_path': audio_full if has_audio else None,
                    'video_full_path': video_full if has_video else None,
                }
                self.valid_samples.append(sample)
            else:
                if PartialState().is_main_process:
                    identifier = audio_rel or video_rel or qa_item.get('question', 'unknown sample')
                    print(f"Warning: Required {missing_type} file missing for {identifier}, skipping...")
        
        # self.valid_samples=self.valid_samples[800:]
    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        return self.valid_samples[idx]


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return batch


def load_dataset(dataset_path, category=None):
    """Load QA annotations from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    if category and category != 'all':
        qa_data = [item for item in qa_data if item.get('category') == category]

    return qa_data


def run_inference(model, processor, sample, modal_type, accelerator, args):
    """Run inference on a single sample with Qwen Omni."""
    question = sample['question']
    ground_truth = sample['answer']

    category = sample.get('category', '')
    sub_category = sample.get('sub_category', '')
    modality = sample.get('modality', '')
    granularity = sample.get('granularity', '')
    correlation_type = sample.get('correlation_type', '')
    original_video_path = sample.get('video_path', '')
    original_audio_path = sample.get('audio_path', '')

    start_time = time.time()
    
    try :
    
        video_path = sample['video_full_path']
        audio_path = sample['audio_full_path']
        if audio_path is None:
            audio_path = video_path.replace('.mp4', '.wav')
        
        question = question + ANSWER_QUERY_PROMPT
        add_prompt = MODALITY_QUERY_PROMPT
        
        response = mm_contrast_decode_qwen(
            model, processor, question, video_path, audio_path, modal_type, args, gamma=args.gamma, add_prompt=add_prompt
        )
        
        inference_time = time.time() - start_time

        result = {
            'category': category,
            'sub_category': sub_category,
            'modality': modality,
            'granularity': granularity,
            'correlation_type': correlation_type,
            'video_path': original_video_path,
            'audio_path': original_audio_path,
            'question': question,
            'answer': ground_truth,
            'pred': response.strip(),
            'inference_time': inference_time,
            'device': str(accelerator.device),
        }

        return result

    except Exception as e:
        if accelerator.is_main_process:
            identifier = original_video_path or original_audio_path or question
            print(f"Error processing {identifier} on {accelerator.device}: {str(e)}")

        result = {
            'category': category,
            'sub_category': sub_category,
            'modality': modality,
            'granularity': granularity,
            'correlation_type': correlation_type,
            'video_path': original_video_path,
            'audio_path': original_audio_path,
            'question': question,
            'answer': ground_truth,
            'pred': f"ERROR: {str(e)}",
            'inference_time': 0,
            'device': str(accelerator.device),
        }
        return result


def evaluate_dataset_qwen_omni(args):
    """Main evaluation function using Qwen Omni with accelerate."""

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])

    if accelerator.is_main_process:
        print(f"Using accelerate with {accelerator.num_processes} processes")
        if torch.cuda.is_available():
            print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        print(f"Current process device: {accelerator.device}")
        print(f"Loading Qwen Omni model from {args.model_path}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=accelerator.device,
        # _attn_implementation = "sdpa"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    if args.disable_talker:
        model.disable_talker()
        if accelerator.is_main_process:
            print("Disabled audio generation to save ~2GB GPU memory")

    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_path}")
        if args.category != 'all':
            print(f"Filtering category: {args.category}")

    qa_data = load_dataset(args.dataset_path, args.category)

    if accelerator.is_main_process:
        print(f"Total samples in dataset: {len(qa_data)}")

    dataset = CMMDataset(qa_data, args.media_base_dir, args.modal_type)

    if accelerator.is_main_process:
        print(f"Total valid samples: {len(dataset)}")
        print(f"Dataset length (after filtering): {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    dataloader = accelerator.prepare(dataloader)

    all_results = []
    start_time = time.time()

    if accelerator.is_main_process:
        print("Starting evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            dataloader,
            desc=f"GPU {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )):
            batch_results = []

            for sample in batch:
                result = run_inference(model, processor, sample, args.modal_type, accelerator, args)
                batch_results.append(result)

                if args.verbose and accelerator.is_main_process:
                    media_id = sample.get('video_path') or sample.get('audio_path') or 'unknown'
                    print(f"\nProcessed: {media_id}")
                    print(f"Category: {sample.get('category', 'unknown')}")
                    print(f"Question: {sample.get('question', 'unknown')}")
                    print(f"Answer: {sample.get('answer', 'unknown')}")
                    print(f"Prediction: {result['pred']}")
                    print(f"Time: {result['inference_time']:.2f}s")

            all_results.extend(batch_results)

            if (batch_idx + 1) % args.save_interval == 0:
                if accelerator.is_main_process:
                    processed = (batch_idx + 1) * args.batch_size * accelerator.num_processes
                    print(f"Processed {processed} samples")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    all_results = accelerator.gather_for_metrics(all_results)

    total_time = time.time() - start_time

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)

        output_path = args.output_path
        if args.category != 'all' and args.output_path == './output.json':
            base_name, ext = os.path.splitext(args.output_path)
            output_path = f"{base_name}_{args.category}{ext}"

        output_file = output_path.replace('.json', f'_{args.modal_type}_{args.gamma}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                result_clean = {k: v for k, v in result.items() if k not in ['inference_time', 'device']}
                f.write(json.dumps(result_clean, ensure_ascii=False) + '\n')

        print("\nEvaluation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {output_file}")
        print(f"Total samples processed: {len(all_results)}")

        successful_samples = [r for r in all_results if not r['pred'].startswith('ERROR:')]
        print(f"Successful inferences: {len(successful_samples)}")
        print(f"Failed inferences: {len(all_results) - len(successful_samples)}")

        if successful_samples:
            avg_time = sum(r['inference_time'] for r in successful_samples) / len(successful_samples)
            print(f"Average inference time per sample: {avg_time:.2f}s")
            print(f"Throughput: {len(successful_samples) / total_time:.2f} samples/s")

        device_stats = {}
        for result in all_results:
            device = result.get('device', 'unknown')
            if device not in device_stats:
                device_stats[device] = {'total': 0, 'successful': 0, 'time': 0}

            device_stats[device]['total'] += 1
            if not result['pred'].startswith('ERROR:'):
                device_stats[device]['successful'] += 1
                device_stats[device]['time'] += result['inference_time']

        print("\nPer-Device Statistics:")
        for device, stats in device_stats.items():
            if stats['successful'] > 0:
                avg_time = stats['time'] / stats['successful']
                print(f"  {device}: {stats['successful']}/{stats['total']} successful, avg time: {avg_time:.2f}s")

        category_stats = {}
        for result in all_results:
            cat = result.get('category', 'unknown')
            if cat not in category_stats:
                category_stats[cat] = {'total': 0, 'successful': 0}

            category_stats[cat]['total'] += 1
            if not result['pred'].startswith('ERROR:'):
                category_stats[cat]['successful'] += 1

        print("\nPer-Category Statistics:")
        for cat, stats in category_stats.items():
            print(f"  {cat}: {stats['successful']}/{stats['total']} successful")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni on CMM benchmark with accelerate")

    parser.add_argument('--model-path', type=str, required=False,
                        default='Qwen/Qwen2.5-Omni-7B',
                        help='Path to the Qwen Omni model')
    parser.add_argument('--dataset-path', type=str, required=False,
                        default='/home/junho/lbk/VideoLLaMA2/CMM/all_data_final_reorg.json',
                        help='Path to CMM dataset JSON file')
    parser.add_argument('--media-base-dir', '--video-base-dir', dest='media_base_dir', type=str, required=False,
                        default='/home/junho/.cache/huggingface/cmm',
                        help='Base directory containing CMM media files')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], required=True,
                        help='Modal type: a=audio, v=video, av=audio-video')
    parser.add_argument('--category', type=str, required=False,
                        default='over-reliance_unimodal_priors',
                        choices=['all', 'inter-modality_spurious_correlation', 'over-reliance_unimodal_priors'],
                        help='Category to evaluate (default: over-reliance_unimodal_priors)')
    parser.add_argument('--output-path', type=str, required=False,
                        default='./output.json',
                        help='Path to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of dataloader workers')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save intermediate results every N batches')
    parser.add_argument('--max-new-tokens', type=int, default=1,
                        help='Maximum new tokens to generate')
    parser.add_argument('--disable-talker', action='store_true',
                        help='Disable audio generation to save GPU memory (~2GB)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each sample')
    parser.add_argument('--gamma', type=float, default=2.5,
                        help='Gamma for contrast decoding')
    args = parser.parse_args()

    evaluate_dataset_qwen_omni(args)
