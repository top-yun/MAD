import os
import torch
from qwen_omni_utils import process_mm_info

MODALITY_QUERY_PROMPT = "To answer this question, which modality is needed (audio, video, or both): "
ANSWER_QUERY_PROMPT = " Answer only 'Yes' or 'No'. Do not include any explanation."

def mm_contrast_decode_qwen(model, processor, question, video_path, audio_path, modal_type, args, gamma=0.5, add_prompt=""):
    """
    Multi-modal contrast decoding for Qwen Omni model
    Similar to VideoLLaMA2's mm_contrast_decode but adapted for Qwen Omni
    """

    # Prepare different modal combinations
    conversations = {}
    head_conversation = None

    # Base system message
    system_message = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    # print("system_message: ", system_message)
    # 1. Audio-Video combination (if both available)
    head_conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Question: " + question + "\n" +add_prompt}
        ]}
    ]
    conversations['av'] = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question}
        ]}
    ]

    # 2. Video only
    if video_path:
        conversations['v'] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question}
            ]}
        ]

    # 3. Audio only
    if audio_path and os.path.exists(audio_path):
        conversations['a'] = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question}
            ]}
        ]

    # 4. Text only
    conversations['t'] = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "text", "text": question}]}
    ]

    # Process conversation
    head_text = processor.apply_chat_template(head_conversation, add_generation_prompt=True, tokenize=False)
    
    # Extract multimodal info
    # use_audio_in_video = modal_key in ['av', 'a']
    audios, images, videos = process_mm_info(head_conversation, use_audio_in_video=True)

    # Prepare inputs
    inputs = processor(
        text=head_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True
    )

    # Move inputs to device
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Get logits for first few tokens
    with torch.inference_mode():
        outputs = model.thinker.forward(**inputs, use_audio_in_video=True)
        logits = outputs.logits
    
        audio_token_idx=processor.tokenizer.encode("audio")[0]
        video_token_idx=processor.tokenizer.encode("video")[0]
        both_token_idx=processor.tokenizer.encode("both")[0]

        audio_logit = logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()
        
        max_new_tokens = args.max_new_tokens
        
        branches = {}
        step_logits = {}
        
        for key, conversation in conversations.items():
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=key in ['av', 'a'])
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=key in ['av', 'a']
            )
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            outputs = model.thinker(**inputs, use_audio_in_video=key in ['av', 'a'], use_cache=True)
            logits = outputs.logits
            step_logits[key] = logits[:, -1, :].detach()
            branches[key] = outputs.past_key_values
            
        generated = []
        alpha_av = 2 + 2 * gamma * both_prob
        alpha_v = 1 -(both_prob - video_prob) * gamma
        alpha_a = 1 -(both_prob - audio_prob) * gamma
        alpha_t = -(video_prob + audio_prob) * gamma
        
        weights = {
            'av': alpha_av,
            'v': alpha_v,
            'a': alpha_a,
            't': alpha_t
        }
        for _ in range(max_new_tokens):
            logits_mat = torch.stack([(w*step_logits[key]).squeeze(0) for key, w in weights.items()], dim=0)   # GPU 상
            avg_logits = logits_mat.sum(dim=0)                                      # [vocab]
            next_token = int(torch.argmax(avg_logits, dim=-1))
            
            # if model.device.index == 0:
            #     print("av: ", processor.tokenizer.decode(torch.argmax(step_logits['av'])),
            #           "v: ", processor.tokenizer.decode(torch.argmax(step_logits['v'])),
            #           "a: ", processor.tokenizer.decode(torch.argmax(step_logits['a'])),
            #           "t: ", processor.tokenizer.decode(torch.argmax(step_logits['t'])),
            #           "next_token: ", processor.tokenizer.decode(next_token),
            #           )
            
            generated.append(next_token)
            
            if next_token == processor.tokenizer.eos_token_id:
                break
            
            next_tok_tensor = torch.tensor([[next_token]], device=model.device, dtype=torch.long)
            
            for key, branch in branches.items():
                outputs = model.thinker(next_tok_tensor, use_audio_in_video=key in ['av', 'a'], use_cache=True, past_key_values=branch)
                logits = outputs.logits[:, -1, :].detach()
                branches[key] = outputs.past_key_values
                step_logits[key] = logits
                
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text
        