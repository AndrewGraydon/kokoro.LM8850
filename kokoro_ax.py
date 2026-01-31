import argparse
import os
import numpy as np
import soundfile as sf
import time
from typing import Tuple
#from onnxruntime import InferenceSession
from axengine import InferenceSession
from scipy import signal

from inference_utils import (
    audio_numpy_concat, load_vocab_from_config, init_g2p,
    split_input_ids_semantic, generate_input_ids_from_text,
    process_and_merge_sentences, run_batch_inference, apply_fade_out,
    SAMPLE_RATE, DEFAULT_SPEED, DEFAULT_PAUSE, DEFAULT_FADE_OUT
)


class InferenceEngine:
    """推理"""
    
    # 常量
    FIXED_SEQ_LEN = 96
    N_FFT = 20
    HOP_LENGTH = 5
    DOUBLE_INPUT_THRESHOLD = 32  # 输入长度小于此值时复制一倍,适配短文本
    
    def __init__(self, axmodel_dir: str):
        self.axmodel_dir = axmodel_dir
        
        # 加载模型
        model_files = {
            'model1': "kokoro_part1_96.axmodel",
            'model2': "kokoro_part2_96.axmodel",
            'model3': "kokoro_part3_96.axmodel",
            'model4': "model4_har_sim.onnx"
        }
        
        providers = ['CPUExecutionProvider']
        self.session1 = InferenceSession(os.path.join(axmodel_dir, model_files['model1']))
        self.session2 = InferenceSession(os.path.join(axmodel_dir, model_files['model2']))
        self.session3 = InferenceSession(os.path.join(axmodel_dir, model_files['model3']))
        import onnxruntime as ort
        self.session4 = ort.InferenceSession(os.path.join(axmodel_dir, model_files['model4']), providers=providers)
        
        # 统计
        self.model1_time = 0.0
        self.model2_time = 0.0
        self.model3_time = 0.0
        self.har_time = 0.0
        self.inference_count = 0

        self.window = signal.windows.hann(self.N_FFT, sym=False)
    
    def _compute_external_preprocessing(self, input_ids: np.ndarray, actual_len: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算输入预处理：长度和mask"""
        if actual_len is None:
            actual_len = self.FIXED_SEQ_LEN
        input_lengths = np.full((input_ids.shape[0],), actual_len, dtype=np.int64)
        text_mask = np.arange(self.FIXED_SEQ_LEN)[np.newaxis, :] >= input_lengths[:, np.newaxis]
        return input_lengths, text_mask
    
    def _compute_har_onnx(self, F0_pred: np.ndarray) -> np.ndarray:
        """使用ONNX模型计算谐波"""
        return self.session4.run(None, {'F0_pred': F0_pred.astype(np.float32)})[0]
    
    def _postprocess_x_to_audio(self, x):
        """
        x: shape [Batch, Freq, Time] (例如 [1, 513, 100])
        """
        # 1. 拆分频谱和相位部分
        # x 的维度假设是 (Batch, Freq, Time)，NumPy 切片逻辑与 Torch 一致
        spec_part = x[:, :self.N_FFT//2+1, :]
        phase_part = x[:, self.N_FFT//2+1:, :]

        # 2. 数学运算 (Torch -> Numpy)
        spec = np.exp(spec_part)
        phase_sin = np.sin(phase_part)
        
        # torch.clamp(0, 1) -> np.clip(..., 0, 1)
        # torch.pow(2) -> ** 2
        cos_part = np.sqrt(np.clip(1.0 - phase_sin**2, 0, 1))
        
        real = spec * cos_part
        imag = spec * phase_sin
        
        # 3. 组合复数频谱
        # shape: (Batch, Freq, Time)
        complex_spec = real + 1j * imag

        # 4. ISTFT (Scipy 替代 Torch)
        # Scipy 的 istft 通常处理单个信号 (Freq, Time)，如果 Batch=1，我们取第一个
        # 如果你的 batch > 1，这里需要用循环或根据维度调整，通常推理时 batch=1
        zxx = complex_spec[0] 

        # 计算重叠长度
        noverlap = self.N_FFT - self.HOP_LENGTH

        # 执行逆短时傅里叶变换
        # boundary=True 对应 center=True 的填充逻辑，但 Scipy 返回的是含填充的全长
        _, audio = signal.istft(
            zxx,
            fs=1.0, # 采样率不影响数值，只影响返回的时间轴 t，这里只取 audio
            window='hann',
            nperseg=self.N_FFT,
            noverlap=noverlap,
            nfft=self.N_FFT,
            boundary=True, # 对应 center=True 的一部分行为
            time_axis=-1,
            freq_axis=0,
            input_onesided=True  # 输入是双边频谱
        )

        # 5. 处理 Center=True 的对齐问题 (关键步骤)
        # Torch 的 center=True 会在 STFT 时两端填充，ISTFT 时会自动切除
        # Scipy 的 istft 不会自动切除这些填充，我们需要手动切掉两端各 n_fft // 2
        pad_len = self.N_FFT // 2
        if pad_len > 0:
            audio = audio[pad_len : -pad_len]

        scale = np.sqrt(1.0 / self.window.sum()**2)
        audio *= scale

        return audio.astype(np.float32)
    
    def _prepare_input_ids(self, input_ids: np.ndarray, actual_len: int) -> Tuple[np.ndarray, int, bool]:
        """准备输入ID，对短输入进行复制处理"""
        is_doubled = False
        original_actual_len = actual_len
        
        if actual_len <= self.DOUBLE_INPUT_THRESHOLD:
            is_doubled = True
            valid_content = input_ids[:, :actual_len]
            input_ids_doubled = np.concatenate([valid_content, valid_content], axis=1)
            
            padding_len = self.FIXED_SEQ_LEN - input_ids_doubled.shape[1]
            if padding_len > 0:
                input_ids = np.concatenate([input_ids_doubled, np.zeros((1, padding_len), dtype=input_ids.dtype)], axis=1)
            else:
                input_ids = input_ids_doubled[:, :self.FIXED_SEQ_LEN]
            
            actual_len = min(original_actual_len * 2, self.FIXED_SEQ_LEN)
        
        return input_ids, actual_len, is_doubled
    
    def inference_single_chunk(
        self,
        input_ids: np.ndarray,
        ref_s: np.ndarray,
        actual_len: int,
        speed: float
    ) -> Tuple[np.ndarray, int, int]:
        """单个chunk的推理"""
        self.inference_count += 1
        
        input_ids, actual_len, is_doubled = self._prepare_input_ids(input_ids, actual_len)
        
        input_lengths, text_mask = self._compute_external_preprocessing(input_ids, actual_len=actual_len)
        
        # Model1: 预测duration
        t1 = time.time()
        outputs1 = self.session1.run(None, {'input_ids': input_ids.astype(np.int32), 'ref_s': ref_s, 'text_mask': text_mask.astype(np.uint8)})
        self.model1_time += time.time() - t1
        duration, d = outputs1
        
        # 处理duration并对齐
        pred_dur, total_frames = self._process_duration(duration, actual_len, speed)
        pred_aln_trg = self._create_alignment_matrix(pred_dur, total_frames)
        
        # Model2: 预测F0和ASR特征
        d_transposed = np.transpose(d, (0, 2, 1))
        en = d_transposed @ pred_aln_trg
        
        t2 = time.time()
        outputs2 = self.session2.run(None, {
            'en': en.astype(np.float32),
            'ref_s': ref_s,
            'input_ids': input_ids.astype(np.int32),
            'text_mask': text_mask.astype(np.float32),
            'pred_aln_trg': pred_aln_trg.astype(np.float32)
        })
        self.model2_time += time.time() - t2
        F0_pred, N_pred, asr = outputs2
        
        # Model4: 计算谐波
        t_har = time.time()
        har = self._compute_har_onnx(F0_pred)
        self.har_time += time.time() - t_har
        
        # Model3: 解码生成频谱
        t3 = time.time()
        outputs3 = self.session3.run(None, {
            'asr': asr, 'F0_pred': F0_pred, 'N_pred': N_pred, 'ref_s': ref_s, 'har': har
        })
        self.model3_time += time.time() - t3
        x = outputs3[0]
        
        # 转换为音频
        audio = self._postprocess_x_to_audio(x)
        actual_content_frames = pred_dur[:actual_len].sum()
        
        # 如果输入被复制了，截取前一半音频
        if is_doubled:
            audio = audio[:len(audio) // 2]
            actual_content_frames = actual_content_frames // 2
            total_frames = total_frames // 2
        
        return audio, actual_content_frames, total_frames
    
    def _process_duration(self, duration: np.ndarray, actual_len: int, speed: float) -> Tuple[np.ndarray, int]:
        """处理duration预测，调整到固定帧数"""
        duration_processed = 1.0 / (1.0 + np.exp(-duration))
        duration_processed = duration_processed.sum(axis=-1) / speed
        pred_dur_original = np.round(duration_processed).clip(min=1).astype(np.int64).squeeze()
        
        # 分离实际内容和padding
        pred_dur_actual = pred_dur_original[:actual_len]
        pred_dur_padding = np.zeros(self.FIXED_SEQ_LEN - actual_len, dtype=np.int64)
        pred_dur = np.concatenate([pred_dur_actual, pred_dur_padding])
        
        # 调整实际内容部分，只处理长度超出情况
        fixed_total_frames = self.FIXED_SEQ_LEN * 2
        diff = fixed_total_frames - pred_dur[:actual_len].sum()
        
        if diff < 0:
            # 减少帧数
            indices = np.argsort(pred_dur[:actual_len])[::-1]
            decreased = 0
            for idx in indices:
                if pred_dur[idx] > 1 and decreased < abs(diff):
                    pred_dur[idx] -= 1
                    decreased += 1
                if decreased >= abs(diff):
                    break
        
        # 将剩余帧数分配到padding部分
        remaining_frames = fixed_total_frames - pred_dur[:actual_len].sum()
        padding_len = self.FIXED_SEQ_LEN - actual_len
        if remaining_frames > 0 and padding_len > 0:
            frames_per_padding = remaining_frames // padding_len
            remainder = remaining_frames % padding_len
            pred_dur[actual_len:] = frames_per_padding
            if remainder > 0:
                pred_dur[actual_len:actual_len+remainder] += 1
        
        total_frames = pred_dur.sum()
        return pred_dur, total_frames
    
    def _create_alignment_matrix(self, pred_dur: np.ndarray, total_frames: int) -> np.ndarray:
        """创建对齐矩阵"""
        indices = np.repeat(np.arange(self.FIXED_SEQ_LEN), pred_dur)
        pred_aln_trg = np.zeros((self.FIXED_SEQ_LEN, total_frames), dtype=np.float32)
        if len(indices) > 0:
            pred_aln_trg[indices, np.arange(total_frames)] = 1.0
        return pred_aln_trg[np.newaxis, ...]
    
    def _trim_audio_by_content(self, audio: np.ndarray, actual_content_frames: int, 
                              total_frames: int, actual_len: int) -> np.ndarray:
        """根据实际内容比例裁剪音频"""
        padding_len = self.FIXED_SEQ_LEN - actual_len
        if padding_len > 0:
            content_ratio = actual_content_frames / total_frames
            audio_len_to_keep = int(len(audio) * content_ratio)
            return audio[:audio_len_to_keep]
        return audio
    
    
    def inference(
        self,
        input_ids: np.ndarray,
        ref_s: np.ndarray,
        phonemes: str,
        vocab: dict,
        speed: float = DEFAULT_SPEED,
        fade_out_duration: float = DEFAULT_FADE_OUT
    ) -> np.ndarray:
        """推理生成音频"""
        chunks = split_input_ids_semantic(input_ids, self.FIXED_SEQ_LEN)
        fade_samples = int(SAMPLE_RATE * fade_out_duration) if fade_out_duration > 0 else 0
        
        # if len(chunks) == 1:
        #     # 单个chunk
        audio, actual_content_frames, total_frames = self.inference_single_chunk(
            chunks[0]['input_ids'], ref_s, chunks[0]['actual_len'], speed
        )
        audio_trimmed = self._trim_audio_by_content(
            audio, actual_content_frames, total_frames, chunks[0]['actual_len']
        )
        if fade_samples > 0:
            audio_trimmed = apply_fade_out(audio_trimmed, fade_samples)
        return audio_trimmed


class Kokoro:
    def __init__(self,
        axmodel_dir: str,
        config_path: str,
        max_seq_len: int = 96
        ):
        self.engine = InferenceEngine(axmodel_dir)
        self.vocab = load_vocab_from_config(config_path)

        # ISO-639 lang code -> kokoro language
        self.lang_map = {
            'en': 'a',
            'zh': 'z',
            'ja': 'j'
        }
        self.g2p_map = {k: init_g2p(v) for k, v in self.lang_map.items()}

        self.max_seq_len = max_seq_len
  
    
    def _process_text(self, text: str, language: str):
        if language not in self.lang_map.keys():
            print(f"Unknown language: {language}, possible choices are {self.lang_map.keys()}")
            return None
        
        g2p, g2p_type = self.g2p_map[language]
        merged_groups = process_and_merge_sentences(
            text, self.lang_map[language], g2p, g2p_type, self.vocab, max_merge_len=self.max_seq_len
        )
        return merged_groups

    
    def _synthesize_audio(self, 
                          merged_groups, 
                          voice: str, 
                          sample_rate: int=16000, 
                          speed: float=1.0, 
                          pause: float=0.0,
                          fade_out: float=0.3):
        audio_list = run_batch_inference(
            self.engine, merged_groups, voice, self.vocab, 
            speed=speed, fade_out_duration=fade_out
        )
        final_audio = audio_numpy_concat(audio_list, 
                                         sr=sample_rate, speed=speed, pause_duration=pause)
        return final_audio
    

    def run(self, 
            text: str, 
            language: str,
            voice: str, 
            sample_rate: int=16000, 
            speed: float=1.0, 
            pause: float=0.0,
            fade_out: float=0.3) -> np.ndarray:
        merged_groups = self._process_text(text, language)
        if merged_groups is None:
            print("process text failed!")
            return None
        
        audio = self._synthesize_audio(merged_groups, voice, sample_rate, speed, pause, fade_out)
        return audio
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axmodel_dir", "-d", type=str, default="onnx")
    parser.add_argument("--text", "-t", type=str, default="The sky above the port was the color of television, tuned to a dead channel.")
    parser.add_argument("--lang", "-l", type=str, default='en')
    parser.add_argument("--voice", "-v", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default="checkpoints/config.json")
    parser.add_argument("--output", "-o", type=str, default="output.wav")
    parser.add_argument("--fade_out", "-f", type=float, default=0.3)
    parser.add_argument("--max_len", "-m", type=int, default=96)
    args = parser.parse_args()
    
    SPEED = 1.0 #速度暂时固定
    PAUSE = 0.0 #不加长停顿
    
    start = time.time()
    kokoro = Kokoro(args.axmodel_dir, args.config, args.max_len)
    end = time.time()
    print(f"Init kokoro take {end - start} seconds")

    start = time.time()
    final_audio = kokoro.run(
        args.text,
        language=args.lang,
        voice=args.voice,
        sample_rate=SAMPLE_RATE,
        speed=SPEED,
        pause=PAUSE,
        fade_out=args.fade_out
    )
    end = time.time()
    inference_time = end - start
    
    if final_audio is not None:
        sf.write(args.output, final_audio, SAMPLE_RATE)
        
        audio_duration = len(final_audio) / SAMPLE_RATE
        print("\n" + "="*60)
        print(f"输出: {args.output} | 时长: {audio_duration:.2f}s")
        print("="*60)
        print(f"\n rtf:{inference_time/audio_duration:.3f}")
        print("="*60)


if __name__ == "__main__":
    main()