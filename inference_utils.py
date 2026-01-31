import re
import json
import numpy as np
import zipfile
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


# 常量
SAMPLE_RATE = 24000
DEFAULT_SPEED = 1.0
DEFAULT_FADE_OUT = 0.05
DEFAULT_PAUSE = 0.05

ALIASES = {
    'en-us': 'a', 'en-gb': 'b', 'es': 'e', 'fr-fr': 'f',
    'hi': 'h', 'it': 'i', 'pt-br': 'p', 'ja': 'j', 'zh': 'z',
}

LANG_CODES = {
    'a': 'American English', 'b': 'British English', 'e': 'es',
    'f': 'fr-fr', 'h': 'hi', 'i': 'it', 'p': 'pt-br',
    'j': 'Japanese', 'z': 'Mandarin Chinese',
}


@dataclass
class G2PContext:
    """G2P"""
    g2p: any
    g2p_type: str
    vocab: Dict[str, int]


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    return text


def split_sentences(text: str, lang_code: str = 'a') -> List[str]:
    if lang_code in ['z', 'j']:
        sentences = re.split(r'([。！？；，、：""''（）【】《》…\n])', text)
    else:
        sentences = re.split(r'([.!?;,:\n])', text)
    
    result = []
    #一句话带一个标点
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sentence = sentences[i] + sentences[i+1]
        else:
            sentence = sentences[i]
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)
    
    # 处理最后没有标点的文本片段，添加默认结束标点
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        last_text = sentences[-1].strip()
        end_punctuation = '。' if lang_code in ['z', 'j'] else '.'
        result.append(last_text + end_punctuation)
    
    return result if result else [text]


def apply_fade_out(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """末尾淡出音频"""
    if len(audio) <= fade_samples or fade_samples <= 0:
        return audio
    fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
    audio_faded = audio.copy()
    audio_faded[-fade_samples:] *= fade_out
    return audio_faded


def audio_numpy_concat(segment_data_list: List[np.ndarray], sr: int = SAMPLE_RATE, 
                       speed: float = DEFAULT_SPEED, pause_duration: float = DEFAULT_PAUSE) -> np.ndarray:
    """拼接音频片段"""
    if not segment_data_list:
        return np.array([], dtype=np.float32)
    
    audio_segments = []
    pause_samples = int((sr * pause_duration) / speed)
    
    for i, segment_data in enumerate(segment_data_list):
        audio_segments.append(segment_data.reshape(-1))
        if i < len(segment_data_list) - 1 and pause_samples > 0:
            audio_segments.append(np.zeros(pause_samples, dtype=np.float32))
    
    return np.concatenate(audio_segments).astype(np.float32)


def load_vocab_from_config(config_path: str) -> Dict[str, int]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('vocab', {})


def init_g2p(lang_code: str, trf: bool = False, en_callable=None):
    lang_code = lang_code.lower()
    lang_code = ALIASES.get(lang_code, lang_code)
    
    if lang_code not in LANG_CODES:
        raise ValueError(f"不支持的语言代码: {lang_code}")
    
    if lang_code in 'ab':
        from misaki import en, espeak
        try:
            fallback = espeak.EspeakFallback(british=lang_code=='b')
        except:
            logger.warning("EspeakFallback 未启用")
            fallback = None
        g2p = en.G2P(trf=trf, british=lang_code=='b', fallback=fallback, unk='')
        return g2p, 'en'
    elif lang_code == 'j':
        from misaki import ja
        return ja.JAG2P(), 'ja'
    elif lang_code == 'z':
        from misaki import zh
        return zh.ZHG2P(version=None, en_callable=en_callable), 'zh'
    else:
        from misaki import espeak
        language = LANG_CODES[lang_code]
        return espeak.EspeakG2P(language=language), 'espeak'


def text_to_phonemes(text: str, g2p, g2p_type: str) -> str:
    if g2p_type == 'en':
        _, tokens = g2p(text)
        phonemes = ''.join(t.phonemes + (' ' if t.whitespace else '') for t in tokens).strip()
        return phonemes
    else:
        phonemes, _ = g2p(text)
        return phonemes


def phonemes_to_input_ids(phonemes: str, vocab: Dict[str, int], debug: bool = False) -> np.ndarray:
    input_ids = []
    skipped_phonemes = []
    skipped_positions = []
    
    for i, p in enumerate(phonemes):
        if p in vocab:
            input_ids.append(vocab[p])
        else:
            skipped_phonemes.append(p)
            skipped_positions.append(i)
            if debug:
                start = max(0, i-5)
                end = min(len(phonemes), i+6)
                context = phonemes[start:end]
                logger.warning(f"未知音素 '{p}' (ord={ord(p)}) 位置={i}, 上下文: ...{context}...")
    
    if skipped_phonemes:
        logger.error(f"总共跳过 {len(skipped_phonemes)} 个音素在位置 {skipped_positions}: {skipped_phonemes}")
    
    return np.array(input_ids, dtype=np.int64)


def load_voice_embedding(voice_path: str, phoneme_len: Optional[int] = None) -> np.ndarray:
    if voice_path.endswith('.pt'):
        with zipfile.ZipFile(voice_path, 'r') as z:
            data_files = [f for f in z.namelist() if f.endswith('data/0')]
            if not data_files:
                raise ValueError(f"No data file found in {voice_path}")
            with z.open(data_files[0]) as f:
                pack = np.frombuffer(f.read(), dtype='<f4').reshape(-1, 1, 256)
    else:
        pack = np.load(voice_path)
    if phoneme_len is not None:
        ref_s = pack[phoneme_len:phoneme_len+1]
    else:
        idx = pack.shape[0] // 2
        ref_s = pack[idx:idx+1]
    return ref_s[0]


def split_input_ids_semantic(
    input_ids: np.ndarray,
    fixed_seq_len: int,
    phonemes: str = None,
    vocab: Dict[str, int] = None
) -> List[Dict]:
    """input_ids分割"""
    content = input_ids[0, 1:-1]
    chunk_with_special = np.concatenate([[0], content, [0]])
    
    # 填充到固定长度
    padding_len = fixed_seq_len - len(chunk_with_special)
    if padding_len > 0:
        chunk_padded = np.concatenate([chunk_with_special, np.zeros(padding_len, dtype=input_ids.dtype)])
    else:
        chunk_padded = chunk_with_special
    
    return [{
        'input_ids': chunk_padded.reshape(1, -1),
        'actual_len': len(chunk_with_special),
        # 'is_last': True,
        # 'is_first': True,
        'trim_end_chars': 0
    }]


def generate_input_ids_from_text(text: str, lang_code: str = None, config_path: str = None, 
                                 g2p=None, g2p_type: str = None, vocab: Dict[str, int] = None,
                                 g2p_context: G2PContext = None):
    """从文本生成input_ids"""
    if g2p_context is not None:
        g2p = g2p_context.g2p
        g2p_type = g2p_context.g2p_type
        vocab = g2p_context.vocab
    else:
        if vocab is None:
            vocab = load_vocab_from_config(config_path)
        if g2p is None:
            g2p, g2p_type = init_g2p(lang_code)
    
    phonemes = text_to_phonemes(text, g2p, g2p_type)
    content_ids = phonemes_to_input_ids(phonemes, vocab, debug=False)
    input_ids = np.concatenate([[0], content_ids, [0]]).reshape(1, -1)
    return input_ids, phonemes


def split_long_sentence(sentence, lang_code, g2p, g2p_type, vocab, max_merge_len=78, depth=0):
    try:
        input_ids, phonemes = generate_input_ids_from_text(
            sentence, g2p=g2p, g2p_type=g2p_type, vocab=vocab
        )
        content_len = input_ids.shape[1]
        
        if content_len <= max_merge_len:
            return [{
                'sentence': sentence,
                'input_ids': input_ids,
                'phonemes': phonemes,
                'content_len': content_len
            }]
        else:
            # 中文/日文按字符长度一半分割
            if lang_code in ['z', 'j']:
                mid = len(sentence) // 2
                first_half = sentence[:mid]
                second_half = sentence[mid:]
            else:
                # 英文按单词个数一半分割
                words = sentence.split()
                mid_word = len(words) // 2
                first_half = ' '.join(words[:mid_word])
                second_half = ' '.join(words[mid_word:])
            
            result_first = split_long_sentence(first_half, lang_code, g2p, g2p_type, vocab, 
                                              max_merge_len, depth + 1)
            result_second = split_long_sentence(second_half, lang_code, g2p, g2p_type, vocab,
                                               max_merge_len, depth + 1)
            
            return result_first + result_second
    except Exception:
        return []


def concat_audios(sub_audios: List[Dict], sr: int = SAMPLE_RATE) -> np.ndarray:
    """拼接音频"""
    if not sub_audios:
        return np.array([], dtype=np.float32)
    
    if len(sub_audios) == 1:
        return sub_audios[0]['audio']
    
    audio_segments = [sub['audio'] for sub in sub_audios]
    return np.concatenate(audio_segments).astype(np.float32)


def process_and_merge_sentences(text: str, lang_code: str, g2p, g2p_type: str, 
                                vocab: Dict[str, int], max_merge_len: int = 96) -> List[Dict]:
    """
    处理文本：清理(待处理)、分句、生成input_ids、长句分割、短句合并
    
    """
    cleaned_text = clean_text(text)
    sentences = split_sentences(cleaned_text, lang_code=lang_code)
    
    # 为每个句子生成 input_ids
    sentence_data = []
    for sentence in sentences:
        try:
            input_ids, phonemes = generate_input_ids_from_text(
                sentence, g2p=g2p, g2p_type=g2p_type, vocab=vocab
            )
            content_len = input_ids.shape[1]
            
            if content_len <= max_merge_len:
                sentence_data.append({
                    'sentence': sentence,
                    'input_ids': input_ids,
                    'phonemes': phonemes,
                    'content_len': content_len,
                    'is_long': False
                })
            else:
                sub_results = split_long_sentence(sentence, lang_code, g2p, g2p_type, vocab, max_merge_len)
                sentence_data.append({
                    'sentence': sentence,
                    'sub_results': sub_results,
                    'is_long': True
                })
        except Exception as e:
            logger.error(f"错误处理句子 '{sentence}': {e}")
    
    if not sentence_data:
        raise ValueError("没有生成任何 input_ids")
    
    # 长句保持分割，短句合并
    merged_groups = []
    i = 0
    
    while i < len(sentence_data):
        if sentence_data[i]['is_long']:
            sub_results = sentence_data[i]['sub_results']
            merged_groups.append({'is_long_split': True, 'sub_results': sub_results})
            i += 1
        else:
            merged_sentences = []
            total_len = 0
            j = i
            
            while j < len(sentence_data) and not sentence_data[j]['is_long']:
                next_len = sentence_data[j]['content_len']
                
                if total_len + next_len < max_merge_len:
                    merged_sentences.append(sentence_data[j]['sentence'])
                    total_len += next_len
                    j += 1
                else:
                    break
            
            if j == i:
                merged_sentences.append(sentence_data[i]['sentence'])
                j = i + 1
            
            # 重新生成合并后的 input_ids
            merged_text = ' '.join(merged_sentences)
            merged_input_ids, merged_phonemes = generate_input_ids_from_text(
                merged_text, g2p=g2p, g2p_type=g2p_type, vocab=vocab
            )
            
            merged_groups.append({'input_ids': merged_input_ids, 'phonemes': merged_phonemes})
            i = j
    
    return merged_groups


def run_batch_inference(engine, merged_groups: List[Dict], voice_path: str, 
                       vocab: Dict[str, int], speed: float = DEFAULT_SPEED, 
                       fade_out_duration: float = DEFAULT_FADE_OUT,
                       sr: int = SAMPLE_RATE) -> List[np.ndarray]:
    """推理"""
    audio_list = []
    
    for group in merged_groups:
        try:
            if group.get('is_long_split', False):
                # 长句分割：对每个子片段推理后拼接
                sub_audios = []
                for sub in group['sub_results']:
                    phoneme_len = sub['input_ids'].shape[1] - 2
                    ref_s = load_voice_embedding(voice_path, phoneme_len=phoneme_len)
                    audio = engine.inference(sub['input_ids'], ref_s, sub['phonemes'], vocab, 
                                           speed=speed, fade_out_duration=0)
                    sub_audios.append({'audio': audio})
                
                combined_audio = concat_audios(sub_audios, sr=sr)
                audio_list.append(combined_audio)
            else:
                # 短句或合并句：直接推理
                phoneme_len = group['input_ids'].shape[1] - 2
                ref_s = load_voice_embedding(voice_path, phoneme_len=phoneme_len)
                audio = engine.inference(group['input_ids'], ref_s, group['phonemes'], vocab,
                                       speed=speed, fade_out_duration=fade_out_duration)
                audio_list.append(audio)
        except Exception as e:
            logger.error(f"推理错误: {e}")
    
    if not audio_list:
        raise ValueError("没有生成任何音频")
    
    return audio_list