import numpy as np
import librosa
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class ISTFTComparison:
    def __init__(self, n_fft=2048, hop_length=512, sr=22050):
        """
        初始化iSTFT比较测试
        
        Parameters:
        -----------
        n_fft: int
            傅里叶变换大小
        hop_length: int
            帧移长度
        sr: int
            采样率
        """
        self.N_FFT = n_fft
        self.HOP_LENGTH = hop_length
        self.SR = sr
        self.window = signal.windows.hann(n_fft, sym=False)
        
    def generate_test_signals(self, duration=1.0):
        """
        生成测试信号
        
        Returns:
        --------
        dict: 包含不同测试信号的字典
        """
        fs = self.SR
        t = np.linspace(0, duration, int(fs * duration))
        
        # 1. 正弦波
        sine_440 = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # 2. 白噪声
        np.random.seed(42)
        noise = 0.3 * np.random.randn(len(t))
        
        # 3. 线性扫频信号
        chirp = 0.7 * signal.chirp(t, f0=100, f1=2000, t1=duration, method='linear')
        
        # 4. 混合信号
        mixed = 0.3 * sine_440 + 0.2 * chirp + 0.1 * noise
        
        # 5. 脉冲信号
        impulse = np.zeros_like(t)
        impulse[len(t)//2] = 1.0
        
        return {
            'sine_440': sine_440,
            'noise': noise,
            'chirp': chirp,
            'mixed': mixed,
            'impulse': impulse
        }
    
    def compute_stft(self, audio):
        """计算STFT（使用librosa）"""
        stft = librosa.stft(
            audio,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            win_length=self.N_FFT,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        return stft
    
    def istft_librosa(self, stft):
        """librosa的iSTFT实现"""
        audio = librosa.istft(
            stft,
            hop_length=self.HOP_LENGTH,
            win_length=self.N_FFT,
            window='hann',
            center=True,
            length=None
        )
        return audio
    
    def istft_scipy(self, stft):
        # """修正的scipy iSTFT实现"""
        # # 获取STFT形状
        # n_freq, n_time = stft.shape
        
        # # 如果是单边频谱（librosa默认），需要扩展为双边
        # if n_freq == self.N_FFT // 2 + 1:
        #     # 创建完整的双边频谱
        #     full_spec = np.zeros((self.N_FFT, n_time), dtype=np.complex128)
            
        #     # 填充正频率
        #     full_spec[:n_freq] = stft
            
        #     # 填充负频率（共轭对称）
        #     # 对于实信号，频谱是共轭对称的
        #     for k in range(1, n_freq - 1):
        #         full_spec[self.N_FFT - k] = np.conj(stft[k])
            
        #     # 如果N_FFT是偶数，处理Nyquist频率
        #     if self.N_FFT % 2 == 0:
        #         full_spec[n_freq] = np.conj(stft[n_freq-1])
        # else:
        #     full_spec = stft
        
        # # 计算noverlap
        # noverlap = self.N_FFT - self.HOP_LENGTH
        
        # # 使用修正的scipy istft参数
        # # 关键修正：确保正确的窗口归一化
        # _, audio = signal.istft(
        #     full_spec,
        #     fs=self.SR,  # 使用实际采样率
        #     window=self.window,
        #     nperseg=self.N_FFT,
        #     noverlap=noverlap,
        #     nfft=self.N_FFT,
        #     boundary=True,  # 假设输入是零填充的
        #     time_axis=-1,
        #     freq_axis=0,
        #     scaling='spectrum',  # 重要：使用'spectrum'而不是默认的'psd'
        #     input_onesided=False  # 输入是双边频谱
        # )
        
        # return audio
        """scipy的iSTFT实现"""
        # scipy需要完整的双边频谱
        n_freq, n_time = stft.shape
        
        # 构建完整的双边频谱
        full_spec = np.zeros((self.N_FFT, n_time), dtype=np.complex64)
        
        # 填充正频率
        full_spec[:n_freq] = stft
        
        # 填充负频率（共轭对称）
        # 注意：对于偶数N_FFT，Nyquist频率需要特殊处理
        for k in range(1, n_freq - (1 if self.N_FFT % 2 == 0 else 0)):
            full_spec[self.N_FFT - k] = np.conj(stft[k])
        
        # 计算重叠长度
        noverlap = self.N_FFT - self.HOP_LENGTH
        
        # 使用scipy的istft
        _, audio = signal.istft(
            full_spec,
            fs=self.SR,  # 采样率设为1，实际时间轴通过hop_length和nperseg控制
            nperseg=self.N_FFT,
            noverlap=noverlap,
            nfft=self.N_FFT,
            window='hann',
            boundary=True,
            time_axis=-1,
            freq_axis=0,
            input_onesided=True  # 输入是双边频谱
        )

        scale = np.sqrt(1.0 / self.window.sum()**2)
        audio *= scale
        
        return audio
    
    def compute_errors(self, original, reconstructed_librosa, reconstructed_scipy):
        """计算重建误差"""
        # 确保长度一致
        min_len = min(len(original), len(reconstructed_librosa), len(reconstructed_scipy))
        original = original[:min_len]
        reconstructed_librosa = reconstructed_librosa[:min_len]
        reconstructed_scipy = reconstructed_scipy[:min_len]
        
        # 计算MSE
        mse_librosa = np.mean((original - reconstructed_librosa) ** 2)
        mse_scipy = np.mean((original - reconstructed_scipy) ** 2)
        
        # 计算SNR
        def compute_snr(original, reconstructed):
            signal_power = np.mean(original ** 2)
            noise_power = np.mean((original - reconstructed) ** 2)
            return 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        snr_librosa = compute_snr(original, reconstructed_librosa)
        snr_scipy = compute_snr(original, reconstructed_scipy)
        
        # 计算最大值差异
        max_diff_librosa = np.max(np.abs(original - reconstructed_librosa))
        max_diff_scipy = np.max(np.abs(original - reconstructed_scipy))
        
        # 计算频谱差异
        stft_original = self.compute_stft(original)
        stft_librosa = self.compute_stft(reconstructed_librosa)
        stft_scipy = self.compute_stft(reconstructed_scipy)
        
        spec_mse_librosa = np.mean(np.abs(stft_original - stft_librosa) ** 2)
        spec_mse_scipy = np.mean(np.abs(stft_original - stft_scipy) ** 2)
        
        return {
            'mse': {'librosa': mse_librosa, 'scipy': mse_scipy},
            'snr_db': {'librosa': snr_librosa, 'scipy': snr_scipy},
            'max_diff': {'librosa': max_diff_librosa, 'scipy': max_diff_scipy},
            'spec_mse': {'librosa': spec_mse_librosa, 'scipy': spec_mse_scipy}
        }
    
    def plot_comparison(self, original, reconstructed_librosa, reconstructed_scipy, 
                       errors, signal_name, save_path=None):
        """绘制比较图"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 时域波形对比
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(original[:1000], 'b-', alpha=0.7, label='Original')
        ax1.plot(reconstructed_librosa[:1000], 'r--', alpha=0.7, label='Librosa')
        ax1.set_title(f'{signal_name} - Time Domain (First 1000 samples)')
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(original[:1000], 'b-', alpha=0.7, label='Original')
        ax2.plot(reconstructed_scipy[:1000], 'g--', alpha=0.7, label='Scipy')
        ax2.set_title(f'{signal_name} - Time Domain (First 1000 samples)')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 2. 误差对比
        # ax3 = plt.subplot(3, 2, 3)
        # librosa_error = original - reconstructed_librosa
        # scipy_error = original - reconstructed_scipy
        # ax3.plot(librosa_error[:1000], 'r-', alpha=0.7, label='Librosa Error')
        # ax3.plot(scipy_error[:1000], 'g-', alpha=0.7, label='Scipy Error')
        # ax3.set_title('Reconstruction Error (First 1000 samples)')
        # ax3.set_xlabel('Samples')
        # ax3.set_ylabel('Error')
        # ax3.legend()
        # ax3.grid(True, alpha=0.3)
        
        # 3. 频谱图对比
        # ax4 = plt.subplot(3, 2, 4)
        # D_original = librosa.amplitude_to_db(np.abs(self.compute_stft(original)), ref=np.max)
        # librosa.display.specshow(D_original, sr=self.SR, hop_length=self.HOP_LENGTH,
        #                        x_axis='time', y_axis='log', ax=ax4)
        # ax4.set_title('Original Spectrogram')
        
        # ax5 = plt.subplot(3, 2, 5)
        # D_librosa = librosa.amplitude_to_db(np.abs(self.compute_stft(reconstructed_librosa)), ref=np.max)
        # librosa.display.specshow(D_librosa, sr=self.SR, hop_length=self.HOP_LENGTH,
        #                        x_axis='time', y_axis='log', ax=ax5)
        # ax5.set_title('Librosa Reconstruction Spectrogram')
        
        # ax6 = plt.subplot(3, 2, 6)
        # D_scipy = librosa.amplitude_to_db(np.abs(self.compute_stft(reconstructed_scipy)), ref=np.max)
        # librosa.display.specshow(D_scipy, sr=self.SR, hop_length=self.HOP_LENGTH,
        #                        x_axis='time', y_axis='log', ax=ax6)
        # ax6.set_title('Scipy Reconstruction Spectrogram')
        
        # 添加误差统计文本
        text_str = f"""
        Error Metrics for {signal_name}:
        
        Librosa:
        - MSE: {errors['mse']['librosa']:.2e}
        - SNR: {errors['snr_db']['librosa']:.2f} dB
        - Max Diff: {errors['max_diff']['librosa']:.2e}
        - Spec MSE: {errors['spec_mse']['librosa']:.2e}
        
        Scipy:
        - MSE: {errors['mse']['scipy']:.2e}
        - SNR: {errors['snr_db']['scipy']:.2f} dB
        - Max Diff: {errors['max_diff']['scipy']:.2e}
        - Spec MSE: {errors['spec_mse']['scipy']:.2e}
        """
        
        plt.figtext(0.02, 0.02, text_str, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.2, 1, 0.98])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # plt.show()
    
    def run_comprehensive_test(self, duration=1.0, save_plots=False):
        """
        运行全面的对比测试
        
        Parameters:
        -----------
        duration: float
            测试信号持续时间（秒）
        save_plots: bool
            是否保存图表
        """
        print("=" * 60)
        print("iSTFT Implementation Comparison Test")
        print("=" * 60)
        print(f"Parameters: N_FFT={self.N_FFT}, HOP_LENGTH={self.HOP_LENGTH}, SR={self.SR}")
        print(f"Test duration: {duration}s")
        print()
        
        # 生成测试信号
        test_signals = self.generate_test_signals(duration)
        
        overall_results = {}
        
        for signal_name, audio in test_signals.items():
            print(f"\nTesting signal: {signal_name}")
            print("-" * 40)
            
            # 计算STFT
            stft = self.compute_stft(audio)
            
            # 使用不同方法进行iSTFT重建
            try:
                reconstructed_librosa = self.istft_librosa(stft)
                reconstructed_scipy = self.istft_scipy(stft)
                
                # 计算误差
                errors = self.compute_errors(audio, reconstructed_librosa, reconstructed_scipy)
                overall_results[signal_name] = errors
                
                # 打印结果
                print(f"Original length: {len(audio)}")
                print(f"Librosa recon length: {len(reconstructed_librosa)}")
                print(f"Scipy recon length: {len(reconstructed_scipy)}")
                print()
                print("Error Metrics:")
                print(f"  MSE - Librosa: {errors['mse']['librosa']:.2e}, Scipy: {errors['mse']['scipy']:.2e}")
                print(f"  SNR - Librosa: {errors['snr_db']['librosa']:.2f} dB, Scipy: {errors['snr_db']['scipy']:.2f} dB")
                print(f"  Max Diff - Librosa: {errors['max_diff']['librosa']:.2e}, Scipy: {errors['max_diff']['scipy']:.2e}")
                
                # 绘制图表
                if save_plots:
                    save_path = f"istft_comparison_{signal_name}.png"
                    self.plot_comparison(audio, reconstructed_librosa, reconstructed_scipy,
                                       errors, signal_name, save_path)
                else:
                    self.plot_comparison(audio, reconstructed_librosa, reconstructed_scipy,
                                       errors, signal_name)
                
            except Exception as e:
                print(f"Error processing {signal_name}: {e}")
                continue
        
        # 打印总体统计
        print("\n" + "=" * 60)
        print("Overall Statistics")
        print("=" * 60)
        
        for metric in ['mse', 'snr_db', 'max_diff', 'spec_mse']:
            print(f"\n{metric.upper()}:")
            librosa_values = [results[metric]['librosa'] for results in overall_results.values()]
            scipy_values = [results[metric]['scipy'] for results in overall_results.values()]
            
            print(f"  Librosa - Mean: {np.mean(librosa_values):.2e}, "
                  f"Std: {np.std(librosa_values):.2e}, "
                  f"Min: {np.min(librosa_values):.2e}, "
                  f"Max: {np.max(librosa_values):.2e}")
            
            print(f"  Scipy   - Mean: {np.mean(scipy_values):.2e}, "
                  f"Std: {np.std(scipy_values):.2e}, "
                  f"Min: {np.min(scipy_values):.2e}, "
                  f"Max: {np.max(scipy_values):.2e}")
        
        return overall_results

# 使用示例
if __name__ == "__main__":
    # 初始化测试
    comparator = ISTFTComparison(
        n_fft=2048,
        hop_length=512,
        sr=22050
    )
    
    # 运行快速测试（不保存图表）
    print("Running quick test...")
    results = comparator.run_comprehensive_test(duration=0.5, save_plots=True)
    
    # 或者运行详细测试（保存图表）
    # print("\nRunning detailed test...")
    # results = comparator.run_comprehensive_test(duration=1.0, save_plots=True)
    
    # 特定信号测试
    print("\n" + "=" * 60)
    print("Additional: Testing with different parameters")
    print("=" * 60)
    
    # 测试不同参数配置
    test_configs = [
        (1024, 256),
        (2048, 512),
        (4096, 1024)
    ]
    
    for n_fft, hop_length in test_configs:
        print(f"\nTesting with N_FFT={n_fft}, HOP_LENGTH={hop_length}")
        comparator.N_FFT = n_fft
        comparator.HOP_LENGTH = hop_length
        
        # 只测试正弦波
        test_signals = comparator.generate_test_signals(duration=0.3)
        audio = test_signals['sine_440']
        stft = comparator.compute_stft(audio)
        
        try:
            reconstructed_librosa = comparator.istft_librosa(stft)
            reconstructed_scipy = comparator.istft_scipy(stft)
            errors = comparator.compute_errors(audio, reconstructed_librosa, reconstructed_scipy)
            
            print(f"  Librosa SNR: {errors['snr_db']['librosa']:.2f} dB")
            print(f"  Scipy SNR: {errors['snr_db']['scipy']:.2f} dB")
        except Exception as e:
            print(f"  Error: {e}")