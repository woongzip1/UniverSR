"""
UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching
Inference wrapper module.
"""

import os
from typing import Optional, Union

import numpy as np
import torch
import torchaudio
import yaml
from huggingface_hub import hf_hub_download

from universr.models.unet import ConvNeXtUNetCond
from universr.flow.path import OriginalCFMPath
from universr.flow.solver import CFGVectorFieldODE, VectorFieldODE, TorchDiffeqSolver
from universr.utils.spectral_ops import AmplitudeCompressedComplexSTFT


# Supported input sample rates (kHz) and their corresponding LR frequency bins
SUPPORTED_INPUT_SR = {8000, 12000, 16000, 24000}
TARGET_SR = 48000


class UniverSR(torch.nn.Module):
    """
    UniverSR inference wrapper.

    Performs audio super-resolution from low sample rates (8/12/16/24 kHz)
    to 48 kHz using vocoder-free flow matching in the complex STFT domain.

    Example:
        >>> model = UniverSR.from_pretrained("woongzip1/universr-speech")
        >>> output = model.enhance("input.wav", input_sr=16000)
        >>> torchaudio.save("output.wav", output.cpu(), 48000)
    """

    def __init__(
        self,
        model: ConvNeXtUNetCond,
        transform: AmplitudeCompressedComplexSTFT,
        path: OriginalCFMPath,
        device: str = "cuda",
    ):
        super().__init__()
        self.model = model
        self.transform = transform
        self.path = path
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        repo_id_or_path: str,
        device: str = "cuda",
        revision: Optional[str] = None,
    ) -> "UniverSR":
        """
        Load a pretrained UniverSR model.

        Args:
            repo_id_or_path: HuggingFace repo ID (e.g. "woongzip1/universr-speech")
                             or local directory path containing config.yaml and pytorch_model.bin.
            device: Device to load the model on.
            revision: Optional HuggingFace revision (branch, tag, or commit hash).

        Returns:
            UniverSR instance ready for inference.
        """
        if os.path.isdir(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.yaml")
            model_path = os.path.join(repo_id_or_path, "pytorch_model.bin")
        else:
            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.yaml", revision=revision
            )
            model_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="pytorch_model.bin", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Build model
        model = ConvNeXtUNetCond(**config["model"])
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Build transform
        transform = AmplitudeCompressedComplexSTFT(**config["transform"])
        transform.to(device)

        # Build probability path
        path_args = config.get("path", {}).get("init_args", {"sigma_min": 1e-4})
        path = OriginalCFMPath(**path_args)

        return cls(model=model, transform=transform, path=path, device=device)

    @classmethod
    def from_local(
        cls,
        ckpt_path: str,
        config_path: str,
        device: str = "cuda",
    ) -> "UniverSR":
        """
        Load UniverSR from a local checkpoint (e.g. training checkpoint with optimizer state).

        This handles the standard training checkpoint format where weights are stored
        under the 'model_state_dict' key, as opposed to from_pretrained() which expects
        a clean state_dict saved as pytorch_model.bin.

        Args:
            ckpt_path: Path to checkpoint file (.pth).
            config_path: Path to YAML config file.
            device: Device to load the model on.

        Returns:
            UniverSR instance ready for inference.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model = ConvNeXtUNetCond(**config["model"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle both formats: raw state_dict or training checkpoint
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.to(device).eval()

        transform = AmplitudeCompressedComplexSTFT(**config["transform"])
        transform.to(device)

        path_args = config.get("path", {}).get("init_args", {"sigma_min": 1e-4})
        path = OriginalCFMPath(**path_args)

        return cls(model=model, transform=transform, path=path, device=device)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def enhance(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        input_sr: Optional[int] = None,
        target_sr: int = TARGET_SR,
        ode_method: str = "midpoint",
        ode_steps: int = 4,
        guidance_scale: Optional[float] = 1.5,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """
        Enhance a low-resolution audio signal to high-resolution.

        Args:
            audio: Input audio. Can be:
                   - str: path to a .wav file
                   - torch.Tensor: waveform tensor of shape (T,), (1, T), or (1, 1, T)
                   - np.ndarray: waveform array
            input_sr: Effective bandwidth of the input in Hz (e.g. 8000, 16000).
                      For file input: auto-detected from the file's native sample rate
                      if it matches a supported rate (8/12/16/24 kHz). Required if the
                      file is already at 48 kHz but has limited bandwidth.
                      For tensor/array input: always required.
            target_sr: Target sample rate in Hz. Default: 48000.
            ode_method: ODE solver method. One of 'euler', 'midpoint', 'rk4'.
            ode_steps: Number of ODE integration steps.
            guidance_scale: Classifier-free guidance scale. None or 0 disables CFG.
            seed: Random seed for deterministic output. None disables seeding.

        Returns:
            Enhanced waveform tensor of shape (1,T) at target_sr.
        """
        # Load audio
        wav, file_sr = self._load_audio(audio, input_sr=input_sr)
        wav = wav.to(self._device)

        # Determine the effective bandwidth SR
        effective_sr = input_sr if input_sr is not None else file_sr

        if effective_sr not in SUPPORTED_INPUT_SR:
            if effective_sr == target_sr and input_sr is None:
                raise ValueError(
                    f"Input audio is already at {target_sr} Hz. "
                    f"Please specify input_sr to indicate the effective bandwidth "
                    f"(e.g., input_sr=16000). Supported: {sorted(SUPPORTED_INPUT_SR)}"
                )
            raise ValueError(
                f"Effective input sample rate {effective_sr} Hz is not supported. "
                f"Supported rates: {sorted(SUPPORTED_INPUT_SR)}"
            )

        # Prepare the 48 kHz LR input for the model
        if file_sr == target_sr:
            # Simulate the training degradation: downsample → upsample to match
            wav = self._apply_bandwidth_limit(wav, effective_sr, target_sr)
        elif file_sr != target_sr:
            # File is truly low-resolution; resample up to 48 kHz
            wav = torchaudio.functional.resample(wav, orig_freq=file_sr, new_freq=target_sr)

        # Minimum length guard
        MIN_SAMPLES = 32_768
        original_len = wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, max(0, MIN_SAMPLES - wav.shape[-1])))

        # Ensure shape is [B, C, T] = [1, 1, T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(0)

        sr_khz = effective_sr // 1000

        # Run flow matching SR
        output = self._inference(wav, sr_khz, ode_method, ode_steps, guidance_scale, seed)

        # (1,T)
        return output[..., :original_len]

    # ------------------------------------------------------------------ #
    #  Internal methods                                                   #
    # ------------------------------------------------------------------ #

    def _load_audio(
        self, audio: Union[str, torch.Tensor, np.ndarray], input_sr: Optional[int] = None,
    ) -> tuple:
        """
        Load and validate audio input.

        Returns:
            (waveform, file_sr): The waveform tensor and its *actual* sample rate.
        """
        if isinstance(audio, str):
            wav, file_sr = torchaudio.load(audio)
            # Mix to mono if stereo
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav, file_sr

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if isinstance(audio, torch.Tensor):
            if input_sr is None:
                raise ValueError("input_sr is required when passing a tensor or array.")
            return audio.float(), input_sr

        raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _apply_bandwidth_limit(
        self, wav: torch.Tensor, effective_sr: int, target_sr: int,
    ) -> torch.Tensor:
        """
        Simulate low-resolution input from a high-sample-rate waveform.

        Applies the same downsample-then-upsample pipeline used during training
        (see WaveformCollator._apply_lpf) so that the spectral cutoff pattern
        matches what the model expects.

        Args:
            wav: Waveform at target_sr. Shape: (1, T) or (T,).
            effective_sr: The effective bandwidth in Hz (e.g. 8000).
            target_sr: The native sample rate of wav (e.g. 48000).

        Returns:
            Bandwidth-limited waveform at target_sr, same length as input.
        """
        original_len = wav.shape[-1]
        lr = torchaudio.functional.resample(wav, orig_freq=target_sr, new_freq=effective_sr)
        lr = torchaudio.functional.resample(lr, orig_freq=effective_sr, new_freq=target_sr)
        return lr[..., :original_len]

    def _preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to amplitude-compressed complex STFT representation.
        [B, C, T] -> [B, 2, F-1, T_frames]  (real/imag channels, drop Nyquist bin)
        """
        spec = self.transform(waveform)              # [B, C, F, T_frames] complex
        real = torch.view_as_real(spec.squeeze(1))    # [B, F, T_frames, 2]
        real = real.permute(0, 3, 1, 2)               # [B, 2, F, T_frames]
        return real[:, :, :-1, :]                      # drop Nyquist bin

    def _postprocess(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert STFT representation back to waveform.
        [B, 2, F-1, T_frames] -> [B, T]
        """
        spec = torch.nn.functional.pad(spec, [0, 0, 0, 1], value=0)  # restore Nyquist
        spec = spec.permute(0, 2, 3, 1).contiguous()                  # [B, F, T, 2]
        spec = torch.view_as_complex(spec)                             # [B, F, T] complex
        waveform = self.transform.invert(spec)                         # [B, T]
        return waveform

    def _inference(
        self,
        lr_audio: torch.Tensor,
        sr_khz: int,
        ode_method: str,
        ode_steps: int,
        guidance_scale: Optional[float],
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """
        Core inference pipeline:
        1. STFT the (resampled) LR audio
        2. Extract LR condition bins
        3. Sample noise for HF region
        4. Solve ODE (flow matching)
        5. Concatenate LR + generated HF
        6. iSTFT to waveform
        """
        # Frequency bin bookkeeping
        lr_bin_count = self.model.sr_to_lr_bins[sr_khz]
        hf_start_bin = self.model.total_freq_bins - self.model.hr_freq_bins

        # STFT
        Y = self._preprocess(lr_audio)          # [B, 2, F-1, T]
        Y_lr = Y[:, :, :lr_bin_count, :]         # LR condition
        Y_hr = Y[:, :, hf_start_bin:, :]         # HR target region (for shape reference)

        # Initial noise
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        x0 = self.path.sample_source(Y_hr, generator=generator).to(self._device)

        # Build ODE solver
        if guidance_scale is not None and guidance_scale > 0:
            ode = CFGVectorFieldODE(net=self.model, guidance_scale=guidance_scale)
        else:
            ode = VectorFieldODE(net=self.model)
        solver = TorchDiffeqSolver(ode, method=ode_method)

        # Time discretization
        ts = torch.linspace(0, 1, ode_steps + 1, device=self._device)

        # Solve ODE
        x1_spec = solver.simulate(
            x0, ts=ts, y=Y_lr, sr_values=torch.tensor([sr_khz], device=self._device)
        )

        # Concatenate LR bins + generated HF bins (handle overlapping region)
        slice_start = max(0, lr_bin_count - hf_start_bin)
        x1_spec = x1_spec[:, :, slice_start:, :]
        full_spec = torch.cat([Y_lr, x1_spec], dim=2)

        # iSTFT
        output = self._postprocess(full_spec)
        return output
