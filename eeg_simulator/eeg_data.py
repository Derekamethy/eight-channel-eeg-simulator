"""EEG data model plus robust NPZ and optional EDF file I/O."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EEGAnnotation:
    onset_seconds: float
    duration_seconds: float
    description: str


@dataclass
class EEGData:
    """Channel-major EEG samples expressed in microvolts."""

    name: str
    samples_uv: NDArray[np.float64]
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    units: str = "uV"
    reference: str = "Cz"
    ground: str = "Pz"
    annotations: list[EEGAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.samples_uv = np.asarray(self.samples_uv, dtype=np.float64)
        if self.samples_uv.ndim != 2:
            raise ValueError("samples_uv must have shape (channels, samples)")
        if self.samples_uv.shape[0] != len(self.channel_names):
            raise ValueError("channel_names must match the first sample dimension")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.samples_uv.shape[1] == 0:
            raise ValueError("EEG data must contain at least one sample")

    @property
    def channel_count(self) -> int:
        return self.samples_uv.shape[0]

    @property
    def sample_count(self) -> int:
        return self.samples_uv.shape[1]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz

    @property
    def time_seconds(self) -> NDArray[np.float64]:
        return np.arange(self.sample_count, dtype=np.float64) / self.sample_rate_hz

    def scaled(self, factor: float) -> "EEGData":
        if factor <= 0:
            raise ValueError("Amplitude scale must be positive")
        metadata = dict(self.metadata)
        metadata["amplitude_scale"] = factor
        return EEGData(
            name=self.name,
            samples_uv=self.samples_uv * factor,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=self.channel_names,
            units=self.units,
            reference=self.reference,
            ground=self.ground,
            annotations=list(self.annotations),
            metadata=metadata,
        )

    def save_npz(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_names": list(self.channel_names),
            "units": self.units,
            "reference": self.reference,
            "ground": self.ground,
            "annotations": [asdict(item) for item in self.annotations],
            "metadata": self.metadata,
        }
        np.savez_compressed(
            output,
            samples_uv=self.samples_uv,
            metadata_json=np.array(json.dumps(payload)),
        )
        return output

    @classmethod
    def load_npz(cls, path: str | Path) -> "EEGData":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            if "samples_uv" not in archive or "metadata_json" not in archive:
                raise ValueError("Not a EEG simulator NPZ file")
            samples = np.asarray(archive["samples_uv"], dtype=np.float64)
            raw_metadata = archive["metadata_json"].item()
        payload = json.loads(str(raw_metadata))
        annotations = [EEGAnnotation(**item) for item in payload.get("annotations", [])]
        return cls(
            name=payload.get("name", source.stem),
            samples_uv=samples,
            sample_rate_hz=float(payload["sample_rate_hz"]),
            channel_names=tuple(payload["channel_names"]),
            units=payload.get("units", "uV"),
            reference=payload.get("reference", "Cz"),
            ground=payload.get("ground", "Pz"),
            annotations=annotations,
            metadata=payload.get("metadata", {}),
        )

    def save_edf(self, path: str | Path) -> Path:
        """Save EDF+ when pyedflib is available; NPZ remains the demo default."""
        try:
            import pyedflib  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("EDF support requires the optional 'pyedflib' package") from exc

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        headers = []
        for channel in self.channel_names:
            headers.append(
                {
                    "label": channel,
                    "dimension": "uV",
                    "sample_frequency": self.sample_rate_hz,
                    "physical_min": -500.0,
                    "physical_max": 500.0,
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "Synthetic engineering demonstration",
                    "prefilter": "None",
                }
            )
        with pyedflib.EdfWriter(
            str(output), self.channel_count, file_type=pyedflib.FILETYPE_EDFPLUS
        ) as writer:
            writer.setSignalHeaders(headers)
            writer.setHeader(
                {
                    "technician": "EEG simulator demo",
                    "recording_additional": f"REF={self.reference}; GND={self.ground}; SYNTHETIC",
                    "patientname": "SYNTHETIC DEMO DATA",
                    "patient_additional": "NOT CLINICAL DATA",
                    "patientcode": "DEMO",
                    "equipment": "Desktop EEG Simulator POC",
                    "admincode": "",
                    "sex": "",
                    "startdate": __import__("datetime").datetime.now(),
                    "birthdate": "",
                }
            )
            writer.writeSamples([row for row in self.samples_uv])
            for annotation in self.annotations:
                writer.writeAnnotation(
                    annotation.onset_seconds,
                    annotation.duration_seconds,
                    annotation.description,
                )
        return output

    @classmethod
    def load_edf(cls, path: str | Path) -> "EEGData":
        try:
            import pyedflib  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("EDF support requires the optional 'pyedflib' package") from exc

        source = Path(path)
        with pyedflib.EdfReader(str(source)) as reader:
            channel_names = tuple(label.strip() for label in reader.getSignalLabels())
            rates = [float(reader.getSampleFrequency(index)) for index in range(reader.signals_in_file)]
            if not rates or not np.allclose(rates, rates[0]):
                raise ValueError("All EDF channels must use the same sample rate")
            samples = np.vstack(
                [reader.readSignal(index).astype(np.float64) for index in range(reader.signals_in_file)]
            )
            raw_annotations = reader.readAnnotations()
            annotations = [
                EEGAnnotation(float(onset), float(duration), str(description))
                for onset, duration, description in zip(*raw_annotations)
            ]
        return cls(
            name=source.stem,
            samples_uv=samples,
            sample_rate_hz=rates[0],
            channel_names=channel_names,
            annotations=annotations,
            metadata={"source_format": "EDF", "synthetic": False},
        )


def load_eeg_file(path: str | Path) -> EEGData:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npz":
        return EEGData.load_npz(source)
    if suffix == ".edf":
        return EEGData.load_edf(source)
    raise ValueError(f"Unsupported EEG file type: {source.suffix or '(none)'}")
