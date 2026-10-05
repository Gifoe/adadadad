#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CROISSANT_CONTEXT = {
    "@vocab": "https://schema.org/",
    "sc": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
}


DATASET_OVERRIDES: dict[str, dict[str, Any]] = {
    "AD65": {
        "display_name": "AD65",
        "task": "Neurodegenerative disease classification",
        "task_description": "Segment-level EEG classification across control, frontotemporal dementia, and Alzheimer disease labels.",
        "reference_url": "https://doi.org/10.1038/s41597-023-02811-y",
        "citation_text": "Miltiadous et al. A dataset of scalp EEG recordings of Alzheimer’s disease, frontotemporal dementia and healthy subjects. Scientific Data, 2023.",
    },
    "ADHD": {
        "display_name": "Adult ADHD",
        "task": "Neurodevelopmental disorder classification",
        "task_description": "Healthy vs. ADHD EEG classification.",
        "reference_url": "https://doi.org/10.1038/s41597-023-02531-3",
        "citation_text": "Bajestani et al. An EEG dataset to study neurophysiological biomarkers of adult attention-deficit/hyperactivity disorder. Scientific Data, 2023.",
    },
    "Awakening": {
        "display_name": "Awakening",
        "task": "Consciousness level detection",
        "task_description": "Awake vs. sedated state classification from resting EEG.",
        "reference_url": "https://doi.org/10.1038/s41597-025-04783-2",
        "citation_text": "Bajwa et al. Repeated resting-state EEG from healthy participants before and after propofol sedation. Scientific Data, 2025.",
    },
    "BCIC2A": {
        "display_name": "BCI Competition IV-2A",
        "task": "Motor imagery classification",
        "task_description": "Four-class motor imagery classification.",
        "reference_url": "https://www.bbci.de/competition/iv/",
        "citation_text": "Brunner et al. BCI Competition 2008 – Graz data set A.",
    },
    "BCIC4_1": {
        "display_name": "BCI Competition IV-1",
        "task": "Motor imagery classification",
        "task_description": "Two-class motor imagery classification.",
        "reference_url": "https://www.bbci.de/competition/iv/",
        "citation_text": "Blankertz et al. The BCI Competition IV data sets, 2007/2008.",
    },
    "BCI_Speech": {
        "display_name": "BCI-speech",
        "task": "Speech intention / imagined speech classification",
        "task_description": "EEG classification for imagined speech / keyword decoding.",
        "reference_url": "https://doi.org/10.82901/nemar.nm000113",
        "citation_text": "2020 BCI competition, track 3. NeurIPS competition dataset release via NeMAR.",
    },
    "BenchmarkSSVEP": {
        "display_name": "Benchmark-SSVEP",
        "task": "SSVEP target classification",
        "task_description": "Forty-class SSVEP target classification benchmark.",
        "reference_url": "https://pubmed.ncbi.nlm.nih.gov/27849543/",
        "citation_text": "Wang et al. A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces. IEEE TNSRE, 2017.",
    },
    "Broderick_CP": {
        "display_name": "Broderick (cocktail party)",
        "task": "Auditory attention classification",
        "task_description": "Left vs. right auditory attention classification under cocktail-party listening.",
        "reference_url": "https://doi.org/10.1038/sdata.2018.3",
        "citation_text": "Broderick et al. Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech. Scientific Data / associated release, 2018.",
    },
    "Broderick_Rev": {
        "display_name": "Broderick (reverse)",
        "task": "Speech perception classification",
        "task_description": "Natural vs. time-reversed speech classification.",
        "reference_url": "https://doi.org/10.1038/sdata.2018.3",
        "citation_text": "Broderick et al. Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech. Scientific Data / associated release, 2018.",
    },
    "CHB_MIT": {
        "display_name": "CHB-MIT",
        "task": "Seizure detection",
        "task_description": "Seizure vs. non-seizure EEG classification.",
        "reference_url": "https://physionet.org/content/chbmit/1.0.0/",
        "citation_text": "Goldberger et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation, 2000. CHB-MIT Scalp EEG Database on PhysioNet.",
    },
    "CIRE": {
        "display_name": "CIRE",
        "task": "Affective / prosody-related classification",
        "task_description": "Speech-related affective classification.",
        "reference_url": "https://doi.org/10.1038/s41597-025-04859-z",
        "citation_text": "He et al. CIRE: a cross-individual emotion EEG dataset. Scientific Data, 2025.",
    },
    "ChineseEEG2_RA_Tone": {
        "display_name": "ChineseEEG2-RA-Tone",
        "task": "Reading / language decoding",
        "task_description": "Tone classification from EEG during Chinese reading aloud.",
        "reference_url": "https://doi.org/10.1038/s41597-025-04881-1",
        "citation_text": "Chen et al. ChineseEEG2: an EEG dataset for Chinese speech and reading decoding. Scientific Data, 2025.",
    },
    "DEAP_arousal": {
        "display_name": "DEAP-arousal",
        "task": "Emotion recognition",
        "task_description": "High vs. low arousal classification derived from DEAP ratings.",
        "reference_url": "https://doi.org/10.1109/T-AFFC.2011.15",
        "citation_text": "Koelstra et al. DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE TAC, 2012.",
    },
    "DEAP_valence": {
        "display_name": "DEAP-valence",
        "task": "Emotion recognition",
        "task_description": "High vs. low valence classification derived from DEAP ratings.",
        "reference_url": "https://doi.org/10.1109/T-AFFC.2011.15",
        "citation_text": "Koelstra et al. DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE TAC, 2012.",
    },
    "DUAL_FREQ_SSVEP": {
        "display_name": "Dual-Freq-SSVEP",
        "task": "SSVEP target classification",
        "task_description": "Dual-frequency SSVEP target classification benchmark.",
        "reference_url": "https://doi.org/10.1016/j.eswa.2024.124144",
        "citation_text": "Sun et al. Efficient dual-frequency SSVEP brain-computer interface system exploiting interocular visual resource disparities. Expert Systems with Applications, 2024.",
    },
    "EAV": {
        "display_name": "EAV",
        "task": "Conversational emotion classification",
        "task_description": "Emotion classification during conversational audio-visual stimulation.",
        "reference_url": "https://doi.org/10.1038/s41597-024-03191-3",
        "citation_text": "Lee et al. EAV: an EEG-audio-visual conversational emotion dataset. Scientific Data, 2024.",
    },
    "EEGDenoiseNet_singleCOz": {
        "display_name": "EEGDenoiseNet",
        "task": "Signal reliability / denoising-related classification",
        "task_description": "Single-channel noise-related binary classification derived from EEGDenoiseNet.",
        "reference_url": "https://doi.org/10.1109/TBME.2020.3021172",
        "citation_text": "Zhang et al. EEGDenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising. IEEE TBME, 2021.",
    },
    "EEGMAT": {
        "display_name": "EEGMAT",
        "task": "Workload detection",
        "task_description": "Cognitive workload / task-state classification during mental arithmetic.",
        "reference_url": "https://doi.org/10.3390/data4010014",
        "citation_text": "Zyma et al. Electroencephalograms during Mental Arithmetic Task Performance. Data, 2019.",
    },
    "EEG_IO": {
        "display_name": "EEG-IO",
        "task": "Basic state identification",
        "task_description": "Eye-open vs. eye-closed / basic state classification.",
        "reference_url": "https://doi.org/10.1038/s41597-019-0194-7",
        "citation_text": "Agarwal et al. Open-source EEG dataset for eye-state / blink behavior analysis. Scientific Data, 2019.",
    },
    "EEG_Mortality_PD": {
        "display_name": "PD-Mortality",
        "task": "Neurodegenerative disease prognosis",
        "task_description": "Mortality vs. survival classification in Parkinson disease EEG.",
        "reference_url": "https://doi.org/10.18112/openneuro.ds007020.v1.0.0",
        "citation_text": "OpenNeuro ds007020: EEG Mortality in Parkinson disease.",
    },
    "EEG_SVRec": {
        "display_name": "EEG-SVRec",
        "task": "Affective / preference decoding",
        "task_description": "Affective or preference-related binary classification from steady-state visual recommendation stimuli.",
        "reference_url": "https://doi.org/10.1038/s41597-024-03344-4",
        "citation_text": "Zhang et al. EEG-SVRec: an EEG dataset for recommendation and affective decoding. Scientific Data, 2024.",
    },
    "ExoEEG_WalkStop": {
        "display_name": "EEG-Controlled Exoskeleton",
        "task": "Closed-loop assistive control",
        "task_description": "Walking vs. stopping control classification from EEG recorded during exoskeleton-assisted locomotion.",
        "reference_url": "https://openneuro.org/datasets/ds006940",
        "citation_text": "OpenNeuro ds006940. A multimodal neuroimaging dataset to study brain activity and gait during real-world walking with and without a lower-limb exoskeleton.",
    },
    "FACED_new": {
        "display_name": "FACED",
        "task": "Emotion recognition",
        "task_description": "Fine-grained emotion classification from video-elicited EEG.",
        "reference_url": "https://doi.org/10.1038/s41597-023-02894-7",
        "citation_text": "Chen et al. FACED: a large-scale multimodal EEG dataset for emotion recognition. Scientific Data, 2023.",
    },
    "HBN_EEG": {
        "display_name": "HBN-EEG",
        "task": "Cognitive task identification",
        "task_description": "Multi-context cognitive task classification.",
        "reference_url": "https://doi.org/10.1038/s41597-024-03347-1",
        "citation_text": "Shirazi et al. HBN-EEG: task and resting EEG from the Healthy Brain Network. Scientific Data, 2024.",
    },
    "HFO": {
        "display_name": "HFO",
        "task": "Epilepsy-related oscillation detection",
        "task_description": "High-frequency oscillation related binary classification.",
        "reference_url": "https://doi.org/10.13026/7j7r-8q57",
        "citation_text": "Frauscher et al. Open dataset for high-frequency oscillations in scalp EEG. PhysioNet / associated release.",
    },
    "HMC": {
        "display_name": "HMC",
        "task": "Sleep staging",
        "task_description": "Five-class sleep stage classification.",
        "reference_url": "https://physionet.org/content/hmc-sleep-staging/",
        "citation_text": "Alvarez-Estevez and Rijsman. Haaglanden Medisch Centrum sleep staging database. PhysioNet, 2022.",
    },
    "ISRUC_S1": {
        "display_name": "ISRUC-Sleep Subgroup I",
        "task": "Sleep staging",
        "task_description": "Five-class sleep stage classification.",
        "reference_url": "https://doi.org/10.1016/j.cmpb.2015.10.013",
        "citation_text": "Khalighi et al. ISRUC-Sleep: a comprehensive public dataset for sleep researchers. Computer Methods and Programs in Biomedicine, 2016.",
    },
    "ISRUC_S2": {
        "display_name": "ISRUC-Sleep Subgroup II",
        "task": "Sleep staging",
        "task_description": "Five-class sleep stage classification.",
        "reference_url": "https://doi.org/10.1016/j.cmpb.2015.10.013",
        "citation_text": "Khalighi et al. ISRUC-Sleep: a comprehensive public dataset for sleep researchers. Computer Methods and Programs in Biomedicine, 2016.",
    },
    "ISRUC_S3": {
        "display_name": "ISRUC-Sleep Subgroup III",
        "task": "Sleep staging",
        "task_description": "Five-class sleep stage classification.",
        "reference_url": "https://doi.org/10.1016/j.cmpb.2015.10.013",
        "citation_text": "Khalighi et al. ISRUC-Sleep: a comprehensive public dataset for sleep researchers. Computer Methods and Programs in Biomedicine, 2016.",
    },
    "LEMON_age": {
        "display_name": "MPI-LEMON-age",
        "task": "Biometrics",
        "task_description": "Age group classification derived from MPI-LEMON.",
        "reference_url": "https://doi.org/10.1038/sdata.2018.308",
        "citation_text": "Babayan et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 2019.",
    },
    "LEMON_gender": {
        "display_name": "MPI-LEMON-gender",
        "task": "Biometrics",
        "task_description": "Gender classification derived from MPI-LEMON.",
        "reference_url": "https://doi.org/10.1038/sdata.2018.308",
        "citation_text": "Babayan et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 2019.",
    },
    "LEMON_extraversion": {
        "display_name": "MPI-LEMON-extraversion",
        "task": "Biometrics",
        "task_description": "Extraversion classification derived from MPI-LEMON.",
        "reference_url": "https://doi.org/10.1038/sdata.2018.308",
        "citation_text": "Babayan et al. A mind-brain-body dataset of MRI, EEG, cognition, emotion, and peripheral physiology in young and old adults. Scientific Data, 2019.",
    },
    "Longitudinal_EEG_Reliability": {
        "display_name": "Longitudinal test-retest",
        "task": "Test-retest reliability",
        "task_description": "Cross-session subject identification / reliability classification.",
        "reference_url": "https://openneuro.org/datasets/ds006940",
        "citation_text": "Current processed benchmark derived from longitudinal repeated EEG recordings.",
    },
    "MDD": {
        "display_name": "MDD",
        "task": "Mental disorder classification",
        "task_description": "Healthy vs. major depressive disorder classification.",
        "reference_url": "https://doi.org/10.1371/journal.pone.0152489",
        "citation_text": "Mumtaz et al. An EEG-based machine learning method to screen depression. PLOS ONE, 2016.",
    },
    "MODMA": {
        "display_name": "MODMA",
        "task": "Mental disorder classification",
        "task_description": "Depression / patient vs. control classification.",
        "reference_url": "https://arxiv.org/abs/2002.09283",
        "citation_text": "Cai et al. MODMA dataset: a Multi-modal Open Dataset for Mental-disorder Analysis, 2020.",
    },
    "MonitoringErrP": {
        "display_name": "Monitoring ErrP",
        "task": "ErrP classification",
        "task_description": "Error-related potential classification.",
        "reference_url": "https://hal.science/hal-01055104",
        "citation_text": "Chavarriaga and Millán. Learning from EEG error-related potentials in noninvasive brain-computer interfaces.",
    },
    "MusicEEG": {
        "display_name": "MusicEEG",
        "task": "Emotion recognition",
        "task_description": "Music-evoked affect classification.",
        "reference_url": "https://openneuro.org/datasets/ds002721",
        "citation_text": "OpenNeuro ds002721 / associated release for music-related EEG affect experiments.",
    },
    "PD31": {
        "display_name": "PD31",
        "task": "Neurodegenerative disease classification",
        "task_description": "Healthy vs. Parkinson disease classification.",
        "reference_url": "https://openneuro.org/datasets/ds002778",
        "citation_text": "OpenNeuro ds002778: Parkinson disease EEG dataset.",
    },
    "PEARL_Neuro": {
        "display_name": "PEARL-Neuro",
        "task": "Cognitive task identification",
        "task_description": "Context / task discrimination classification.",
        "reference_url": "https://doi.org/10.1038/s41597-024-03142-y",
        "citation_text": "Dzianok et al. PEARL-Neuro: a large-scale EEG dataset for task and context decoding. Scientific Data, 2024.",
    },
    "PerceiveImagine": {
        "display_name": "PerceiveImagine",
        "task": "Cognitive task identification",
        "task_description": "Perceiving vs. imagining discrimination.",
        "reference_url": "https://openneuro.org/datasets/ds005697",
        "citation_text": "OpenNeuro ds005697: PerceiveImagine dataset.",
    },
    "Physionet_MI": {
        "display_name": "PhysioNet-MI",
        "task": "Motor imagery classification",
        "task_description": "Four-class motor imagery classification.",
        "reference_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "citation_text": "Schalk et al. BCI2000: a general-purpose brain-computer interface system. IEEE TBME, 2004. EEG Motor Movement/Imagery Dataset on PhysioNet.",
    },
    "RestCog": {
        "display_name": "RestCog",
        "task": "Cognitive task identification",
        "task_description": "Task-type classification across multiple resting and cognitive conditions.",
        "reference_url": "https://openneuro.org/datasets/ds002721",
        "citation_text": "OpenNeuro ds002721: RestCog dataset release.",
    },
    "SEED": {
        "display_name": "SEED",
        "task": "Emotion recognition",
        "task_description": "Positive / negative / neutral emotion classification.",
        "reference_url": "https://bcmi.sjtu.edu.cn/home/seed/",
        "citation_text": "Duan et al. Differential entropy feature for EEG-based emotion classification and the SEED dataset.",
    },
    "SEEDIV": {
        "display_name": "SEED-IV",
        "task": "Emotion recognition",
        "task_description": "Four-class emotion classification.",
        "reference_url": "https://bcmi.sjtu.edu.cn/home/seed/",
        "citation_text": "Zheng et al. EmotionMeter: a multimodal framework for recognizing human emotions.",
    },
    "SEED_FRA": {
        "display_name": "SEED-FRA",
        "task": "Emotion recognition",
        "task_description": "Three-class emotion classification with French movie stimuli.",
        "reference_url": "https://doi.org/10.1109/EMBC.2015.7319296",
        "citation_text": "Imtiaz et al. Open access dataset for emotion recognition with film stimuli, 2015.",
    },
    "SEED_V": {
        "display_name": "SEED-V",
        "task": "Emotion recognition",
        "task_description": "Five-class emotion classification.",
        "reference_url": "https://bcmi.sjtu.edu.cn/home/seed/",
        "citation_text": "Liu et al. Comparing recognition performance of discrete and dimensional emotion models on EEG data.",
    },
    "SEED_VIG": {
        "display_name": "SEED-VIG",
        "task": "Vigilance detection",
        "task_description": "Vigilance state classification.",
        "reference_url": "https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html",
        "citation_text": "Zheng and Lu. Investigating critical frequency bands and channels for EEG-based vigilance estimation.",
    },
    "SEED_VII": {
        "display_name": "SEED-VII",
        "task": "Emotion recognition",
        "task_description": "Seven-class emotion classification.",
        "reference_url": "https://bcmi.sjtu.edu.cn/home/seed/",
        "citation_text": "Jiang et al. SEED-VII dataset release.",
    },
    "SHU_MI": {
        "display_name": "SHU-MI",
        "task": "Motor imagery classification",
        "task_description": "Two-class motor imagery classification.",
        "reference_url": "https://doi.org/10.1038/s41597-022-01583-9",
        "citation_text": "Ma et al. A large EEG dataset for motor imagery. Scientific Data, 2022.",
    },
    "Siena_EEG": {
        "display_name": "Siena EEG",
        "task": "Epilepsy and abnormalities",
        "task_description": "Seizure-related binary classification.",
        "reference_url": "https://physionet.org/content/siena-scalp-eeg/1.0.0/",
        "citation_text": "Detti et al. Siena Scalp EEG Database. PhysioNet, 2020.",
    },
    "SleepEDF_full": {
        "display_name": "Sleep-EDF",
        "task": "Sleep staging",
        "task_description": "Five-class sleep stage classification.",
        "reference_url": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "citation_text": "Kemp et al. The Sleep-EDF Database expanded. PhysioNet.",
    },
    "SSVEP": {
        "display_name": "SSVEP-9-ch",
        "task": "SSVEP target classification",
        "task_description": "Nine-channel SSVEP target classification.",
        "reference_url": "https://doi.org/10.1038/s41597-020-00695-z",
        "citation_text": "Public SSVEP benchmark release.",
    },
    "TDBRAIN": {
        "display_name": "TDBRAIN",
        "task": "Mental disorder classification",
        "task_description": "Psychiatric phenotype classification.",
        "reference_url": "https://doi.org/10.1038/s41597-022-01459-y",
        "citation_text": "van Dijk et al. TDBRAIN: a large-scale neurophysiological dataset for psychiatry. Scientific Data, 2022.",
    },
    "Things_EEG2": {
        "display_name": "Things-EEG2",
        "task": "Visual concept decoding",
        "task_description": "Animate vs. inanimate concept classification.",
        "reference_url": "https://doi.org/10.1038/s41597-022-01808-1",
        "citation_text": "Gifford et al. THINGS-EEG2: a large-scale EEG dataset for object concept recognition. Scientific Data, 2022.",
    },
    "TUAB": {
        "display_name": "TUAB",
        "task": "Epilepsy and abnormalities",
        "task_description": "Clinical normal vs. abnormal EEG classification.",
        "reference_url": "https://doi.org/10.3389/fnins.2016.00196",
        "citation_text": "Obeid and Picone. The Temple University Hospital EEG Data Corpus. Frontiers in Neuroscience, 2016.",
    },
    "TUEP": {
        "display_name": "TUEP",
        "task": "Epilepsy and abnormalities",
        "task_description": "Seizure-related binary classification.",
        "reference_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml",
        "citation_text": "NEDC Temple University EEG corpus downloads page for seizure-related subsets.",
    },
    "TUEV": {
        "display_name": "TUEV",
        "task": "Epilepsy and abnormalities",
        "task_description": "EEG event classification.",
        "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4874511/",
        "citation_text": "Harati et al. Improved EEG Event Classification Using Differential Energy. IEEE SPMB, 2015.",
    },
    "TUSL": {
        "display_name": "TUSL",
        "task": "Epilepsy and abnormalities",
        "task_description": "Sleep slowing / sleep-state related classification.",
        "reference_url": "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml",
        "citation_text": "NEDC Temple University EEG corpus downloads page for slowing-related subsets.",
    },
    "Workload": {
        "display_name": "Workload",
        "task": "Workload detection",
        "task_description": "Cognitive workload level classification.",
        "reference_url": "https://www.nature.com/articles/s41597-022-01898-y",
        "citation_text": "Hinss et al. Open multi-session and multi-task EEG cognitive Dataset for passive brain-computer Interface Applications. Scientific Data, 2023.",
    },
}


TEXT_MIME_BY_SUFFIX = {
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
    ".md": "text/markdown",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def maybe_load_json(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def infer_base_name(name: str) -> str:
    suffixes = [
        "_wsn",
        "_old_badscale",
        "_old_60cls",
        "_mono",
        "_128",
        "_10s",
        "_unhaunted",
    ]
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def infer_channel_names(config: dict[str, Any], sample_df: pd.DataFrame) -> list[str]:
    channel_names = config.get("channel_names")
    if isinstance(channel_names, list) and channel_names:
        return channel_names
    if "channel_names_json" in sample_df.columns and len(sample_df):
        raw = sample_df["channel_names_json"].dropna()
        if not raw.empty:
            try:
                return json.loads(raw.iloc[0])
            except Exception:
                return []
    return []


def infer_split_summary(sample_df: pd.DataFrame) -> str:
    if "split" not in sample_df.columns:
        return "No per-sample split column is present in sample_index.parquet."
    split_series = sample_df["split"].dropna()
    if split_series.empty:
        return "A split column exists, but no split values are populated."
    counts = split_series.value_counts().to_dict()
    ordered = ", ".join(f"{k}: {v}" for k, v in counts.items())
    return f"Per-sample split assignments recorded in sample_index.parquet: {ordered}."


def build_preprocessing_text(config: dict[str, Any], conv: dict[str, Any], valid: dict[str, Any]) -> str:
    parts: list[str] = []
    source_format = config.get("source_format")
    if source_format:
        parts.append(f"source format {source_format}")
    sr = valid.get("sampling_rate") or conv.get("sampling_rate") or config.get("target_sampling_rate")
    if sr:
        parts.append(f"resampled/validated at {sr} Hz")
    seg = config.get("segment_seconds")
    if seg:
        parts.append(f"segmented into {seg} second windows")
    low = config.get("bandpass_low")
    high = config.get("bandpass_high")
    if low is not None or high is not None:
        parts.append(f"band-pass filtered ({low}–{high} Hz)")
    notch = config.get("notch_hz")
    if notch:
        parts.append(f"notch filtered at {notch} Hz")
    norm = valid.get("normalization") or conv.get("normalization")
    if norm:
        parts.append(f"normalization: {norm}")
    qc_flags = config.get("qc_flags")
    if qc_flags:
        if isinstance(qc_flags, dict):
            qc_desc = ", ".join(f"{k}={v}" for k, v in qc_flags.items())
        else:
            qc_desc = str(qc_flags)
        parts.append(f"QC flags available ({qc_desc})")
    if not parts:
        return "Preprocessing metadata were not fully specified in the local dataset JSON files."
    return "Signals were processed as follows: " + "; ".join(parts) + "."


def build_dataset_description(
    dataset_name: str,
    display_name: str,
    override: dict[str, Any],
    config: dict[str, Any],
    conv: dict[str, Any],
    valid: dict[str, Any],
    channel_names: list[str],
    split_summary: str,
) -> str:
    task_desc = override.get("task_description", "EEG segment classification dataset.")
    n_subjects = conv.get("n_subjects", "unknown")
    n_samples = conv.get("n_samples", valid.get("sample_index_rows", "unknown"))
    sr = valid.get("sampling_rate") or conv.get("sampling_rate") or config.get("target_sampling_rate")
    seg = config.get("segment_seconds")
    n_channels = valid.get("signals_shape", [None, len(channel_names), None])[1] if valid.get("signals_shape") else len(channel_names)
    return (
        f"{display_name} ({dataset_name}) is a processed EEG benchmark dataset. "
        f"It contains {n_samples} segment-level samples from {n_subjects} subjects. "
        f"The mounted release uses {n_channels} channels, a target sampling rate of {sr} Hz, "
        f"and a window length of {seg} seconds. {task_desc} {split_summary}"
    )


def build_field(name: str, description: str, data_type: str) -> dict[str, Any]:
    return {
        "@type": "cr:Field",
        "@id": name,
        "name": name,
        "description": description,
        "dataType": data_type,
        "source": {
            "fileObject": {"@id": "sample_index"},
            "extract": {"column": name},
        },
    }


def build_record_fields(sample_df: pd.DataFrame) -> list[dict[str, Any]]:
    preferred = [
        ("sample_id", "Unique segment identifier.", "sc:Text"),
        ("subject_id", "Subject identifier.", "sc:Text"),
        ("session_id", "Session identifier.", "sc:Text"),
        ("trial_id", "Trial identifier when available.", "sc:Text"),
        ("segment_id", "Segment identifier.", "sc:Text"),
        ("segment_start_sec", "Window start time in seconds.", "sc:Float"),
        ("segment_end_sec", "Window end time in seconds.", "sc:Float"),
        ("label_id", "Numeric class identifier.", "sc:Integer"),
        ("label_raw", "Human-readable label value when available.", "sc:Text"),
        ("label_value", "Original or alternative numeric label value.", "sc:Float"),
        ("split", "Data split assignment when available.", "sc:Text"),
        ("source_relpath", "Relative path to the source recording.", "sc:Text"),
        ("channel_count", "Number of channels in the sample.", "sc:Integer"),
        ("channel_names_json", "JSON-encoded channel name list for the sample.", "sc:Text"),
        ("raw_sfreq", "Original recording sampling frequency in Hz.", "sc:Float"),
        ("valid_time_samples", "Number of valid samples in the stored time axis.", "sc:Integer"),
        ("qc_flags", "Quality-control bitmask.", "sc:Integer"),
    ]
    fields: list[dict[str, Any]] = []
    cols = set(sample_df.columns)
    for name, desc, dtype in preferred:
        if name in cols:
            fields.append(build_field(name, desc, dtype))
    if not fields:
        first_col = sample_df.columns[0]
        fields.append(build_field(first_col, f"Column extracted from sample_index.parquet: {first_col}.", "sc:Text"))
    return fields


def build_distribution(dataset_dir: Path, dataset_name: str) -> list[dict[str, Any]]:
    distribution: list[dict[str, Any]] = []
    file_names = [
        "sample_index.parquet",
        "dataset_config.json",
        "conversion_summary.json",
        "validation_summary.json",
        "label_vocab.json",
    ]
    for file_name in file_names:
        file_path = dataset_dir / file_name
        if not file_path.exists():
            continue
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": file_name.rsplit(".", 1)[0],
                "name": file_name,
                "contentUrl": file_name,
                "encodingFormat": TEXT_MIME_BY_SUFFIX.get(file_path.suffix, "application/octet-stream"),
            }
        )
    zarr_path = dataset_dir / f"{dataset_name}.zarr"
    if zarr_path.exists():
        distribution.append(
            {
                "@type": "cr:FileSet",
                "@id": "zarr_store",
                "name": f"{dataset_name}.zarr",
                "description": "Zarr store containing EEG arrays and auxiliary arrays.",
                "includes": f"{dataset_name}.zarr/**",
                "encodingFormat": "application/octet-stream",
            }
        )
    return distribution


def build_croissant(dataset_dir: Path, public_base_url: str | None) -> dict[str, Any]:
    dataset_name = dataset_dir.name
    base_name = infer_base_name(dataset_name)
    override = DATASET_OVERRIDES.get(dataset_name, DATASET_OVERRIDES.get(base_name, {}))

    config = maybe_load_json(dataset_dir / "dataset_config.json")
    conv = maybe_load_json(dataset_dir / "conversion_summary.json")
    valid = maybe_load_json(dataset_dir / "validation_summary.json")
    vocab = maybe_load_json(dataset_dir / "label_vocab.json")
    sample_df = pd.read_parquet(dataset_dir / "sample_index.parquet")

    channel_names = infer_channel_names(config, sample_df)
    split_summary = infer_split_summary(sample_df)
    preprocessing_text = build_preprocessing_text(config, conv, valid)
    display_name = override.get("display_name", dataset_name)

    distribution = build_distribution(dataset_dir, dataset_name)
    record_fields = build_record_fields(sample_df)

    signal_shape = valid.get("signals_shape") or conv.get("shape")
    n_channels = signal_shape[1] if isinstance(signal_shape, list) and len(signal_shape) > 1 else len(channel_names)
    seq_len = signal_shape[2] if isinstance(signal_shape, list) and len(signal_shape) > 2 else None

    content_url = f"{public_base_url.rstrip('/')}/{dataset_name}" if public_base_url else dataset_name
    source_root = config.get("source_root", "")

    top_level: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": display_name,
        "description": build_dataset_description(
            dataset_name,
            display_name,
            override,
            config,
            conv,
            valid,
            channel_names,
            split_summary,
        ),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": content_url,
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "distribution": distribution,
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "samples",
                "name": "samples",
                "description": "Segment-level sample index extracted from sample_index.parquet.",
                "field": record_fields,
            }
        ],
        "keywords": [
            "EEG",
            "electroencephalography",
            override.get("task", "classification"),
            f"{n_channels}_channels",
            f"{config.get('segment_seconds', 'unknown')}_seconds",
        ],
        "citation": override.get("citation_text"),
        "sameAs": override.get("reference_url"),
        "variableMeasured": [
            "EEG signals",
            "segment label",
            "subject identifier",
            "session identifier",
        ],
        "channelCount": n_channels,
        "sampleRateHz": valid.get("sampling_rate") or conv.get("sampling_rate") or config.get("target_sampling_rate"),
        "windowLengthSeconds": config.get("segment_seconds"),
        "timepointsPerSample": seq_len,
        "labelVocab": vocab,
        "channelNames": channel_names,
        "splitSummary": split_summary,
        "prov:wasDerivedFrom": [source_root] if source_root else [],
        "prov:wasGeneratedBy": [
            {
                "@type": "sc:CreateAction",
                "name": "Preprocessing and segmentation",
                "description": preprocessing_text,
            },
            {
                "@type": "sc:UpdateAction",
                "name": "Split and sample indexing",
                "description": split_summary,
            },
        ],
        "rai:dataLimitations": (
            "This is a processed segment-level EEG release derived from an original dataset. "
            "Its applicability is limited by the original cohort composition, acquisition protocol, "
            "labeling quality, and the preprocessing choices recorded in the attached JSON files. "
            "Use is not recommended outside the task and population implied by the source dataset."
        ),
        "rai:dataBiases": (
            "Potential biases may arise from cohort imbalance, recording-site effects, label imbalance, "
            "and preprocessing-induced selection effects. Users should inspect label counts, subject counts, "
            "QC flags, and source-study documentation before model development."
        ),
        "rai:personalSensitiveInformation": (
            "This processed release stores segment-level EEG arrays and task labels. "
            "Depending on the source dataset, the original data may involve sensitive health, demographic, "
            "or behavioral information. Source-dataset access controls and usage restrictions still apply."
        ),
        "rai:dataUseCases": (
            f"Primary use case: {override.get('task_description', 'EEG benchmark modeling')}. "
            "Suitable for representation learning, classification benchmarking, and reproducible preprocessing audits "
            "within the domain described by the original dataset."
        ),
        "rai:dataSocialImpact": (
            "Potential benefits include reproducible EEG benchmarking and improved methodological comparison. "
            "Potential harms include misuse for unsupported clinical or high-stakes decisions, propagation of cohort bias, "
            "and overclaiming beyond the source population."
        ),
        "rai:hasSyntheticData": False,
    }
    return top_level


def should_process(dataset_dir: Path) -> bool:
    required = ["dataset_config.json", "sample_index.parquet"]
    return dataset_dir.is_dir() and all((dataset_dir / name).exists() for name in required)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Croissant metadata for EEG dataset directories.")
    parser.add_argument("root", type=Path, help="Root directory containing per-dataset subdirectories.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional separate output root. If omitted, writes croissant.json into each dataset directory.",
    )
    parser.add_argument(
        "--public-base-url",
        type=str,
        default=None,
        help="Optional public base URL prepended to dataset directory names.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional explicit dataset directory names to process.",
    )
    args = parser.parse_args()

    root = args.root
    dataset_names = args.datasets or sorted(p.name for p in root.iterdir() if should_process(p))

    processed = 0
    for dataset_name in dataset_names:
        dataset_dir = root / dataset_name
        if not should_process(dataset_dir):
            continue
        croissant = build_croissant(dataset_dir, args.public_base_url)
        target_dir = (args.output_root / dataset_name) if args.output_root else dataset_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "croissant.json"
        target_path.write_text(json.dumps(croissant, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote {target_path}")
        processed += 1
    print(f"Generated {processed} croissant files.")


if __name__ == "__main__":
    main()
