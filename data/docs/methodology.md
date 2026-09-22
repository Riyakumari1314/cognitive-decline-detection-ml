# Proposed Methodology

## 1. Research Objective

This study investigates whether machine learning can identify
cognitive-decline-related patterns from speech and typing behaviour.

Three experimental settings are considered:

1. Speech-only
2. Typing-only
3. Speech + Typing multimodal analysis

## 2. Overall Pipeline

```text
Raw Data
   |
   +-------------------+
   |                   |
   v                   v
Speech Data        Typing Data
   |                   |
   v                   v
Preprocessing      Preprocessing
   |                   |
   v                   v
Speech Features   Typing Features
   |                   |
   v                   v
Standardization   Standardization
   |                   |
   +---------+---------+
             |
             v
       Feature Fusion
             |
             v
    Multimodal Classifier
             |
             v
      Cognitive Status
