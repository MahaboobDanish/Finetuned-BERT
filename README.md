# Build AI Phishing Email Alert Web App with BERT and Granite

#### Introduction

This project, LLM Email Guard, bridges that gap. `Fine-tune BERT`, a transformer model for `natural language processing (NLP)`, to detect phishing emails with high accuracy, then pair it with a powerful `enterprise LLM (like Granite or ChatGPT)` that can clearly explain the reasoning behind each decision in plain language. Finally, you’ll package it all into an easy-to-use `Gradio web app` that anyone can try.

#### Objectives 
- Key components and architecture of BERT and why it is suitable for classification fine-tuning.
- Fine-tune a BERT-based model for classificaiton tasks using the HuggingFace libraries.
- Leverage a LL to provide human-readable explanations for why emails are classified as suspicious.
- build and share a simple Gradio Web Application that integrates the fine-tuned phishing email detector with an LLM-based agent.
### Background
##### What is BERT? What's its difference from GPT?

BERT (Bidirectional Encoder Representations from Transformers) is a transformer model designed to understand text by looking at words in context from both the left and the right. 

This bidirectional property makes BERT especially effective at capturing sentence- and paragraph-level meaning through representative numerical embeddings, enabling it to be easily adapted to many downstream tasks such as text classification. 

In detail, for classification task, a new learnable Multi-Layer Perceptron (MLP) with a softmax output layer is attached to the last layer of the BERT model.

During the training, the BERT as well as the newly added layers will be fine-tune based on the new datasets.

In contrast, GPT models (e.g. OpenAI's ChatGPT) are unidirectional, reading text only from left to right and predicting the next word in sequence.

Because GPT does not have access to the full input context during training, it is less well-suited for classification tasks. 

For this reason, BERT is often the preferred choice for fine-tuning on downstream tasks that require deep text understanding.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/TZU0xfrskSllMPVnn__ONA/Screenshot%202025-09-16%20at%203-31-14%E2%80%AFPM.png" alt="drawing" width="70%" style="display: block; margin: 0 auto;"/>

#### Prepare the PhishingEmailDetectionv2.0 Dataset

To start fine-tuning a language model for phishing email detection, we must prepare a valid dataset. We use an existing dataset PhishingEmailDetectionv2.0 (https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0) available on huggingface. Here are some descriptive statistics of the dataset:

Before training, it’s important to understand the data we are working with. Below are some descriptive statistics of the dataset:

- Number of samples: 22,644 Emails and 177,356 URLs
- Classes: Legitimate Email (0), Phishing Email (1), - Legitimate URL (2), Phishing URL (3)
- Class distribution: 5.7% (0), 5.6% (1), 44.3% (2), 44.5% (3)

These statistics provide a quick overview of the dataset’s balance, size, and text characteristics. First, we note that the dataset contains multiple types of content; since our focus is on email text, we will extract only the email-related data. Second, the dataset is fairly balanced between the two classes, providing a strong foundation for effective model training.

#### Why LLM for Explaining Phishing Context?

A Large Language Model (LLM) is an advanced machine learning model trained on massive amounts of text data to understand, generate, and reason about human language.

Unlike traditional classifiers that only output “spam” vs. “ham,” LLMs bring contextual understanding, reasoning, and explainability to text analysis.

In the context of phishing email detection, an LLM is useful because:

1. `Suspicious Keyword Extraction`: LLMs can identify and highlight words or phrases in an email that are typically associated with phishing attempts (e.g., “verify your account”, “urgent”, “click here”). This makes the system more transparent by showing why an email was flagged, not just the prediction.

2. `Explainability & User Trust`: Instead of a black-box prediction, the LLM can summarize in plain English why a message looks suspicious (e.g., “The sender domain is unusual and the message creates urgency”). This improves trust and adoption in cybersecurity tools.

3. `Adaptive Detection`: Attackers constantly change wording to bypass detection. LLMs, because of their broad training data and ability to generalize, can detect new phishing strategies beyond the training distribution of a small classifier, effectively acting as an additional security layer that catches threats which slip past the primary model.

### Environment Setup

- `torch`: for processing tensor data and essential deep learning tools for AI training.
- `transformers`: for various open-sorced state-of-art language and multimodal models.
- `datasets`: for a wide range of structured datasets for NLP and vision tasks.
- `accelerate`: for tools that optimize and simplify the AI trianing process.
- `ibm_watson_ai`: for IBM Watson.ai APIs and services.
- `matplotlib`: for additional plotting tools.

- 
