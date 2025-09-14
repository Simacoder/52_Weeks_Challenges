# 🌍 Language Detection with Python

Most industries dealing with textual data focus on using digital capabilities because it is the fastest way of processing documents.  
At an international level, it is often beneficial to **automatically identify the underlying language of a document** before any further processing.  

A simple use case could be for a company to detect the language of incoming textual information in order to route it to the relevant department for processing.  

This project provides a **practical overview and implementation** of language detection using Python, focusing on popular libraries that make this task easy and efficient.

---

## 📌 Project Overview

This project explores **four Python libraries** that can be used for text language detection.  
We implement and demonstrate how these libraries can be used to detect:

- **Single language detection** → identifying the most likely language of a text.
- **Multiple language detection** → identifying the top N most frequent languages in a text (useful for mixed-language content).

Currently, this repository includes an implementation using **[gcld3](https://pypi.org/project/gcld3/)** (Google’s Compact Language Detector v3).

---

## 🚀 Features

- 🔍 **Single Language Detection** – get the most likely language and confidence score.
- 🌐 **Multiple Language Detection** – detect top N languages in multilingual text.
- ⚡ **Lightweight & Fast** – uses Google's neural network-based language identifier.

---

## 📂 Installation

### 1. Install Protocol Buffers Compiler (Required for gcld3)

#### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install -y protobuf-compiler
protoc --version
```

# Windows

Download protoc from protobuf releases
.
```bash
Extract it and add the bin/ folder to your PATH.

Run protoc --version to confirm installation.
```
2. Install Python Dependencies
```bash
pip install gcld3
```
# 📝 Usage

```bash
import gcld3

# Single language detection
def cld3_single_language_detection(text):
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                            max_num_bytes=len(text))
    result = detector.FindLanguage(text=text)
    return {
        "language": result.language,
        "probability": result.probability
    }

# Multiple language detection
def cld3_multiple_language_detection(text, nb_language=2):
    detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                            max_num_bytes=len(text))
    languages = detector.FindTopNMostFreqLangs(text=text, num_langs=nb_language)
    return [{"language": l.language, "probability": l.probability} for l in languages]
```
# Example

```bash
english_text = "This is a simple English sentence."
print(cld3_single_language_detection(english_text))
```
# 📊 Example Output

```bash
{'language': 'en', 'probability': 0.9999971389770508}

```
# 📚 Planned Libraries to Explore

This repository will also cover:

- langdetect – Google language-detection library (pure Python).

- langid.py – Language identification tool with pre-trained models.

- fastText – Facebook AI’s text classification tool that can detect languages.

# 🤝 Contributing

Pull requests are welcome!
If you have suggestions for additional libraries, performance improvements, or better examples, feel free to open an issue or submit a PR.

# 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.

# 🙌 Acknowledgements
```bash
gcld3
 – Google's Compact Language Detector v3

Protocol Buffers
 – for serialization and parsing
 ```

 # AUTHOR
 - Simanga Mchunu