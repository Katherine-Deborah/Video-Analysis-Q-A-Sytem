# 🎥 Video Analysis QA System

An intelligent video analysis system that extracts insights from videos through automated captioning, audio transcription, and natural language question-answering capabilities.

## ✨ Features

- **Video Processing**: Upload videos or capture directly from webcam
- **Frame Analysis**: Automatic extraction and intelligent captioning of video frames
- **Audio Transcription**: Speech-to-text conversion using advanced AI models
- **Question Answering**: Natural language queries about video content
- **Session Management**: Organize and revisit previous video analyses
- **Real-time Webcam**: Live video capture and processing

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository
2. Install dependencies
3. Run the application:

```bash
python main.py
```

## 🏗️ Architecture

The system consists of several modular components:

- **`main.py`**: Gradio web interface and application orchestration
- **`models.py`**: AI model loading and initialization with caching
- **`processing.py`**: Video processing pipeline coordinator
- **`captions.py`**: Frame extraction and image captioning
- **`audio.py`**: Audio extraction and transcription
- **`QA.py`**: Question-answering and context retrieval

## 🤖 AI Models Used

### Image Captioning
- **Model**: [QuadrantTechnologies/qhub-blip-image-captioning-finetuned](https://huggingface.co/quadranttechnologies/qhub-blip-image-captioning-finetuned)
- **Purpose**: Generate descriptive captions for video frames

### Audio Transcription
- **Model**: [OpenAI/whisper-medium](https://huggingface.co/openai/whisper-medium)
- **Purpose**: Convert speech to text from video audio tracks

### Question Answering
- **Model**: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- **Purpose**: Answer natural language questions about video content

## 📱 Usage

### Video Input Options

1. **File Upload**: Support for MP4, AVI, MOV, MKV formats
2. **Webcam Capture**: Real-time recording with customizable duration and FPS

### Analysis Process

1. **Frame Extraction**: Automatically samples frames at specified intervals
2. **Caption Generation**: Creates descriptive text for each frame
3. **Audio Processing**: Extracts and transcribes speech content
4. **Database Storage**: Stores results for persistent access

### Question Answering

Ask natural language questions about your videos:
- "What objects were visible in the video?"
- "What was the person doing?"
- "What did someone say about [topic]?"

## 💾 Data Management

- **SQLite Database**: Stores captions, transcriptions, and session data
- **Session System**: Organize analyses by unique session IDs
- **Persistent Storage**: Access previous analyses anytime

## 🛠️ Technical Details

### Video Processing
- Configurable frame sampling intervals
- Multi-format video support
- Real-time webcam integration

### AI Pipeline
- GPU acceleration when available
- Efficient model caching with Gradio
- Batch processing for improved performance

### Database Schema
- `video_sessions`: Session metadata
- `captions`: Frame-level descriptions with timestamps
- `transcriptions`: Full audio transcripts per session

## 🔧 Configuration

### Webcam Settings
- Adjustable recording duration (3-30 seconds)
- Configurable frame rate (1-10 FPS)
- Real-time preview and progress tracking

### Processing Parameters
- Frame extraction interval (default: 0.5 seconds)
- Caption generation limits
- Audio sampling rate (16kHz for Whisper compatibility)

## 🚨 System Requirements

- **Python 3.8+**
- **CUDA-compatible GPU** (optional, for faster processing)
- **Webcam** (for live capture functionality)
- **FFmpeg** (for video processing)

## 🤝 Contributing

This system is modular and extensible. Key areas for enhancement:
- Additional video formats
- More sophisticated AI models
- Advanced question types
- Export capabilities

## 📄 License

Open source project - see individual model licenses for AI components.

---

*Built with Gradio, PyTorch, and Transformers for seamless video intelligence.*
