# 🎙️ Typhoon ASR Desktop Transcriber

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://astral.sh/uv)

**Typhoon ASR Desktop Transcriber** is a lightweight, cross-platform desktop application for real-time and offline Thai speech-to-text transcription powered by **Typhoon ASR (FastConformer)** and **ONNX Runtime**. 

Built with **PySide6 (Qt for Python)**, it operates **100% offline** without heavy PyTorch or NeMo runtime dependencies on client machines, maintaining a strict RAM footprint under **500 MB**.

---

## 📖 สารบัญ (Table of Contents)

1. [✨ ฟีเจอร์เด่น (Key Features)](#-key-features)
2. [💻 ความต้องการของระบบและฮาร์ดแวร์ที่รองรับ (System Requirements & Hardware Compatibility)](#-ความต้องการของระบบและฮาร์ดแวร์ที่รองรับ-system-requirements--hardware-compatibility)
   - [2.1 ตารางเปรียบเทียบสเปกระบบขั้นต่ำและสเปกที่แนะนำ (System Requirements Matrix)](#21-ตารางเปรียบเทียบสเปกระบบขั้นต่ำและสเปกที่แนะนำ-system-requirements-matrix)
   - [2.2 รายละเอียดฮาร์ดแวร์ที่รองรับ (Hardware Compatibility Breakdown)](#22-รายละเอียดฮาร์ดแวร์ที่รองรับ-hardware-compatibility-breakdown)
3. [🛠️ สิ่งที่ต้องเตรียมก่อนใช้งานใน Linux & โปรแกรมเสริม (System Prerequisites)](#️-prerequisites-สิ่งที่ต้องเตรียมใน-linux--โปรแกรมเสริม)
4. [🚀 วิธีติดตั้ง Virtual Environment และการใช้งาน (Installation via uv)](#-installation--virtual-environment-วิธีสร้าง-venv)
5. [⚡ คำแนะนำการเลือกฮาร์ดแวร์ประมวลผล (Hardware Selection: CPU vs GPU รุ่นต่างๆ)](#-hardware-selection-คำแนะนำการเลือกใช้-cpu-vs-gpu-รุ่นต่างๆ)
   - [5.1 โหมด CPU (สำหรับคอมพิวเตอร์ทั่วไป / ประหยัด RAM)](#51-โหมด-cpu-สำหรับคอมพิวเตอร์ทั่วไป--ประหยัด-ram-500-mb)
   - [5.2 โหมด GPU สำหรับ NVIDIA Pascal (GTX 1050, 1060, 1070, 1080)](#52-โหมด-gpu-สำหรับ-nvidia-pascal-gtx-1050-1060-1070-1080--sm_61)
   - [5.3 โหมด GPU สำหรับ NVIDIA รุ่นใหม่ (RTX 20xx, 30xx, 40xx, 50xx)](#53-โหมด-gpu-สำหรับ-nvidia-รุ่นใหม่-rtx-20xx-30xx-40xx-50xx--sm_75-sm_86-sm_89-sm_120)
   - [5.4 โหมด GPU สำหรับ AMD Radeon & Intel Arc](#54-โหมด-gpu-สำหรับ-amd-radeon--intel-arc-windows-directml--linux)
6. [📥 ระบบตรวจสอบและดาวน์โหลดโมเดลอัตโนมัติ (Automated Model Setup)](#-automated-model-setup-ระบบตรวจสอบและดาวน์โหลดโมเดลอัตโนมัติ)
7. [💼 วิธีจัดทำ Portable App (Copy ไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับแอป)](#-วิธีจัดทำ-portable-app-copy-ไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับแอป)
8. [🔎 ลำดับ Model Search Path (การค้นหาโมเดลของระบบ)](#-ลำดับ-model-search-path-การค้นหาโมเดลของระบบ)
9. [📂 ระบบ Batch Conversion & Destination Popup](#-ระบบ-batch-conversion--destination-popup)
10. [🤖 คำแนะนำการปรับปรุงข้อความด้วย AI (AI Post-Processing)](#-คำแนะนำการปรับปรุงข้อความด้วย-ai-ai-post-processing)
11. [⌨️ คีย์ลัด (Keyboard Shortcuts)](#️-keyboard-shortcuts)
12. [🧪 การรันชุดทดสอบอัตโนมัติ (Running Automated Tests)](#-running-automated-tests)
13. [📁 โครงสร้างโปรเจกต์ (Repository Structure)](#-repository-structure)
14. [📄 สัญญาอนุญาต (License)](#-license)

---

## ✨ Key Features

- 🎤 **Real-Time Live Streaming Transcription:** ถอดเสียงสดความหน่วงต่ำจากไมโครโฟนหรือเสียงภายในคอมพิวเตอร์ (WASAPI Loopback บน Windows, PulseAudio/PipeWire monitor บน Linux)
- 🧠 **Pure ONNX Runtime Inference:** ถอดรหัสด้วย FastConformer Dual RNN-T และ SentencePiece Tokenizer โดยไม่จำเป็นต้องใช้ PyTorch หรือ NeMo
- ⚡ **Cross-Platform Hardware Acceleration (CPU / GPU Auto-Detect):** รองรับทั้งโหมด CPU ประหยัดพลังงาน และโหมด GPU เร่งความเร็ว (CUDA, DirectML, ROCm, OpenVINO) พร้อมระบบ Dynamic Preload และ Automatic Fallback ป้องกันโปรแกรมแครช
- 📥 **Automated Zero-Click Model Downloader:** มีระบบตรวจจับไฟล์โมเดลในเครื่องอัตโนมัติเมื่อเปิดโปรแกรม หากยังไม่มีไฟล์ ระบบจะถามและเริ่มดาวน์โหลดให้ทันที ไม่จำเป็นต้องค้นหาปุ่มดาวน์โหลดเอง
- 🔇 **Voice Activity Detection (VAD):** ตัดช่วงเสียงเงียบอัตโนมัติ เพื่อประหยัดการประมวลผลและลดปัญหาตัวอักษรหลอน (Hallucination)
- 🎚️ **Live Audio Level (VU Meter):** แถบแสดงระดับความดังเสียงแบบเรียลไทม์ พร้อมสีแสดงสถานะ (เขียว/เหลือง/แดง)
- 📂 **Tabbed UI with Batch Table:** แยกแท็บการทำงานระหว่าง Live Streaming และ Batch Transcription พร้อมระบบตารางจัดการคิวแปลงไฟล์
- 🗂️ **Batch Drag & Drop & Hierarchy Mirroring:** ลากไฟล์หรือทั้งโฟลเดอร์มาวางลงในตารางได้ทันที พร้อมหน้าต่าง Popup กำหนดโฟลเดอร์ปลายทางและเลือกฟอร์แมต (.txt, .srt, .vtt, .json) ได้อย่างสะดวก และจำลองโครงสร้างโฟลเดอร์ย่อย (Subfolder Hierarchy Mirroring) ให้อัตโนมัติ
- 💾 **Multi-Format Subtitle Export:** ส่งออกผลลัพธ์ได้ทั้ง **Plain Text (`.txt`)**, **SubRip Subtitles (`.srt`)**, **WebVTT (`.vtt`)**, และ **JSON (`.json`)** พร้อม Timecodes
- 🇹🇭 **Thai Typography Standards:** ปรับระดับความสูงบรรทัด (Line-height 1.5x) ป้องกันสระบน-ล่างและวรรณยุกต์ภาษาไทยตกหล่นหรือถูกตัดทอน
- 🌓 **Modern Dark & Light Themes:** รองรับทั้งธีมสว่างและธีมมืด สลับได้ทันที
- 💡 **AI Post-Processing Guidance:** มีหน้าต่างแนะนำการนำข้อความไปคลีนต่อด้วย AI พร้อมปุ่ม 1-Click Copy Prompt สำหรับสร้างบอตเฉพาะทาง (Gemini Gems, Custom GPTs, Claude/Qwen Projects) ขัดเกลาคำผิดตามบริบทโดยคงสำนวนภาษาเดิม

---

## 💻 ความต้องการของระบบและฮาร์ดแวร์ที่รองรับ (System Requirements & Hardware Compatibility)

### 2.1 ตารางเปรียบเทียบสเปกระบบขั้นต่ำและสเปกที่แนะนำ (System Requirements Matrix)

| รายการ (Component) | สเปกขั้นต่ำ (Minimum Requirements) | สเปกที่แนะนำ (Recommended Requirements) |
| :--- | :--- | :--- |
| **ระบบปฏิบัติการ (OS)** | • **Windows 10 / 11** (64-bit)<br>• **Linux** (Ubuntu 20.04+, Debian 11+, Fedora 38+, Arch Linux 64-bit) | • **Windows 11** (64-bit)<br>• **Linux** (Kernel 6.x+, Ubuntu 22.04 / 24.04 LTS) |
| **หน่วยประมวลผล (CPU)** | • **Dual-Core 64-bit x86_64** (พร้อมชุดคำสั่ง AVX2)<br>• Intel Core i3 Gen 4+ (Haswell 2013+)<br>• AMD Ryzen 1000 series+ / Athlon 200GE+ | • **Quad-Core หรือ 6-Core 64-bit x86_64** หรือสูงกว่า<br>• Intel Core i5 / i7 Gen 8+ (Coffee Lake ขึ้นไป)<br>• AMD Ryzen 3000 / 5000 / 7000 / 9000 series |
| **หน่วยความจำระบบ (RAM)** | • **4 GB RAM**<br>*(โปรแกรมและ ONNX Runtime ใช้ RAM จริงเพียง ~140 MB – 250 MB)* | • **8 GB – 16 GB RAM** ขึ้นไป (ทำงานร่วมกับแอปพลิเคชันอื่นได้อย่างราบรื่น) |
| **การ์ดแสดงผล (GPU Acceleration)**<br>*(ทางเลือกสำหรับโหมด GPU)* | • **NVIDIA 2 GB VRAM** (สถาปัตยกรรม Pascal `sm_61` ขึ้นไป)<br>• การ์ดเริ่มต้น: **GeForce GTX 1050 (2GB)** / 1050 Ti<br>• หรือ AMD / Intel GPU ที่รองรับ DirectX 12 (บน Windows) | • **NVIDIA 4 GB – 8 GB VRAM ขึ้นไป** พร้อม Tensor Cores<br>• GeForce RTX 2060 / 3050 / 3060 / 4060 หรือสูงกว่า<br>• ไดรเวอร์ NVIDIA Driver 550+ |
| **หน่วยความจำการ์ดจอ (VRAM)** | • **2 GB VRAM**<br>*(โมเดล FastConformer RNN-T ใช้ VRAM จริงเพียง ~450 MB – 600 MB)* | • **4 GB – 8 GB VRAM** ขึ้นไป |
| **พื้นที่จัดเก็บข้อมูล (Storage)** | • **1.5 GB** พื้นที่ว่างบนฮาร์ดดิสก์ (SSD หรือ HDD)<br>*(สำหรับไฟล์โมเดล ~480 MB + Virtual Environment ~700 MB)* | • **3.0 GB** พื้นที่ว่างบน **NVMe / SATA SSD**<br>*(ช่วยให้โหลดโมเดลเข้าหน่วยความจำได้เร็วภายใน 1-2 วินาที)* |
| **อุปกรณ์นำเข้าเสียง (Audio Input)** | • ไมโครโฟนทั่วไป (Built-in / USB 3.5mm)<br>• หรือ Virtual Loopback (Stereo Mix, Monitor of Audio) | • ไมโครโฟนแบบตัดเสียงรบกวน (Noise Cancelling) หรือ USB Audio Interface |
| **การเชื่อมต่ออินเทอร์เน็ต** | • จำเป็นเฉพาะตอนติดตั้งครั้งแรกและดาวน์โหลดโมเดล (~480 MB) | • หลังจากดาวน์โหลดโมเดลแล้ว **ใช้งานแบบ 100% Offline ได้ตลอดชีพ** |

---

### 2.2 รายละเอียดฮาร์ดแวร์ที่รองรับ (Hardware Compatibility Breakdown)

#### 🖥️ 1. สถาปัตยกรรม CPU
* **x86_64 / AMD64 (Intel & AMD):** รองรับคำสั่งชุด **AVX2** และ **FMA3** เพื่อเร่งความเร็วการแปลง Mel-Spectrogram (DSP) และการรัน ONNX Graph บน CPU
* **ARM64 / aarch64:** รองรับสำหรับ Linux ARM64 (เช่น Raspberry Pi 5 หรือ Server SBC) ผ่าน CPU Execution Provider

#### ⚡ 2. สถาปัตยกรรม GPU ที่ผ่านการทดสอบและรับรอง (Tested GPU Architectures)

| แบรนด์ / สถาปัตยกรรม | ตัวอย่างรุ่นการ์ดจอ | Compute Capability | VRAM ขั้นต่ำ | สถานะการรองรับ & Execution Provider |
| :--- | :--- | :---: | :---: | :--- |
| **NVIDIA Pascal** | GTX 1050, 1050 Ti, 1060, 1070, 1080, Titan X | `sm_61` | 2 GB | ✅ **รองรับสมบูรณ์** (ตรึง cuDNN 9.10.2.21 + ONNX Runtime 1.22.0 แก้ปัญหา Backend API Failed) |
| **NVIDIA Volta** | Titan V, Quadro GV100 | `sm_70` | 12 GB | ✅ **รองรับสมบูรณ์** (CUDA 12) |
| **NVIDIA Turing** | GTX 1650 Super, 1660, 1660 Ti, RTX 2060, 2070, 2080 | `sm_75` | 4 GB | ✅ **รองรับสมบูรณ์ + Tensor Cores** (CUDA 12) |
| **NVIDIA Ampere** | RTX 3050, 3060, 3070, 3080, 3090, A100, RTX A-Series | `sm_80`, `sm_86` | 4 GB | ✅ **รองรับสมบูรณ์ + Tensor Cores** (CUDA 12) |
| **NVIDIA Ada Lovelace** | RTX 4050, 4060, 4070, 4080, 4090, RTX 4000 Ada | `sm_89` | 6 GB | ✅ **รองรับสมบูรณ์ + 4th Gen Tensor Cores** (CUDA 12) |
| **NVIDIA Blackwell** | RTX 50 Series (RTX 5070, 5080, 5090), B200 | `sm_120`, `sm_100` | 8 GB | ✅ **รองรับ** (ผ่าน Forward PTX JIT บน CUDA 12) |
| **AMD Radeon (Windows)** | RX 5000, RX 6000, RX 7000 Series, Radeon Vega | DirectX 12 | 4 GB | ✅ **รองรับสมบูรณ์** (ผ่าน `DirectMLExecutionProvider`) |
| **AMD Radeon (Linux)** | RX 6000, RX 7000 Series | ROCm | 8 GB | ⚙️ รองรับผ่าน `onnxruntime-rocm` (หรือ Fallback รัน CPU ปลอดภัย) |
| **Intel Arc / Iris (Windows)** | Intel Arc A380, A580, A750, A770, Iris Xe Graphics | DirectX 12 | 4 GB | ✅ **รองรับสมบูรณ์** (ผ่าน `DirectMLExecutionProvider`) |
| **Intel Arc (Linux)** | Intel Arc A-Series | OpenVINO | 4 GB | ⚙️ รองรับผ่าน `onnxruntime-openvino` (หรือ Fallback รัน CPU ปลอดภัย) |

> [!TIP]
> **การใช้หน่วยความจำ VRAM ต่ำมาก:** ตัวโมเดล FastConformer RNN-T ใช้ VRAM จริงเพียงประมาณ **450 MB – 600 MB** เท่านั้น ทำให้การ์ดจอขนาดเล็กที่มี VRAM 2 GB เช่น **NVIDIA GeForce GTX 1050** สามารถประมวลผลถอดเสียงได้แบบเรียลไทม์โดยไม่มีปัญหา Out of Memory (OOM)

> [!NOTE]
> **ระบบ Dynamic Fallback ป้องกันการแครช:** หากผู้ใช้เลือกโหมด GPU แต่เครื่องไม่มี Driver ที่ตรงกัน หรือขาดไลบรารี CUDA ที่จำเป็น ระบบของ Typhoon Transcriber จะแจ้งเตือนและสลับกลับมาประมวลผลบน **CPU Execution Provider** ให้อัตโนมัติทันที ทำให้โปรแกรมยังคงทำงานต่อไปได้อย่างต่อเนื่อง ปลอดภัย ไม่แครช

---

## 🛠️ Prerequisites (สิ่งที่ต้องเตรียมใน Linux & โปรแกรมเสริม)

### 1. โปรแกรมเสริมด้านมัลติมีเดียและเสียงในระบบ (System Libraries)
ก่อนเริ่มติดตั้ง กรุณาติดตั้ง **PortAudio** (สำหรับสตรีมมิ่งเสียง) และ **FFmpeg** (สำหรับแปลงและถอดรหัสไฟล์เสียง/วิดีโอ) บน Linux:

* **Ubuntu / Debian / Linux Mint:**
  ```bash
  sudo apt update
  sudo apt install -y git curl libportaudio2 ffmpeg
  ```
* **Fedora / RHEL:**
  ```bash
  sudo dnf install -y git curl portaudio ffmpeg
  ```
* **Arch Linux / Manjaro:**
  ```bash
  sudo pacman -S --needed git curl portaudio ffmpeg
  ```

> [!NOTE]
> บน **Windows** ตัวไบนารีของ PortAudio จะถูกเรียกใช้งานอัตโนมัติผ่านแพ็กเกจ `sounddevice` โดยไม่ต้องติดตั้งไลบรารีระบบเพิ่มเติม

---

### 2. เครื่องมือจัดการสภาพแวดล้อม Python: `uv` (แนะนำอย่างยิ่ง)
โปรเจกต์นี้ใช้ **Astral `uv`** ในการจัดการแพ็กเกจและความเร็วในการติดตั้ง:

* **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source ~/.bashrc  # หรือ source ~/.zshrc
  ```
* **Windows (PowerShell):**
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

## 🚀 Installation & Virtual Environment (วิธีสร้าง venv)

### 1. Clone Source Code
```bash
git clone https://github.com/deawpic/THasrGui.git
cd THasrGui
```

### 2. สร้าง Virtual Environment ด้วย Python 3.11
โปรเจกต์นี้กำหนดให้ใช้ **Python 3.11** เพื่อความเข้ากันได้ 100% กับทั้ง ONNX Runtime, CUDA, และ Qt 6:

```bash
# 1. ให้ uv ติดตั้ง Python 3.11 ลงในระบบอัตโนมัติ (หากยังไม่มี)
uv python install 3.11

# 2. สร้างโฟลเดอร์ Virtual Environment (.venv) ที่เจาะจง Python 3.11
uv venv --python 3.11

# 3. สั่ง Activate Virtual Environment
source .venv/bin/activate
```

### 3. ซิงค์และติดตั้ง Dependencies ทั้งหมด
รันคำสั่งเดียวเพื่อให้ `uv` ติดตั้งทุกแพ็กเกจตามไฟล์ `uv.lock` อย่างแม่นยำ:
```bash
uv sync
```

### 4. เปิดใช้งานโปรแกรม
```bash
# รันผ่าน Entry Point ของแอปพลิเคชัน
uv run typhoon-transcriber

# หรือรันผ่านสคริปต์ตัวช่วยบน Linux
bash ../scripts/run_linux.sh
```

---

## ⚡ Hardware Selection (คำแนะนำการเลือกใช้ CPU vs GPU รุ่นต่างๆ)

โปรแกรมมีเมนู **⚡ Hardware / การประมวลผล** บนแถบด้านบนของหน้าจอให้เลือกสลับระหว่าง **🖥️ CPU** และ **⚡ GPU (Auto-Detect)** ได้ทันที:

### 5.1 โหมด CPU (สำหรับคอมพิวเตอร์ทั่วไป / ประหยัด RAM < 500 MB)
* **ความเหมาะสม:** คอมพิวเตอร์สำนักงานทั่วไป โน้ตบุ๊ก หรือเครื่องที่ไม่มีการ์ดจอแยก NVIDIA
* **ประสิทธิภาพ:** ตัวโมเดล FastConformer RNN-T ถูกปรับแต่งมาอย่างดี ใช้หน่วยความจำ RAM เพียง ~140 MB – 250 MB และใช้ CPU เพียง 2 Cores ก็สามารถถอดเสียงแบบเรียลไทม์ได้ลื่นไหล
* **การติดตั้ง:** เพียงรัน `uv sync` ก็พร้อมใช้งานได้ทันทีโดยไม่ต้องตั้งค่าไดรเวอร์ใดๆ เพิ่มเติม

---

### 5.2 โหมด GPU สำหรับ NVIDIA Pascal (GTX 1050, 1060, 1070, 1080 — `sm_61`)
* **ปัญหาทางเทคนิคของการ์ดตระกูล Pascal:** การ์ดจอรุ่น GTX 10xx ใช้สถาปัตยกรรม Pascal (`sm_61`) ซึ่งไลบรารี cuDNN เวอร์ชัน 9.11 ขึ้นไปได้ยกเลิกการรองรับ Convolution Frontend Graph ไปแล้ว ส่งผลให้เกิดข้อผิดพลาด `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED`
* **การแก้ปัญหาที่โปรเจกต์นี้กำหนดไว้ให้เรียบร้อยแล้ว:**
  1. ตรึงแพ็กเกจเป็น `onnxruntime-gpu==1.22.0` (คอมไพล์สำหรับ CUDA 12)
  2. ตรึงไลบรารี cuDNN เป็น `nvidia-cudnn-cu12==9.10.2.21` (เวอร์ชันสมบูรณ์สุดท้ายที่รองรับชิป Pascal `sm_61`)
  3. มีระบบ `_preload_nvidia_cuda_libraries()` ดึงไฟล์ `.so` ของ NVIDIA เข้าหน่วยความจำอัตโนมัติ โดยที่ผู้ใช้ไม่ต้อง export `LD_LIBRARY_PATH` เอง
* **ไดรเวอร์ที่ต้องการ:** ตรวจสอบว่ามี NVIDIA Driver 525+ ขึ้นไป (เช็กด้วยคำสั่ง `nvidia-smi`)
* **คำสั่งติดตั้ง:**
  ```bash
  # รัน uv sync ตามปกติ ระบบจะติดตั้งแพ็กเกจที่ตรึงเวอร์ชันไว้ให้เอง
  uv sync
  ```

---

### 5.3 โหมด GPU สำหรับ NVIDIA รุ่นใหม่ (RTX 20xx, 30xx, 40xx, 50xx — `sm_75`, `sm_86`, `sm_89`, `sm_120`)
* **ความเหมาะสม:** การ์ดจอ NVIDIA รุ่นใหม่ เช่น:
  - **Turing (`sm_75`):** RTX 2060, 2070, 2080, GTX 1660 Ti
  - **Ampere (`sm_86`, `sm_80`):** RTX 3050, 3060, 3070, 3080, 3090, A100
  - **Ada Lovelace (`sm_89`):** RTX 4060, 4070, 4080, 4090
  - **Blackwell (`sm_120`):** RTX 50 series (ผ่าน Forward PTX JIT Compatibility)
* **ประสิทธิภาพ:** ทำงานได้รวดเร็วกว่า CPU หลายเท่าตัว รองรับ Fused Kernels และ Tensor Cores อย่างเต็มประสิทธิภาพ
* **ไดรเวอร์ที่ต้องการ:** NVIDIA Driver 550+ ขึ้นไป
* **คำสั่งติดตั้ง:** ทำงานได้ทันทีผ่าน `uv sync`

---

### 5.4 โหมด GPU สำหรับ AMD Radeon & Intel Arc (Windows DirectML & Linux)

#### 🪟 การใช้งานบน Windows (DirectML — Zero Config)
* ระบบจะตรวจจับและเลือกใช้ **`DirectMLExecutionProvider`** ผ่าน DirectX 12 ให้อัตโนมัติ รองรับทั้ง AMD Radeon (RX 5000 / 6000 / 7000) และ Intel Arc (A380 / A580 / A750 / A770) โดยไม่ต้องติดตั้งไดรเวอร์ AI พิเศษเพิ่มเติม

---

#### 🐧 การใช้งานบน Linux (AMD ROCm & Intel OpenVINO)

บน Linux ค่าเริ่มต้นของโปรเจกต์ถูกคอนฟิกไว้สำหรับ NVIDIA CUDA (`onnxruntime-gpu`) หากคุณต้องการเปิดใช้งานการเร่งความเร็วบน **AMD Radeon GPU (ROCm)** หรือ **Intel Arc GPU (OpenVINO)** ให้ทำตามขั้นตอนการติดตั้งคำสั่งใน Linux ดังนี้:

##### 🅰️ สำหรับ AMD Radeon GPU (ผ่าน ROCm)
1. **ติดตั้ง ROCm Driver และกำหนดสิทธิ์การเข้าถึงอุปกรณ์บน Linux:**
   ```bash
   # เพิ่มผู้ใช้ปัจจุบันเข้ากลุ่ม video และ render เพื่อให้เข้าถึงฮาร์ดแวร์ GPU ได้
   sudo usermod -aG render,video $USER

   # สำหรับ Ubuntu/Debian ติดตั้ง ROCm HIP runtime (ตรวจสอบรุ่นการ์ดจอที่รองรับ เช่น RX 6000/7000 series)
   sudo apt update
   sudo apt install -y rocm-hip-sdk
   ```
2. **ติดตั้งแพ็กเกจ `onnxruntime-rocm` ใน Virtual Environment ของแอป (`src/`):**
   ```bash
   cd src

   # ถอนการติดตั้ง onnxruntime-gpu ตัวเดิมออกก่อน เพื่อป้องกันไลบรารีชนกัน
   uv pip uninstall onnxruntime-gpu onnxruntime

   # ติดตั้ง onnxruntime-rocm สำหรับ Linux
   uv pip install onnxruntime-rocm
   ```
3. **เปิดใช้งานโปรแกรม:**
   ```bash
   uv run typhoon-transcriber
   ```
   เมื่อเลือกเมนู **⚡ Hardware -> GPU** ระบบจะตรวจจับ `ROCmExecutionProvider` และประมวลผลบน GPU ของ AMD ให้อัตโนมัติ

---

##### 🟦 สำหรับ Intel Arc / Iris Xe / Core Ultra (ผ่าน OpenVINO)
1. **ติดตั้ง Intel Compute Runtime & Level Zero Driver บน Linux:**
   * **Ubuntu / Debian:**
     ```bash
     sudo apt update
     sudo apt install -y intel-opencl-icd intel-level-zero-gpu
     sudo usermod -aG render,video $USER
     ```
   * **Fedora:**
     ```bash
     sudo dnf install -y intel-compute-runtime level-zero
     sudo usermod -aG render,video $USER
     ```
   * **Arch Linux:**
     ```bash
     sudo pacman -S --needed intel-compute-runtime level-zero-loader
     sudo usermod -aG render,video $USER
     ```
2. **ติดตั้งแพ็กเกจ `onnxruntime-openvino` ใน Virtual Environment ของแอป (`src/`):**
   ```bash
   cd src

   # ถอนการติดตั้ง onnxruntime-gpu ตัวเดิมออกก่อน
   uv pip uninstall onnxruntime-gpu onnxruntime

   # ติดตั้ง onnxruntime-openvino
   uv pip install onnxruntime-openvino
   ```
3. **เปิดใช้งานโปรแกรม:**
   ```bash
   uv run typhoon-transcriber
   ```
   ระบบจะตรวจจับ `OpenVINOExecutionProvider` และดึงพลังของ Intel Arc GPU หรือ iGPU/NPU มาช่วยถอดรหัสเสียงให้อัตโนมัติ

> [!TIP]
> **หากไม่ต้องการลงไดรเวอร์เสริมบน Linux:** หากคุณใช้การ์ดจอ AMD หรือ Intel บน Linux แล้วไม่ต้องการเซ็ตอัป ROCm หรือ OpenVINO คุณสามารถสลับไปใช้ **🖥️ CPU** ได้ทันที ซึ่งทำงานได้เสถียรและรวดเร็วแบบ Real-time (ความเร็ว ~1.5x ของเสียงสด) ใช้ RAM ต่ำมากเพียง ~140 MB – 250 MB โดยไม่ต้องลงไดรเวอร์หรือแพ็กเกจเสริมใดๆ ทั้งสิ้น

---

## 📥 Automated Model Setup (ระบบตรวจสอบและดาวน์โหลดโมเดลอัตโนมัติ)

แอปพลิเคชันถูกออกแบบมาให้ทำงานแบบ **Zero-Configuration & Zero-Click**:

### 6.1 ระบบตรวจสอบและดาวน์โหลดอัตโนมัติเมื่อเปิดโปรแกรม (Zero-Click Auto Downloader)
1. **ตรวจสอบอัตโนมัติเมื่อเปิดโปรแกรม:**
   - เมื่อเปิดแอปพลิเคชัน ระบบจะตรวจสอบไฟล์โมเดลทั้ง 3 ไฟล์ (`encoder`, `decoder_joint`, และ `vocab.json`) โดยอัตโนมัติ
   - **กรณีมีไฟล์อยู่แล้ว:** แอปพลิเคชันจะโหลดโมเดลจริงเข้าหน่วยความจำทันที พร้อมใช้งานทันที
   - **กรณีเปิดใช้งานครั้งแรก (ยังไม่มีไฟล์โมเดล):** จะมีหน้าต่างถามยืนยันการดาวน์โหลดโมเดลทางการ (~480 MB) หากกด **Yes** ระบบจะดาวน์โหลดและเริ่มใช้งานถอดเสียงจริงให้ทันทีอัตโนมัติ
2. **ระบบกู้คืนอัตโนมัติเมื่อกดเริ่มทำงาน:**
   - หากผู้ใช้ยังไม่ได้ดาวน์โหลดไฟล์โมเดล แล้วกดปุ่ม **▶ Start Live** หรือ **▶ Start Batch** ระบบจะตรวจจับและแนะนำให้ดาวน์โหลดโมเดลจริงทันทีก่อนเริ่มประมวลผล

---

### 6.2 การดาวน์โหลดและวางไฟล์ด้วยตนเองสำหรับสภาพแวดล้อม Offline (Manual Download)
หากต้องการดาวน์โหลดไฟล์โมเดลมาวางไว้ล่วงหน้า (เช่น สำหรับเครื่องที่ไม่มีอินเทอร์เน็ต) ให้สร้างโฟลเดอร์ `models/` ไว้ข้างแอปพลิเคชันหรือที่ `~/.cache/typhoon-asr/`:

#### 1. Official Model Links (Hugging Face):
* **Hugging Face Model Hub:** [wannaphong/typhoon-asr-realtime-onnx](https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx)
* **PyThaiASR:** [Python Thai Automatic Speech Recognition](https://github.com/PyThaiNLP/pythaiasr)
 
#### 2. โครงสร้างโฟลเดอร์สำหรับวางไฟล์:
นำไฟล์มาวางไว้ที่โฟลเดอร์ `models/` ภายในโปรเจกต์ หรือ `~/.cache/typhoon-asr/`:
```text
App-Dir/
└── models/
    ├── encoder-fastconformer-quran-ar.onnx        # กราฟคำนวณ Encoder (435 MB)
    ├── decoder_joint-fastconformer-quran-ar.onnx  # กราฟคำนวณ Decoder & Joint (26.5 MB)
    └── vocab.json                                 # พจนานุกรม Token Vocab (36 KB)
```

#### 3. คำสั่งดาวน์โหลดผ่าน Terminal (CLI Download Commands):
```bash
# สร้างโฟลเดอร์ models
mkdir -p models

# 1. ดาวน์โหลด Vocabulary JSON (36 KB)
curl -L -o models/vocab.json \
  "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/tokenizer/vocab.json"

# 2. ดาวน์โหลด FastConformer Encoder Graph (435 MB)
curl -L -o models/encoder-fastconformer-quran-ar.onnx \
  "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/encoder-fastconformer-quran-ar.onnx"

# 3. ดาวน์โหลด Decoder & Joint Graph (26.5 MB)
curl -L -o models/decoder_joint-fastconformer-quran-ar.onnx \
  "https://huggingface.co/wannaphong/typhoon-asr-realtime-onnx/resolve/main/decoder_joint-fastconformer-quran-ar.onnx"
```

---

### 6.3 การนำเข้าโมเดลแบบกำหนดเอง (Custom Local Models)
หากคุณแปลงโมเดล FastConformer หรือ CTC เป็น ONNX ด้วยตนเอง คุณสามารถวางไฟล์โมเดลลงในโฟลเดอร์ `models/` ตามชื่อ candidate ที่ระบบรองรับ หรือตั้งค่า Environment Variable `TYPHOON_MODEL_DIR=/path/to/models` ได้ทันที ระบบจะตรวจจับและโหลดโมเดลของคุณขึ้นมาใช้งานโดยอัตโนมัติ

---

## 💼 วิธีจัดทำ Portable App (Copy ไฟล์โมเดลไว้ในโฟลเดอร์เดียวกับแอป)

หากคุณต้องการนำแอปพลิเคชันไปรันแบบ **Portable (พกพาใน Flash Drive / USB หรือนำไปใช้บนเครื่องที่ไม่มีการเชื่อมต่ออินเทอร์เน็ต)** โดยไม่ต้องการให้แอปพึ่งพาการดาวน์โหลดไปยังแคชส่วนกลาง (`~/.cache/typhoon-asr/`) คุณสามารถ Copy ไฟล์โมเดลทั้ง 3 ไฟล์มาวางไว้ในโฟลเดอร์เดียวกับตัวแอปพลิเคชันได้โดยตรง:

### 📦 ไฟล์โมเดลที่ต้องใช้ (ทั้งหมด 3 ไฟล์):
1. **`encoder-fastconformer-quran-ar.onnx`** (456 MB) — กราฟการคำนวณ FastConformer Encoder
2. **`decoder_joint-fastconformer-quran-ar.onnx`** (26.5 MB) — กราฟการคำนวณ Decoder & Joint Network
3. **`vocab.json`** (15 KB) — พจนานุกรม Token Vocab ภาษาไทย

---

### 📂 รูปแบบการวางไฟล์สำหรับ Portable App:

คุณสามารถเลือกวางไฟล์ได้ 2 รูปแบบตามความสะดวก โดยระบบจะค้นหาและตรวจจับให้อัตโนมัติ:

#### รูปแบบที่ 1: วางไว้ในโฟลเดอร์ `models/` ภายในแอป (แนะนำ — เป็นระเบียบเรียบร้อย)
สร้างโฟลเดอร์ชื่อ `models` ไว้ในโฟลเดอร์แอป (`src/` หรือโฟลเดอร์ Portable Bundle):
```text
TyphoonTranscriber-Portable/
├── pyproject.toml
├── typhoon_transcriber/
│   ├── main.py
│   └── ...
└── models/                                      <-- โฟลเดอร์ models ของแอป
    ├── encoder-fastconformer-quran-ar.onnx      <-- วางไฟล์ที่นี่
    ├── decoder_joint-fastconformer-quran-ar.onnx <-- วางไฟล์ที่นี่
    └── vocab.json                               <-- วางไฟล์ที่นี่
```

#### รูปแบบที่ 2: วางรวมไว้ใน Root Directory เดียวกับตัวรันโปรแกรม
คุณสามารถวางไฟล์ `.onnx` และ `vocab.json` ไว้ในระดับเดียวกันกับตัวรันโปรแกรมหรือไฟล์ `.exe` ได้ทันที โดยไม่ต้องสร้างโฟลเดอร์ย่อย:
```text
TyphoonTranscriber-Portable/
├── typhoon-transcriber (หรือ typhoon-transcriber.exe)
├── encoder-fastconformer-quran-ar.onnx
├── decoder_joint-fastconformer-quran-ar.onnx
├── vocab.json
└── ...
```

> [!TIP]
> **การทำงานเมื่อเปิดแอป:** คลาส `ModelManager` จะตรวจพบไฟล์โมเดลในโฟลเดอร์ของแอปทันที แถบสีเหลือง `Mock Mode` จะหายไป และแอปพลิเคชันจะเข้าสู่โหมดถอดเสียงจริงแบบ **100% Offline** โดยไม่ต้องดาวน์โหลดอะไรเพิ่มเติม!

---

## 🔎 ลำดับ Model Search Path (การค้นหาโมเดลของระบบ)

คลาส [`ModelManager`](typhoon_transcriber/models/model_manager.py) มีระบบค้นหาโมเดลอัตโนมัติที่ยืดหยุ่นสูง เพื่อรองรับทั้งการพัฒนาโปรแกรม (Dev), การใช้งานในเครื่อง (Local User), การพกพา (Portable App), และการติดตั้งบนระบบองค์กร/เซิร์ฟเวอร์ โดยมีลำดับความสำคัญ (Priority Order) ดังต่อไปนี้:

| ลำดับ (Priority) | ตำแหน่งค้นหา (Search Directory) | กรณีการใช้งานหลัก (Use Case) |
| :---: | :--- | :--- |
| **1** | **`$TYPHOON_MODEL_DIR`** | กำหนด Path ของโมเดลเองผ่าน Environment Variable เช่น ใน Server, Docker หรือแชร์โมเดลร่วมกันในระบบ |
| **2** | **`<Executable_Dir>/models/` และ `<Executable_Dir>/`** | สำหรับ **Portable App** เมื่อ build เป็น standalone binary หรือ PyInstaller `.exe` (วางไฟล์ไว้ข้าง `.exe` ได้ทันที) |
| **3** | **`<App_Root>/models/` และ `<App_Root>/`** | สำหรับ Source Code / Portable Folder (เช่น โฟลเดอร์ `src/models/` หรือ `src/`) |
| **4** | **`<Package_Dir>/models/` และ `<Package_Dir>/`** | โฟลเดอร์แพ็กเกจ `typhoon_transcriber/` |
| **5** | **`./models/`, `./assets/`, และ `./`** | โฟลเดอร์ปัจจุบันที่เปิดเทอร์มินัลรันคำสั่ง (Current Working Directory) |
| **6** | **`~/.cache/typhoon-asr/` (หรือ `$TYPHOON_CACHE_DIR`)** | โฟลเดอร์ Default Cache ประจำเครื่องของผู้ใช้ (ตำแหน่งที่ปุ่ม Auto-Downloader บันทึกไฟล์ลงมา) |

### 📋 รายชื่อไฟล์ที่ระบบตรวจจับอัตโนมัติ (Candidate Filenames)

ในแต่ละ Directory ด้านบน ระบบจะค้นหาชื่อไฟล์ตามลำดับดังต่อไปนี้:

1. **Acoustic Encoder Model:**
   - `encoder-fastconformer-quran-ar.onnx` *(Official FastConformer RNN-T Encoder)*
   - `encoder.onnx`
   - `typhoon_asr_realtime.onnx`
   - `model.onnx`

2. **Decoder & Joint Model (สำหรับ FastConformer RNN-T):**
   - `decoder_joint-fastconformer-quran-ar.onnx` *(Official FastConformer RNN-T Decoder/Joint)*
   - `decoder_joint.onnx`
   - `decoder.onnx`

3. **Tokenizer / Vocabulary Mapping:**
   - `vocab.json` *(Official JSON Vocab Mapping)*
   - `tokenizer.model` *(SentencePiece Processor File)*
   - `vocab.txt`

---

## 📂 ระบบ Batch Conversion & Destination Popup

โปรแกรมรองรับการแปลงไฟล์เสียงและวิดีโอแบบกลุ่ม (Batch Mode) โดยแยกหน้าตารางการทำงานไว้บนแท็บ **Batch Transcription** อย่างเป็นระเบียบ:

### 1. วิธีเพิ่มไฟล์ / โฟลเดอร์ พร้อมหน้าต่างตั้งค่า (Destination Popup):
- **ลากและวางไฟล์ (Drag & Drop):** ลากไฟล์เสียง/วิดีโอ หรือลากทั้งโฟลเดอร์มาปล่อยลงบนตารางได้ทันที
- **ปุ่มเลือกไฟล์ / โฟลเดอร์:**
  - กดปุ่ม **"➕ Add Files..."** เพื่อเลือกไฟล์แบบ Multi-select
  - กดปุ่ม **"📂 Add Folder..."** เพื่อเลือก Root Folder (สแกนหาไฟล์ใน Subfolder ทั้งหมดแบบ Recursive อัตโนมัติ)
- **หน้าต่าง Popup ถามโฟลเดอร์ปลายทาง (Batch Configuration Popup):**
  - เมื่อเพิ่มหรือวางไฟล์ ระบบจะเปิดหน้าต่าง Popup ขึ้นมาให้คุณตรวจสอบรายชื่อไฟล์และขนาดไฟล์
  - กำหนดหรือเปลี่ยนโฟลเดอร์ปลายทาง (Destination Directory) ผ่านปุ่ม `📂 Browse...` ได้ทันที (ค่าเริ่มต้น: `~/Transcripts/`)
  - เลือกฟอร์แมตผลลัพธ์ที่ต้องการ เช่น `.txt`, `.srt`, `.vtt`, `.json`
  - สามารถเลือกกด **"📥 Add to Queue"** เพื่อเพิ่มเข้ารายการในตาราง หรือกด **"🚀 Start Batch Transcription"** เพื่อเริ่มแปลงไฟล์ทันที

### 2. ระบบ Folder Structure Mirroring:
เมื่อคุณเพิ่มโฟลเดอร์ที่มีโครงสร้างซับซ้อน เช่น:
```text
/home/user/AudioRecordings/
├── 2026/
│   ├── Q1/
│   │   ├── meeting_jan.mp3
│   │   └── meeting_feb.wav
│   └── Q2/
│       └── planning.m4a
└── Interviews/
    └── candidate_01.flac
```
เมื่อเลือก Destination Folder เป็น `/home/user/Transcripts/` และเลือก Output Format เป็น `.txt` และ `.srt`:
ระบบจะสร้างโฟลเดอร์ย่อยและไฟล์ผลลัพธ์ให้ตรงตามโครงสร้างต้นทาง 100%:
```text
/home/user/Transcripts/
├── 2026/
│   ├── Q1/
│   │   ├── meeting_jan.txt & meeting_jan.srt
│   │   └── meeting_feb.txt & meeting_feb.srt
│   └── Q2/
│       └── planning.txt & planning.srt
└── Interviews/
    └── candidate_01.txt & candidate_01.srt
```

### 3. ตารางจัดการคิวแบบใหม่ & การแก้ไขปลายทางเฉพาะไฟล์ (Output Edit & Multi-Select):
- **ตาราง 5 คอลัมน์กระชับ ชัดเจน:** `#`, `File Name`, `Size (MB)`, `Status`, และ `Output Destination / Details`
- **คลิกขวาแก้ไขปลายทาง (Right-Click Context Menu):**
  - คลิกขวาที่แถวหรือช่อง Output เพื่อเลือก **"✏️ Change Output Destination..."** กำหนดโฟลเดอร์เซฟแยกเฉพาะไฟล์หรือกลุ่มไฟล์ที่เลือกได้ทันที
  - สามารถดับเบิลคลิก (Double-click) ที่ช่องคอลัมน์ Output เพื่อเปิดหน้าต่างเลือกโฟลเดอร์ได้ทันทีเช่นกัน
  - เมนู **"🔄 Reset Status to Pending"** เพื่อรีเซ็ตสถานะกลับเป็น Pending กรณีต้องการรันแปลงไฟล์นั้นใหม่อีกรอบ
- **เลือกหลายแถวพร้อมกัน (Multi-Selection):**
  - กดคีย์ `Shift` + คลิก เพื่อเลือกเป็นช่วงแถวต่อเนื่อง
  - กดคีย์ `Ctrl` + คลิก เพื่อเลือกหรือยกเลิกหลายแถวแบบกระจาย
  - กด `Ctrl + A` เพื่อเลือกทั้งหมด จากนั้นกด **"➖ Remove Selected"** หรือคลิกขวาเปลี่ยนโฟลเดอร์พร้อมกันทุกไฟล์ได้

### 4. ระบบบันทึกคิวงาน & กู้คืนอัตโนมัติเมื่อปิดหรือ Crash (Queue Persistence & Auto-Recovery):
- **💾 ปุ่ม "Save Queue...":** บันทึกรายการคิวงานทั้งหมดพร้อมสถานะ (เช่น ไฟล์ไหน Completed แล้ว หรือยัง Pending) ลงไฟล์ `.json` เพื่อนำกลับมาทำต่อวันหลังได้
- **📂 ปุ่ม "Load Queue... (รองรับ Multiple Queue & Append):**
  - สามารถเลือกไฟล์คิวงาน `.json` ได้ **หลายไฟล์พร้อมกัน** ในครั้งเดียว (กด `Ctrl` หรือ `Shift` ขณะเลือกไฟล์)
  - หากปัจจุบันมีรายการในคิวอยู่แล้ว ระบบจะแสดงตัวเลือกว่าต้องการ:
    - **`➕ เพิ่มต่อท้าย (Append)`**: นำรายการจากคิวที่เลือกมาต่อท้ายคิวเดิม โดยตรวจเช็คไม่ให้มีไฟล์ซ้ำ และหากไฟล์เดิมยังไม่เสร็จแต่ในคิวที่โหลดมามีสถานะ `Completed` แล้ว ระบบจะอัปเดตสถานะเป็นเสร็จให้อัตโนมัติ
    - **`🔄 แทนที่คิวเดิม (Replace)`**: ล้างคิวเดิมออกแล้วแทนที่ด้วยคิวใหม่ทั้งหมด
  - เมื่อกด **"🚀 Start Batch Transcription"** ระบบจะข้ามไฟล์ที่ `Completed` แล้ว และประมวลผลเฉพาะไฟล์ที่เหลืออยู่ให้อัตโนมัติ
- **🛡️ Last Session Auto-Recovery (กัน Crash / เผลอปิด):**
  - ทุกครั้งที่มีการเพิ่มไฟล์ เปลี่ยนสถานะ หรือแปลงไฟล์เสร็จ ระบบจะ Auto-save ลงแคชเซสชันทันที
  - เมื่อเปิดโปรแกรมใหม่ รายการคิวล่าสุดจะถูกโหลดกลับมาให้อัตโนมัติ โดยไฟล์ที่เคยค้างสถานะ `Processing` ขณะปิด/Crash จะถูกปรับเป็น `Pending` เพื่อให้กดรันต่อได้ทันทีโดยไม่ต้องเริ่มต้นใหม่ทั้งหมด

---

## 🤖 คำแนะนำการปรับปรุงข้อความด้วย AI (AI Post-Processing)

โมเดลถอดเสียง ASR (รวมถึง Typhoon ASR) อาจมีกรณีที่คำพ้องเสียง, คำที่สะกดผิดตามบริบท, หรือการตัดคำ/เว้นวรรคมีความคลาดเคลื่อน ท่านสามารถใช้พลังของโมเดลภาษาขนาดใหญ่ (LLMs) ยุคปัจจุบัน เช่น **Gemini Gems**, **ChatGPT (Custom GPTs)**, **Claude Projects**, หรือ **Qwen Projects** ช่วยขัดเกลาและคลีนข้อความภาษาไทยได้อย่างมีประสิทธิภาพสูงและตรงตามบริบท 100%

ภายในโปรแกรม Typhoon ASR Desktop Transcriber มีหน้าต่าง **💡 AI Post-Processing** ให้กดดูคำแนะนำและมีปุ่ม **📋 คัดลอก Prompt** ได้ในคลิกเดียว (สามารถเปิดได้จากปุ่ม **💡 AI Post-Processing** บน Header Bar ของหน้าต่างหลัก และจากหน้าต่าง **Export Dialog**)

### 📌 กฎเหล็กในการทำความสะอาดข้อความ (Clean & Correct Rules)
1. **รักษาภาษาเดิม:** ห้ามเปลี่ยนภาษาพูดให้กลายเป็นภาษาเขียน ห้ามปรับคำสแลงหรือคำสร้อย (เช่น *ครับ, ค่ะ, นะครับ, อะ, ป่ะ, เนอะ*) ให้หายไป หากต้นฉบับเป็นภาษาพูดที่เป็นกันเอง ให้คงระดับความเป็นกันเองไว้ หากต้นฉบับเป็นทางการ ให้คงความทางการไว้
2. **แก้ไขเฉพาะคำผิดตามบริบท:** เปลี่ยนเฉพาะคำที่พิมพ์ผิด/สะกดผิด/ตัดคำเพี้ยน ให้เป็นคำที่ถูกต้อง (เช่น *"พุ่งนี้" ➔ "พรุ่งนี้"*, *"น้ารัก" ➔ "น่ารัก"*, *"ลบกวน" ➔ "รบกวน"*)
3. **จัดช่องไฟให้อ่านง่าย:** เนื่องจากข้อความจาก ASR มักเว้นวรรคสะเปะสะปะตามจังหวะหยุดพูด ให้ช่วยรวบคำและเว้นวรรคประโยคให้อ่านง่ายตามหลักภาษาไทยที่ถูกต้อง
4. **รูปแบบการ Output:** ให้ตอบกลับเฉพาะข้อความที่แก้ไขเสร็จแล้วเท่านั้น ห้ามทักทาย ห้ามอธิบาย และห้ามใส่เครื่องหมายคำพูดคร่อมข้อความ
5. **การจัดการไฟล์เอกสาร (File Input & Export Format):** เมื่ออัปโหลดไฟล์ข้อความ (`.txt`) ให้ AI อ่านและประมวลผลเนื้อหาทั้งหมด และนำผลลัพธ์ใส่ไว้ใน **Markdown Code Block** เพื่อให้สามารถกดปุ่มคัดลอก (Copy) นำข้อความกลับไปใช้งานต่อได้ง่ายในคลิกเดียว

### 📋 คำสั่ง System Instruction / Prompt (พร้อมคัดลอกไปใช้งาน)

คัดลอกข้อความด้านล่างไปวางในช่อง System Instruction หรือ Custom Instructions ของ AI Bot ของท่าน :

```text
คุณคือ "ผู้เชี่ยวชาญด้านการตรวจสอบและแก้ไขคำผิดภาษาไทยจากข้อความเสียง (ASR)" หน้าที่หลักของคุณคือ แก้ไขคำที่สะกดผิด คำพ้องเสียง หรือคำที่วรรณยุกต์เพี้ยน ให้ถูกต้องตามบริบท โดยยังคง "สไตล์ น้ำเสียง และระดับภาษา (Tone & Style)" ของข้อความต้นฉบับไว้อย่างเคร่งครัด

กฎเหล็กในการทำความสะอาดข้อความ (Clean & Correct):
1. รักษาภาษาเดิม: ห้ามเปลี่ยนภาษาพูดให้กลายเป็นภาษาเขียน ห้ามปรับคำสแลงหรือคำสร้อย (เช่น ครับ, ค่ะ, นะครับ, อะ, ป่ะ, เนอะ) ให้หายไป หากต้นฉบับเป็นภาษาพูดที่เป็นกันเอง ให้คงระดับความเป็นกันเองไว้ หากต้นฉบับเป็นทางการ ให้คงความทางการไว้
2. แก้ไขเฉพาะคำผิดตามบริบท: เปลี่ยนเฉพาะคำที่พิมพ์ผิด/สะกดผิด/ตัดคำเพี้ยน ให้เป็นคำที่ถูกต้อง (เช่น "พุ่งนี้" -> "พรุ่งนี้", "น้ารัก" -> "น่ารัก", "ลบกวน" -> "รบกวน")
3. จัดช่องไฟให้อ่านง่าย: เนื่องจากข้อความจาก ASR มักเว้นวรรคสะเปะสะปะ ให้ช่วยรวบคำและเว้นวรรคประโยคให้อ่านง่ายตามหลักภาษาไทยที่ถูกต้อง
4. รูปแบบการ Output: ให้ตอบกลับเฉพาะข้อความที่แก้ไขเสร็จแล้วเท่านั้น ห้ามทักทาย ห้ามอธิบาย และห้ามใส่เครื่องหมายคำพูดคร่อมข้อความ

ตัวอย่างการรักษาตัวตนดั้งเดิม (Examples):
Input: วัน นี้ อากาศ ดี ม้าก เวย แกร อยาก ไป เที่ยว อะ
Output: วันนี้อากาศดีมากเลยแก อยากไปเที่ยวอะ

Input: คับ พี่ ตอน นี้ ผม ตรวด สอบ ข้อมล ไห้ ยุ นะ คับ สัด ครู่
Output: ครับพี่ ตอนนี้ผมตรวจสอบข้อมูลให้อยู่ระครับ สักครู่

Input: ขอกราบ เรียน ท่าน ประทาน และ คณะ กรรม การ ทุก ท่าน คับ
Output: ขอกราบเรียนท่านประธานและคณะกรรมการทุกท่านครับ

5. การจัดการไฟล์เอกสาร (File Input & Export Format):
- เมื่อผู้ใช้อัปโหลดไฟล์ข้อความ (.txt) ให้ทำการอ่านและประมวลผลเนื้อหาทั้งหมดด้านในไฟล์ตามกฎข้อ 1-4
- ให้นำข้อความผลลัพธ์ที่แก้ไขและขัดเกลาเสร็จเรียบร้อยแล้วทั้งหมด ใส่ไว้ใน "กล่องข้อความโค้ด (Markdown Code Block)" เพื่อให้ผู้ใช้สามารถกดปุ่มคัดลอก (Copy) นำข้อความกลับไปใช้งานต่อได้ง่ายในคลิกเดียว
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **`F9`** | Toggle Live Transcription Start / Stop |
| **`➕` / `➖` (Font Size)** | ขยาย / ลดขนาดตัวอักษรของข้อความ (Zoom Transcript Font Size in / out) |
| **`Ctrl + C`** | Copy selected text |

---

## 🧪 Running Automated Tests

Run the full deterministic unit and GUI test suite via `pytest`:

```bash
uv run pytest -v tests/
```

Test coverage includes:
- Pure NumPy 80-channel Log-Mel Spectrogram extraction.
- Thai text normalization and SentencePiece tokenization.
- Audio downmixing, resampling (48kHz -> 16kHz), and sliding window buffer.
- Energy VAD silence detection.
- Subtitle timecode formatting (`.srt`, `.vtt`, `.json`).
- Headless PySide6 GUI lifecycle and controls (`pytest-qt`).

---

## 📁 Repository Structure

```text
src/
├── pyproject.toml                     # Project dependencies & packaging config
├── uv.lock                            # Deterministic package lockfile
├── .python-version                    # Pinned Python 3.11 runtime
├── README.md                          # Application documentation
├── typhoon-transcriber.desktop        # Linux desktop launcher entry
├── typhoon_transcriber/               # Core application package
│   ├── __init__.py
│   ├── main.py                        # Entry point
│   ├── config.py                      # Global audio & VAD settings
│   ├── audio/                         # DSP, device manager, VAD, capture engine
│   ├── models/                        # ONNX engine, tokenizer, Mel feature extractor, model manager
│   ├── transcriber/                   # Streaming & batch workers, text stitcher, exporter
│   └── ui/                            # PySide6 MainWindow, VU meter, download dialog, styles
└── tests/                             # Pytest test suites
    ├── test_audio_dsp.py
    ├── test_batch_queue.py
    ├── test_model_pipeline.py
    ├── test_transcriber.py
    └── test_ui.py
```

---

## 📄 License

Licensed under the Apache License, Version 2.0.
