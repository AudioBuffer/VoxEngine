# VoxEngine (AudioBuffer)

**VoxEngine** is an end-side native AI Singing Voice Synthesis (SVS) engine tailored for individual geeks and low-spec mobile devices. 

The core mission is to break free from expensive cloud computing, laggy browser shells (WebView), and complex local compilation environments. It enables millisecond-level, high-quality AI singing output on older devices, making private, offline, and lightweight vocal creation accessible to everyone.

## Core Features

- **Native & Lightweight Architecture**: Completely abandons heavy WebView containers. Built with a custom native UI and process scheduling system, resulting in ultra-low memory usage and consistent high-frame-rate performance. Solves the common mobile AI synthesis issues of lag, latency, and background process killing.
- **Fully Offline Local Inference**: All inference, rendering, and parsing are done on-device. No internet required, zero data uploads, zero privacy risks. Fully functional without network connectivity, enabling truly private AI audio creation.
- **Controllable "Humanity" System**: Features a core self-developed "Humanity Slider" mechanism. Supports seamless, gradient control of perfectionism. Users can instantly switch between "flawless robotic perfection" and "natural human flaws" (breath fluctuations, pitch deviations, emotional nuances) with a single tap.
- **Ultra-Fast Asynchronous Rendering Pipeline**: Utilizes a disruptive compute-splitting architecture and parallel scheduling logic. Breaks the traditional serial frame-by-frame synthesis bottleneck, supporting rapid rendering of long audio. Generation speed approaches that of streaming media.
- **Hot-Swappable "Cartridge" Engine Mechanism**: Pioneering a complete decoupling of the UI and the core engine. Models, inference runtimes, and voicebanks are fully modularized. Upgrading, swapping, or rolling back engines is like changing a game cartridge—seconds to deploy, no need to reinstall the application.

## Macro Technical Route

- **Three-Tier Architecture**: Strictly separates the "UI Rendering & Scheduling Layer", "Business Logic Distribution Layer", and "Compute Engine Layer". Decoupled layers with independent responsibilities ensure extreme stability, extensibility, and iterability.
- **Hybrid Rendering (Physical Acoustics + Lightweight Neural Models)**: Utilizes low-power physical acoustic algorithms (Rosenberg/LF glottal pulse + 5-cavity formant cascade) to generate highly accurate, stable vocal skeletons and pitch curves. Lightweight neural diffusion models are then used to fill in details, polish textures, and impart personality and emotion to the voice, balancing speed, precision, and device compatibility.
- **Fully Asynchronous Non-Blocking Pipeline**: Breaks traditional serial synthesis logic. Employs asynchronous parallel queue processing to achieve millisecond-level response, lag-free multitasking, and non-blocking continuous generation.
- **Open & Controllable Data Ecosystem**: Features an open editable project node structure (plain-text JSON). Allows creators to freely modify lyrics, pitch, dynamics, breath, and style parameters. Combined with a private encrypted packaging standard, balancing openness, extensibility, and copyright security.

## Project Roadmap

- **Phase 1 (V1.0 Physical Skeleton)**: Complete the foundational physical acoustics core, base scheduling logic, and simple synthesis closed-loop. Verify underlying feasibility. (Current Stage)
- **Phase 2 (V2.0 High-Precision Synthesizer)**: Introduce the LF glottal model and waveguide vocal tract model to fill in consonants and nasals. Utilize WebAssembly (WASM) for performance release.
- **Phase 3 (Complete Product Launch)**: Native client shell, hot-swappable engine system, and project file system fully implemented. Forming a standalone, fully functional application with freely swappable engines.
- **Phase 4 (Multi-platform & Extreme Optimization)**: Complete multi-platform adaptation, legacy device compatibility, and extreme compute optimization. Maximize performance on low-spec hardware for smooth, high-quality on-device synthesis across all devices.

## Development & Distribution Principles

- **Pure Mobile Lightweight Dev Ecosystem**: Completely decoupled from heavy IDEs and complex local compilation chains. Enables most development and deployment operations to be done directly on mobile devices.
- **Cloud Automated Build System**: All core low-level modules are built via cloud CI/CD, eliminating local environment configuration, dependency errors, and compilation failures. Enables rapid iteration.
- **Open-Source Core + Community Sponsorship**: The main framework and base features are fully open-source and free. High-tier custom engines and specialized optimization models are distributed via community sponsorship. No mandatory paywalls, relying entirely on community passion and recognition.

## License

MIT License
