# Vector Playground (C + WebAssembly Neural Network Visualizer)

An interactive neural network playground powered by a custom-built 
autograd and tensor engine written in C and compiled to WebAssembly.

Recreates core ideas behind TensorFlow Playground — but from scratch at the systems level.

# Some Visuals:
1. <img width="1920" height="928" alt="Image" src="https://github.com/user-attachments/assets/1d61c52b-dcfc-448d-867c-83de65930e3d" />
2. <img width="1490" height="720" alt="Image" src="https://github.com/user-attachments/assets/3ecd1870-45d1-46aa-a639-218825ad1e38" />
3. <img width="1490" height="720" alt="Image" src="https://github.com/user-attachments/assets/f95f347e-47b4-49fa-89fa-433b3812b0db" />
4. <img width="1490" height="720" alt="Image" src="https://github.com/user-attachments/assets/f2efac1a-d0ae-43b1-9e03-bb03724b8a6d" />

## Why this project matters

Most ML tools abstract away the internals.

This project exposes:
- how computation graphs are built
- how gradients flow
- how training actually happens

All powered by a low-level C backend running in the browser via WebAssembly.

## 🚀 Overview

This repository bridges a low-level C architecture with a sleek, reactive HTML/JS frontend. It allows users to dynamically build neural networks, adjust hyperparameters, and watch the learning process happen in real-time across various datasets. 

### Core Components
* **`playground.html`**: The main frontend visualization engine. It provides the UI for tweaking learning rates, adding/removing layers, and visualizing the decision boundary and loss curve. It falls back to a vanilla JS engine if the Wasm binary isn't found.
* **`wasm_bridge.c`**: The Emscripten bridge that exposes the C engine's functions (like `init_network`, `train_epoch`, and `predict_grid`) to JavaScript. It also manages Wasm memory allocations to prevent leaks during dynamic layer creation.
* **`morph_graph_animator.html`**: A standalone interactive graph morphing tool. It was just added to the repository for fun so you can click around and watch mathematical functions morph into curves.

## Architecture

Frontend (JS/HTML)
        ↓
WASM Bridge (Emscripten)
        ↓
C Autograd Engine
        ↓
Tensor Ops + Graph Execution

## 🧠 Architecture Notes

The underlying C engine (often referred to as Scaler-Engine or Vector Engine) is designed to be highly memory-efficient, targeting constrained or embedded environments. 

* **Stateless Tensors:** To maintain a minimal memory footprint, the core `Tensor` data structures **do not have gradients**.
* **External Context:** All gradients, computation graphs, and topological sorting operations are managed externally via a `GraphContext` (`ctx->tape`). 
* **Dynamic Memory:** The `wasm_bridge.c` handles dynamic layer sizing and batch resizing, ensuring that intermediate tensor arrays are tracked and freed correctly during the forward pass.

## 🛠️ Build Instructions

To compile the C backend into WebAssembly, you will need [Emscripten](https://emscripten.org/) installed and activated.

1. **Compile the engine:**
   Navigate to the root of the project and run the following Emscripten compiler command. This will generate the necessary `net.js` and `net.wasm` files.

   ```bash
   emcc wasm_bridge.c src/*.c -Iinclude -lm \
     -s EXPORTED_FUNCTIONS="['_init_network','_train_epoch','_predict_grid','_malloc','_free']" \
     -s EXPORTED_RUNTIME_METHODS="['cwrap','HEAPF64']" \
     -s ALLOW_MEMORY_GROWTH=1 \
     -o build/net.js
   ```

2. **Serve the project:**
   WebAssembly requires a local web server to bypass browser CORS restrictions. You can use Python to spin one up quickly:

   ```bash
   python3 -m http.server 8000
   ```

3. **Run it:**
   Open your browser and navigate to `http://localhost:8000/playground.html`. Ensure the status bar indicates that `WASM: ON` to confirm the C engine has successfully loaded.

## 📁 Project Structure

```text
adaptive-graph/
├── src/                        # C backend source files (engine, graph, ops)
├── include/                    # C header files (graph.h, layers.h, etc.)
├── build/                      # Compiled WebAssembly artifacts (net.js, net.wasm)
│
├── examples/
│   └── wasm_bridge.c           # C ↔ WASM bridge + memory interface
│
├── web/                        # Frontend (visualizer + playground)
│   ├── playground.html         # Main UI for interacting with the engine
│   └── morph_graph_animator.html  # Un-related Graph morphing / visualization toy
│
└── README.md
```

## 👨‍💻 Author

**Dibyendu Mukherjee**
* Contact: dibyendumukherjee916@gmail.con
