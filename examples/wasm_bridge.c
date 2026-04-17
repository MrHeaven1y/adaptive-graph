#include <emscripten.h>
#include <stdlib.h>
#include <stdio.h>

#include "graph.h"
#include "layers.h"
#include "optimizers.h"
#include "ops.h"

#define MAX_LAYERS 4
// Compile:
//   emcc wasm_bridge.c src/*.c -Iinclude -lm \
//     -s EXPORTED_FUNCTIONS="['_init_network','_train_epoch','_predict_grid','_malloc','_free']" \
//     -s EXPORTED_RUNTIME_METHODS="['cwrap','HEAPF64']" \
//     -s ALLOW_MEMORY_GROWTH=1 \
//     -o net.js

static GraphContext* ctx = NULL;
static Model*        model = NULL;
static Optimizer*    opt = NULL;

// Dynamic layer storage
static DenseLayer* hidden_layers[MAX_LAYERS];
static DenseLayer* output_layer = NULL;
static int layer_sizes[MAX_LAYERS];
static int current_num_layers = 0;
static int current_batch_size = 4;

// Initialize with up to 4 hidden layers. 
// Pass 0 for h2, h3, or h4 if you want fewer layers.
EMSCRIPTEN_KEEPALIVE
void init_network(int num_layers, int h1, int h2, int h3, int h4, double learning_rate, int batch_size) {

    if (ctx) {
        reset_graph(ctx);
        free(ctx->tape);
        free(ctx->params);
        free(ctx->inputs);
        free(ctx);
    }
    if (model) { free(model->params); free(model); }
    if (opt)   { free(opt->velocity); free(opt); }

    current_batch_size = batch_size;
    current_num_layers = (num_layers > MAX_LAYERS) ? MAX_LAYERS: num_layers;

    int requested_sizes[4] = {h1, h2, h3, h4};
    for(int i=0; i < current_num_layers; i++){
        layer_sizes[i] = requested_sizes[i];
    }

    ctx = init_graph(10000, 1000, 256);
    model = init_model(256);

    int prev_size = 2; // Input is always (x, y)

    for(int i = 0; i < current_num_layers; i++){
        
        hidden_layers[i] = _dense_layer(ctx, model, prev_size, layer_sizes[i], batch_size);
        prev_size = layer_sizes[i];
        
    }

    output_layer = _dense_layer(ctx, model, prev_size, 1, batch_size);

    opt = sgd(model->params, model->count, batch_size, learning_rate, 0.9);

}

// Helper function for the dynamic forward pass
// Returns the final prediction tensor
Tensor* dynamic_forward_pass(Tensor** initial_inputs) {
    Tensor** current_in = initial_inputs;

    // We must track these intermediate arrays to free them and prevent Wasm memory leaks
    Tensor*** all_h = (Tensor***)malloc(sizeof(Tensor**) * current_num_layers);
    Tensor*** all_h_act = (Tensor***)malloc(sizeof(Tensor**) * current_num_layers);

    for(int i=0; i < current_num_layers; i++){
        
        all_h[i] = Dense(ctx, hidden_layers[i], current_in);
        all_h_act[i] = (Tensor**)malloc(sizeof(Tensor*) * layer_sizes[i]);
        
        for(int j=0; j < layer_sizes[i]; j++){
            all_h_act[i][j] = Tanh(ctx, all_h[i][j]);
        }

        // The output of this layer becomes the input of the next
        current_in = all_h_act[i];
    }

    Tensor** out = Dense(ctx, output_layer, current_in);
    Tensor* pred = Sigmoid(ctx, out[0]);

    for(int i=0; i < current_num_layers; i++){
        free(all_h[i]);
        free(all_h_act[i]);
    }

    free(all_h);
    free(all_h_act);
    free(out);

    return pred;

}

// Train one full epoch. Returns MSE loss averaged over the batch.
EMSCRIPTEN_KEEPALIVE
double train_epoch(double* x0_data, double* x1_data, double* y_data) {

    Tensor* inputs[2];
    inputs[0] = Input(ctx, x0_data, current_batch_size);
    inputs[1] = Input(ctx, x1_data, current_batch_size);
    Tensor* target = Input(ctx, y_data, current_batch_size);

    Tensor* pred = dynamic_forward_pass(inputs);
    Tensor* loss = Mse(ctx, target, pred);

    backward(ctx, loss);
    opt->step(opt);

    double current_loss = loss->data[0];
    reset_tape(ctx);

    return current_loss;

}

// Run forward pass over an arbitrary grid/point set.
// Temporarily resizes model weights to num_points, infers, then restores batch_size.
EMSCRIPTEN_KEEPALIVE
void predict_grid(double* grid_x0, double* grid_x1, double* out_preds, int num_points) {

    resize_model_batch(model, num_points);

    Tensor* inputs[2];
    inputs[0] = Input(ctx, grid_x0, num_points);
    inputs[1] = Input(ctx, grid_x1, num_points);

    Tensor* pred = dynamic_forward_pass(inputs);

    for(int i=0; i < num_points; i++){
        out_preds[i] = pred->data[i];
    }

    reset_tape(ctx);
    resize_model_batch(model, current_batch_size);
}
