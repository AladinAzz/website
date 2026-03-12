import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Shared model state ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
optimizer = None
criterion = nn.BCELoss()

# ── Model Architecture ────────────────────────────────────────────────────────
class NeuralNet(nn.Module):
    def __init__(self, hidden_layers):
        super(NeuralNet, self).__init__()
        layers = []
        input_size = 2
        
        for units, activation in hidden_layers:
            layers.append(nn.Linear(input_size, units))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            input_size = units
            
        layers.append(nn.Linear(input_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/api/build', methods=['POST'])
def build():
    global model, optimizer
    try:
        data = request.json
        hidden_layers = data.get('hiddenLayers', [[4, 'relu'], [2, 'relu']])
        learning_rate = data.get('learningRate', 0.01)
        
        model = NeuralNet(hidden_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"✅ Model built on {device}")
        return jsonify({"ok": True})
    except Exception as e:
        print(f"Build error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    global model, optimizer
    try:
        if model is None:
            return jsonify({"error": "No model built"}), 400
            
        data = request.json
        train_x = torch.tensor(data['trainX'], dtype=torch.float32).to(device)
        train_y = torch.tensor(data['trainY'], dtype=torch.float32).view(-1, 1).to(device)
        test_x = torch.tensor(data['testX'], dtype=torch.float32).to(device)
        test_y = torch.tensor(data['testY'], dtype=torch.float32).view(-1, 1).to(device)
        batch_size = max(1, data.get('batchSize', 32))
        epochs = data.get('epochs', 5)
        
        losses = []
        for epoch in range(epochs):
            model.train()
            # Simple batching (to keep it fast without complex DataLoader)
            permutation = torch.randperm(train_x.size()[0])
            epoch_loss = 0
            for i in range(0, train_x.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = train_x[indices], train_y[indices]
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_x)
                test_loss = criterion(test_outputs, test_y).item()
                
            losses.append({
                "trainLoss": epoch_loss / (train_x.size()[0] / batch_size),
                "testLoss": test_loss
            })
            
        return jsonify({"ok": True, "epochs": epochs, "losses": losses})
    except Exception as e:
        print(f"Train error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/grid', methods=['GET'])
def get_grid():
    try:
        if model is None:
            return jsonify({"error": "No model built"}), 400
            
        n = 50
        model.eval()
        with torch.no_grad():
            x_range = np.linspace(-1.5, 1.5, n)
            y_range = np.linspace(-1.5, 1.5, n)
            grid_x_np, grid_y_np = np.meshgrid(x_range, y_range)
            
            xs = np.stack([grid_x_np.ravel(), grid_y_np.ravel()], axis=1)
            xs_torch = torch.tensor(xs, dtype=torch.float32).to(device)
            
            output = model(xs_torch).cpu().numpy().reshape(n, n)
            
            return jsonify({
                "gridX": grid_x_np.tolist(),
                "gridY": grid_y_np.tolist(),
                "gridZ": output.tolist()
            })
    except Exception as e:
        print(f"Grid error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/neuron-grid', methods=['GET'])
def get_neuron_grid():
    try:
        if model is None:
            return jsonify({"error": "No model built"}), 400
            
        layer_idx = int(request.args.get('layer', 0))
        neuron_idx = int(request.args.get('neuron', 0))
        n = 20
        
        # In PyTorch sequential, linear layers are at even indices (0, 2, 4...)
        # because the odd indices are activations.
        torch_layer_idx = layer_idx * 2
        
        # Sub-model for intermediate output
        sub_network = model.network[:torch_layer_idx + 1]
        
        model.eval()
        with torch.no_grad():
            x_range = np.linspace(-1.5, 1.5, n)
            y_range = np.linspace(-1.5, 1.5, n)
            grid_x_np, grid_y_np = np.meshgrid(x_range, y_range)
            
            xs = np.stack([grid_x_np.ravel(), grid_y_np.ravel()], axis=1)
            xs_torch = torch.tensor(xs, dtype=torch.float32).to(device)
            
            output = sub_network(xs_torch).cpu().numpy()
            # output shape: (n*n, num_neurons)
            
            neuron_output = output[:, neuron_idx].reshape(n, n)
            
            return jsonify({"grid": neuron_output.tolist()})
    except Exception as e:
        print(f"Neuron grid error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/weights', methods=['GET'])
def get_weights():
    try:
        if model is None:
            return jsonify({"error": "No model built"}), 400
            
        weights_data = []
        for i, layer in enumerate(model.network):
            if isinstance(layer, nn.Linear):
                # Only include hidden layers for weight visualization
                # (matching the filter in TF.js logic)
                w = layer.weight.detach().cpu().numpy().T.tolist()
                weights_data.append({
                    "layerIndex": i // 2,
                    "weights": w
                })
                
        # The last layer is the output layer, which usually isn't shown in the same way 
        # in the frontend's NetworkGraph weight logic.
        return jsonify({"weights": weights_data[:-1]})
    except Exception as e:
        print(f"Weights error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "backend": "torch",
        "isCuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "connected": True
    })

if __name__ == '__main__':
    print(f"\n🚀 NeuralViz CUDA Server (Python/PyTorch)")
    print(f"   Device : {device}")
    if torch.cuda.is_available():
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   Port   : 3001")
    print(f"   Ready!\n")
    app.run(port=3001)
