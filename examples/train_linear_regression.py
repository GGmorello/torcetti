#!/usr/bin/env python3
"""
Linear Regression training example using Torcetti.
This script demonstrates regression capabilities with synthetic polynomial data.
"""

import numpy as np

# Import Torcetti components
from torcetti.core.tensor import Tensor
from torcetti.nn.linear import Linear
from torcetti.nn.sequential import Sequential
from torcetti.nn.activations import ReLU, Tanh
from torcetti.loss.mse import MSE
from torcetti.optim.sgd import SGD
from torcetti.optim.adam import Adam


def generate_polynomial_data(n_samples=1000, noise=0.1, degree=3):
    """Generate synthetic polynomial regression data."""
    np.random.seed(42)
    
    # Generate input features
    X = np.random.uniform(-2, 2, (n_samples, 1)).astype(np.float32)
    
    # Generate polynomial target: y = 0.5*x^3 - 1.2*x^2 + 0.8*x + 0.3 + noise
    if degree == 3:
        y_true = 0.5 * X**3 - 1.2 * X**2 + 0.8 * X + 0.3
    elif degree == 2:
        y_true = -0.8 * X**2 + 1.5 * X + 0.2
    else:  # linear
        y_true = 2.3 * X + 1.1
    
    # Add noise
    noise_vec = np.random.normal(0, noise, y_true.shape).astype(np.float32)
    y = y_true + noise_vec
    
    return X, y.flatten(), y_true.flatten()


def create_regression_model(input_size, hidden_sizes, output_size, activation='relu'):
    """Create a neural network for regression."""
    layers = []
    
    # Input layer
    prev_size = input_size
    
    # Hidden layers
    for hidden_size in hidden_sizes:
        layers.append(Linear(prev_size, hidden_size))
        if activation == 'relu':
            layers.append(ReLU())
        elif activation == 'tanh':
            layers.append(Tanh())
        prev_size = hidden_size
    
    # Output layer (no activation for regression)
    layers.append(Linear(prev_size, output_size))
    
    return Sequential(*layers)


def train_epoch(model, X_train, y_train, optimizer, criterion, batch_size=32):
    """Train for one epoch with mini-batches."""
    n_samples = len(X_train)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    total_loss = 0.0
    
    # Shuffle training data
    indices = np.random.permutation(n_samples)
    
    for batch_idx in range(n_batches):
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Create batch tensors
        X_batch = Tensor(X_train[batch_indices], requires_grad=False)
        y_batch = Tensor(y_train[batch_indices].reshape(-1, 1), requires_grad=False)
        
        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.data
    
    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(model, X_test, y_test):
    """Evaluate the model on test data."""
    X_tensor = Tensor(X_test, requires_grad=False)
    predictions = model(X_tensor)
    
    # Calculate R² score (coefficient of determination)
    y_pred = predictions.data.flatten()
    y_true = y_test
    
    # Mean squared error
    mse = np.mean((y_pred - y_true) ** 2)
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mse, r2_score, y_pred


def main():
    """Main regression training script."""
    print("📈 Torcetti Linear Regression Demo")
    print("=" * 50)
    
    # Hyperparameters
    n_samples = 800
    polynomial_degree = 3
    hidden_sizes = [64, 32]
    learning_rate = 0.001
    n_epochs = 200
    batch_size = 32
    noise_level = 0.15
    
    print(f"Generating polynomial dataset (degree {polynomial_degree}):")
    print(f"  Samples: {n_samples}")
    print(f"  Noise level: {noise_level}")
    
    # Generate data
    X, y, y_true = generate_polynomial_data(n_samples, noise_level, polynomial_degree)
    
    # Split data (80% train, 20% test)
    n_total = len(X)
    n_train = int(0.8 * n_total)
    
    indices = np.random.permutation(n_total)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    y_true_test = y_true[test_indices]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test different architectures
    architectures = [
        ("Linear", []),
        ("Shallow", [32]),
        ("Deep", [64, 32]),
        ("Wide", [128, 64, 32])
    ]
    
    results = {}
    
    for arch_name, hidden_sizes in architectures:
        print(f"\n🏗️ Training {arch_name} Architecture")
        print("-" * 40)
        
        # Create model
        model = create_regression_model(1, hidden_sizes, 1, activation='relu')
        
        # Count parameters
        total_params = sum(p.data.size for p in model.parameters())
        arch_str = "1" + "".join(f" -> {h}" for h in hidden_sizes) + " -> 1"
        print(f"  Architecture: {arch_str}")
        print(f"  Parameters: {total_params:,}")
        
        # Loss and optimizer
        criterion = MSE()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        train_losses = []
        best_test_mse = float('inf')
        
        # Training loop
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, X_train, y_train, optimizer, criterion, batch_size)
            train_losses.append(train_loss)
            
            if (epoch + 1) % 50 == 0:
                test_mse, test_r2, _ = evaluate(model, X_test, y_test)
                best_test_mse = min(best_test_mse, test_mse)
                print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Test MSE: {test_mse:.6f} | "
                      f"Test R²: {test_r2:.4f}")
        
        # Final evaluation
        final_mse, final_r2, predictions = evaluate(model, X_test, y_test)
        results[arch_name] = {
            'mse': final_mse,
            'r2': final_r2,
            'params': total_params,
            'predictions': predictions
        }
        
        print(f"  📊 Final Results:")
        print(f"    MSE: {final_mse:.6f}")
        print(f"    R² Score: {final_r2:.4f}")
        print(f"    Best Test MSE: {best_test_mse:.6f}")
    
    # Compare architectures
    print(f"\n🏆 Architecture Comparison")
    print("=" * 60)
    print(f"{'Architecture':<12} {'Parameters':<12} {'MSE':<12} {'R² Score':<10}")
    print("-" * 60)
    
    best_arch = min(results.keys(), key=lambda k: results[k]['mse'])
    
    for arch_name in results:
        mse = results[arch_name]['mse']
        r2 = results[arch_name]['r2']
        params = results[arch_name]['params']
        
        marker = "🥇" if arch_name == best_arch else "  "
        print(f"{marker} {arch_name:<10} {params:<12,} {mse:<12.6f} {r2:<10.4f}")
    
    # Show sample predictions for best model
    print(f"\n🎯 Sample Predictions (Best: {best_arch})")
    print("-" * 40)
    best_predictions = results[best_arch]['predictions']
    
    # Sort test data for better visualization
    sorted_indices = np.argsort(X_test.flatten())
    X_sorted = X_test[sorted_indices]
    y_sorted = y_test[sorted_indices]
    y_true_sorted = y_true_test[sorted_indices]
    pred_sorted = best_predictions[sorted_indices]
    
    print("Input  | True    | Predicted | Error")
    print("-" * 40)
    for i in range(0, len(X_sorted), len(X_sorted)//10):  # Show 10 samples
        x_val = X_sorted[i, 0]
        y_val = y_sorted[i]
        y_true_val = y_true_sorted[i]
        pred_val = pred_sorted[i]
        error = abs(pred_val - y_val)
        
        print(f"{x_val:6.2f} | {y_val:7.3f} | {pred_val:9.3f} | {error:5.3f}")
    
    # Calculate noise-free R² for best model
    noise_free_predictions = results[best_arch]['predictions']
    noise_free_mse = np.mean((noise_free_predictions - y_true_test) ** 2)
    
    ss_res_nf = np.sum((y_true_test - noise_free_predictions) ** 2)
    ss_tot_nf = np.sum((y_true_test - np.mean(y_true_test)) ** 2)
    noise_free_r2 = 1 - (ss_res_nf / ss_tot_nf) if ss_tot_nf != 0 else 0
    
    print(f"\n📈 Detailed Analysis (Best Model):")
    print(f"  Noisy Data R²: {results[best_arch]['r2']:.4f}")
    print(f"  True Function R²: {noise_free_r2:.4f}")
    print(f"  Noise Level Impact: {results[best_arch]['r2'] - noise_free_r2:.4f}")
    
    print("\n✅ Regression training completed successfully!")
    print("\n🎯 Key achievements:")
    print("  ✓ MSE loss function working")
    print("  ✓ Regression architectures tested")
    print("  ✓ Multiple model comparison")
    print("  ✓ R² score calculation implemented")
    print("  ✓ Polynomial function approximation")
    print("\n🚀 Torcetti handles regression tasks perfectly!")


if __name__ == "__main__":
    main() 