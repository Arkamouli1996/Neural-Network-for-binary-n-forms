import numpy as np
import torch
import torch.nn as nn
import json

# Load scaler parameters
with open('/Users/zengrf/Documents/GitHub/machine-learning-math/triple-root-locus/python/scaler_params.json', 'r') as f:
    scaler_params = json.load(f)

mean = np.array(scaler_params['mean'])
scale = np.array(scaler_params['scale'])

# Define model architecture (must match training)
class TripleRootClassifier(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=512):
        super(TripleRootClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TripleRootClassifier(input_dim=10, hidden_dim=2048).to(device)
model.load_state_dict(torch.load('/Users/zengrf/Documents/GitHub/machine-learning-math/triple-root-locus/python/triple_root_model.pth', map_location=device))
model.eval()

print("Model loaded successfully!\n")

# Test function
def test_polynomial(coeffs_real, coeffs_imag=None):
    """
    Test if a binary quartic has a triple root.
    coeffs_real: [a0, a1, a2, a3, a4] - real parts
    coeffs_imag: [a0, a1, a2, a3, a4] - imaginary parts (optional, defaults to zeros)
    """
    if coeffs_imag is None:
        coeffs_imag = np.zeros(5)
    
    # Normalize to projective coordinates
    combined = np.array(list(coeffs_real) + list(coeffs_imag))
    max_val = np.max(np.abs(combined[:5]) + np.abs(combined[5:]))
    if max_val > 1e-10:
        combined /= max_val
    
    # Standardize
    combined_scaled = (combined - mean) / scale
    
    # Predict
    with torch.no_grad():
        x = torch.FloatTensor(combined_scaled).unsqueeze(0).to(device)
        prob = model(x).item()
    
    return prob

# Helper function to create triple root polynomial
def create_triple_root_poly(L_coeffs, M_coeffs):
    """
    Create L^3 * M where L and M are linear forms.
    L_coeffs = [u, v] for L = u*x + v*y
    M_coeffs = [s, t] for M = s*x + t*y
    Returns [a0, a1, a2, a3, a4] for a0*x^4 + a1*x^3*y + a2*x^2*y^2 + a3*x*y^3 + a4*y^4
    """
    u, v = L_coeffs
    s, t = M_coeffs
    
    # L^3 = (ux + vy)^3
    L3 = [u**3, 3*u**2*v, 3*u*v**2, v**3]
    
    # L^3 * M = (ux + vy)^3 * (sx + ty)
    a0 = L3[0] * s
    a1 = L3[0] * t + L3[1] * s
    a2 = L3[1] * t + L3[2] * s
    a3 = L3[2] * t + L3[3] * s
    a4 = L3[3] * t
    
    return np.array([a0, a1, a2, a3, a4])

print("="*60)
print("TESTING NEURAL NETWORK")
print("="*60)

# Test 1: Known triple root examples
print("\n1. Testing polynomials WITH triple roots (L^3*M):")
print("-" * 60)

test_cases_positive = [
    {"L": [1, 0], "M": [1, 1], "name": "x^3 * (x+y)"},
    {"L": [1, 1], "M": [1, -1], "name": "(x+y)^3 * (x-y)"},
    {"L": [2, 1], "M": [1, 3], "name": "(2x+y)^3 * (x+3y)"},
    {"L": [1, 2], "M": [3, 1], "name": "(x+2y)^3 * (3x+y)"},
    {"L": [1, 0], "M": [0, 1], "name": "x^3 * y = x^3*y"},
]

for i, tc in enumerate(test_cases_positive, 1):
    coeffs = create_triple_root_poly(tc["L"], tc["M"])
    prob = test_polynomial(coeffs)
    print(f"  Test {i}: {tc['name']}")
    print(f"    Coefficients: {coeffs}")
    print(f"    Prediction: {prob:.6f} → {'✓ HAS triple root' if prob > 0.5 else '✗ NO triple root'}")
    print()

# Test 2: Generic polynomials (should not have triple roots)
print("\n2. Testing polynomials WITHOUT triple roots (generic):")
print("-" * 60)

test_cases_negative = [
    {"coeffs": [1, -6, 11, -6, 0], "name": "x(x-1)(x-2)(x-3)"},
    {"coeffs": [1, -2, -5, 6, 0], "name": "x(x-1)(x+2)(x-3)"},
    # {"coeffs": [1, 0, -2, 0, 1], "name": "x^4 - 2x^2y^2 + y^4"},
    {"coeffs": [1, 0, 0, 0, 0], "name": "x^4"},
    {"coeffs": [1, 2, 3, 2, 1], "name": "x^4 + 2x^3y + 3x^2y^2 + 2xy^3 + y^4"},
    {"coeffs": [1, -1, 1, -1, 1], "name": "x^4 - x^3y + x^2y^2 - xy^3 + y^4"},
]

for i, tc in enumerate(test_cases_negative, 1):
    coeffs = np.array(tc["coeffs"])
    prob = test_polynomial(coeffs)
    print(f"  Test {i}: {tc['name']}")
    print(f"    Coefficients: {coeffs}")
    print(f"    Prediction: {prob:.6f} → {'✓ HAS triple root' if prob > 0.5 else '✗ NO triple root'}")
    print()

# Test 3: Double root examples (L^2 * M * N) - should NOT have triple root
print("\n3. Testing polynomials with DOUBLE roots (L^2*M*N):")
print("-" * 60)

def create_double_root_poly(L, M, N):
    """L^2 * M * N"""
    u, v = L
    s, t = M
    r, q = N
    
    # L^2 = (ux + vy)^2
    L2 = [u**2, 2*u*v, v**2]
    
    # M*N
    MN = [s*r, s*q + t*r, t*q]
    
    # L^2 * M * N - convolve
    result = [0, 0, 0, 0, 0]
    for i in range(3):
        for j in range(3):
            result[i+j] += L2[i] * MN[j]
    
    return np.array(result)

test_cases_double = [
    {"L": [1, 0], "M": [1, 1], "N": [1, -1], "name": "x^2 * (x+y) * (x-y)"},
    {"L": [1, 1], "M": [1, 0], "N": [0, 1], "name": "(x+y)^2 * x * y"},
]

for i, tc in enumerate(test_cases_double, 1):
    coeffs = create_double_root_poly(tc["L"], tc["M"], tc["N"])
    prob = test_polynomial(coeffs)
    print(f"  Test {i}: {tc['name']}")
    print(f"    Coefficients: {coeffs}")
    print(f"    Prediction: {prob:.6f} → {'✓ HAS triple root' if prob > 0.5 else '✗ NO triple root'}")
    print()

# Test 4: Random polynomials
print("\n4. Testing 10 random polynomials:")
print("-" * 60)
np.random.seed(42)
correct = 0
for i in range(10):
    coeffs = np.random.randn(5)
    prob = test_polynomial(coeffs)
    expected = "NO triple root"
    prediction = "HAS triple root" if prob > 0.5 else "NO triple root"
    match = "✓" if prediction == expected else "✗"
    if prediction == expected:
        correct += 1
    print(f"  Random {i+1}: pred={prob:.4f} → {prediction} {match}")

print(f"\nRandom test accuracy: {correct}/10 = {correct*10}%")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
