# FL Breaker  
**Federated Learning Breaker: Sybil-Based Data Poisoning Attack Simulator**  

## Description  
**FL Breaker** is a simulation tool for Sybil-based untargeted data poisoning attacks in Federated Learning (FL) systems.  

This project was developed for the **NUS Summer Workshop 2025 DADA class**, and is based on the research from the paper:

> [*"SPoiL: Sybil-Based Untargeted Data Poisoning Attacks in Federated Learning"*](https://dl.acm.org/doi/abs/10.1007/978-3-031-39828-5_13)  

It also builds upon the implementation from:  
[YunshiuanOAO/Sybil-Based-Data-Poisoning-Attacks-in-Federated-Learning-Poc](https://github.com/YunshiuanOAO/Sybil-Based-Data-Poisoning-Attacks-in-Federated-Learning-Poc)

The tool simulates how malicious actors can use Sybil clients to manipulate FL model updates through data poisoning and gradient amplification.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShreyPaul/untargeted-poisoning.git
   cd fl-breaker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Usage

Run the simulation with customizable parameters:

```bash
python main.py [OPTIONS]
```

### 🔧 Example Commands

- **Default attack simulation**:
  ```bash
  python main.py
  ```

- **Check environment setup only**:
  ```bash
  python main.py --setup-only
  ```

- **Simulate large-scale Sybil attack**:
  ```bash
  python main.py --rounds 40 --start-round 20 --honest-clients 10 --sybil-clients 1 --num-sybil-per-malicious 10
  ```

---

## ⚙️ Parameters

| Option                    | Default       | Description |
|---------------------------|---------------|-------------|
| `--rounds`                | 15            | Number of training rounds |
| `--start-round`           | 2             | Round to start attacks |
| `--honest-clients`        | 5             | Number of honest clients |
| `--sybil-clients`         | 3             | Number of malicious Sybil clients |
| `--num-sybil-per-malicious` | 64         | Number of Sybil nodes per malicious client |
| `--amplification-factor`  | 5.0           | Gradient amplification factor |
| `--poison-ratio`          | 1.0           | Fraction of poisoned data |
| `--attack-module`         | `attack`      | Use `attack` or `amplify_attack` |
| `--output`                | Auto-generated | Output JSON file name |
| `--quiet`                 | False         | Reduce console output |
| `--setup-only`            | False         | Validate environment only |

---

## 🧠 Attack Modules

- **`attack`**: Standard Sybil-based poisoning attack.
- **`amplify_attack`**: Uses gradient amplification to enhance attack effectiveness (as described in the paper).

---

## 📤 Output

Simulation results are saved in JSON format:

- **Default filename**: `sybil_attack_results_YYYYMMDD_HHMMSS.json`
- Contains:
  - Global model accuracy per round
  - Attack success metrics
  - Detailed logs of each training round
