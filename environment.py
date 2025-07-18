import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import random
from typing import List, Dict, Tuple
import copy

# 新增 Keras 相關導入（帶回退方案）
import pandas as pd
try:
    from keras.datasets import mnist
    from keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    print("⚠️  Keras 不可用，將使用 PyTorch MNIST")
    KERAS_AVAILABLE = False

class SimpleNN(nn.Module):
    """簡單的神經網絡模型用於聯邦學習"""
    
    def __init__(self, input_size: int = 784, hidden_size1: int = 128, 
                 hidden_size2: int = 64, num_classes: int = 10, dropout_rate: float = 0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class KerasMNISTDataset(Dataset):
    """Keras MNIST 數據集包裝類，使其與 PyTorch Dataset 接口兼容"""
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class PoisonedDataset(Dataset):
    """投毒數據集類"""
    
    def __init__(self, data_list: List, labels_list: List):
        self.data = data_list
        self.labels = labels_list
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class HonestClient:
    """誠實客戶端類"""
    
    def __init__(self, client_id: int, data: Dataset, input_size: int, 
                 num_classes: int, learning_rate: float = 0.01):
        self.client_id = client_id
        self.data = data
        self.data_loader = DataLoader(data, batch_size=32, shuffle=True)
        self.model = SimpleNN(input_size, num_classes=num_classes)
        self.learning_rate = learning_rate
        
    def train_local_model(self, global_model: nn.Module, epochs: int = 1) -> nn.Module:
        """基於全局模型進行本地訓練"""
        # 複製全局模型參數
        self.model.load_state_dict(global_model.state_dict())
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        return self.model
    
    def get_data_size(self):
        """獲取客戶端數據量"""
        return len(self.data)

class SybilClient:
    """Sybil 惡意客戶端類"""
    
    def __init__(self, client_id: str, original_dataset: Dataset, 
                 input_size: int, num_classes: int, poison_ratio: float = 0.3,
                 learning_rate: float = 0.05):
        self.client_id = client_id
        self.input_size = input_size
        self.num_classes = num_classes
        self.poison_ratio = poison_ratio
        self.learning_rate = learning_rate
        
        # 創建投毒數據集
        self.poisoned_data = self._create_poisoned_dataset(original_dataset)
        self.data_loader = DataLoader(self.poisoned_data, batch_size=32, shuffle=True)
        self.model = SimpleNN(input_size, num_classes=num_classes)
        
    def _create_poisoned_dataset(self, original_dataset: Dataset) -> Dataset:
        """創建投毒數據集，包含標籤翻轉和噪聲注入"""
        # 採樣原始數據的子集
        sample_size = min(1000, len(original_dataset) // 10)
        indices = random.sample(range(len(original_dataset)), sample_size)
        
        poisoned_samples = []
        poisoned_labels = []
        
        for idx in indices:
            data, label = original_dataset[idx]
            
            # 按照投毒比例進行投毒
            if random.random() < self.poison_ratio:
                # 策略1：標籤翻轉攻擊 - 隨機翻轉標籤
                poisoned_label = torch.tensor(random.randint(0, self.num_classes - 1), dtype=torch.long)
                
                # 策略2：數據噪聲注入
                noise = torch.randn_like(data) * 0.1
                poisoned_data = torch.clamp(data + noise, 0, 1)
                
                poisoned_samples.append(poisoned_data)
                poisoned_labels.append(poisoned_label)
            else:
                poisoned_samples.append(data)
                # 確保原始標籤也是 tensor
                if not isinstance(label, torch.Tensor):
                    label = torch.tensor(label, dtype=torch.long)
                poisoned_labels.append(label)
                
        return PoisonedDataset(poisoned_samples, poisoned_labels)
    
    def train_local_model(self, global_model: nn.Module, epochs: int = 1) -> nn.Module:
        """進行惡意訓練以最大化損害"""
        # 複製全局模型參數
        self.model.load_state_dict(global_model.state_dict())
        
        # 使用更強的攻擊策略
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                
                # 多種攻擊策略組合
                
                # 策略1: 對抗目標攻擊 - 將所有標籤指向錯誤的固定類別
                wrong_target = torch.full_like(target, (target[0].item() + 5) % self.num_classes)
                
                # 策略2: 添加強烈的模型擾動
                # 在計算損失之前添加對抗性擾動
                perturbation = torch.randn_like(output) * 0.5
                perturbed_output = output + perturbation
                
                # 使用錯誤標籤進行訓練
                loss = criterion(perturbed_output, wrong_target)
                
                # 策略3: 梯度反轉（模擬PoisonSGD攻擊）
                loss.backward()
                
                # 反轉梯度方向來破壞模型
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data *= -2.0  # 反轉並放大梯度
                
                # 添加梯度剪裁避免過度擾動
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # 策略4: 額外的參數擾動
                with torch.no_grad():
                    for param in self.model.parameters():
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)
                
        return self.model
    
    def get_data_size(self):
        """獲取客戶端數據量"""
        return len(self.poisoned_data)

class FederatedLearningEnvironment:
    """聯邦學習環境類"""
    
    def __init__(self, num_honest_clients: int = 5, num_sybil_clients: int = 3, 
                 dataset_name: str = 'MNIST', honest_lr: float = 0.01, 
                 sybil_lr: float = 0.05, poison_ratio: float = 0.3):
        
        self.num_honest_clients = num_honest_clients
        self.num_sybil_clients = num_sybil_clients
        self.total_clients = num_honest_clients + num_sybil_clients
        self.dataset_name = dataset_name
        self.honest_lr = honest_lr
        self.sybil_lr = sybil_lr
        self.poison_ratio = poison_ratio
        
        # 初始化客戶端列表
        self.honest_clients = []
        self.sybil_clients = []
        
        # 全局模型
        self.global_model = None
        
        # 訓練歷史記錄
        self.training_history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'honest_accuracy': [],
            'attack_strength': []
        }
        
        # 初始化環境
        self._setup_dataset()
        self._initialize_clients()
        
    def _setup_dataset(self):
        """設置數據集 - 優先使用 Keras，回退到 PyTorch"""
        if self.dataset_name == 'MNIST':
            if KERAS_AVAILABLE:
                # 使用 Keras 方式下載 MNIST 數據集
                print("\t[Info] 正在使用 Keras 下載 MNIST 數據集...")
                np.random.seed(10)
                
                # 下載 MNIST 數據
                (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
                print(f"\t[Info] train data={len(X_train_image):7,}")
                print(f"\t[Info] test  data={len(X_test_image):7,}")
                
                # 數據預處理：標準化到 [0,1] 並調整形狀
                X_train_norm = X_train_image.astype('float32') / 255.0
                X_test_norm = X_test_image.astype('float32') / 255.0
                
                # 轉換為 PyTorch tensor 格式以保持與現有代碼兼容
                self.train_data = torch.from_numpy(X_train_norm).unsqueeze(1)  # 添加通道維度
                self.train_labels = torch.from_numpy(y_train_label).long()
                self.test_data = torch.from_numpy(X_test_norm).unsqueeze(1)
                self.test_labels = torch.from_numpy(y_test_label).long()
                
                # 創建 PyTorch Dataset
                self.train_dataset = KerasMNISTDataset(self.train_data, self.train_labels)
                self.test_dataset = KerasMNISTDataset(self.test_data, self.test_labels)
            else:
                # 回退到 PyTorch 的 MNIST 數據集
                print("\t[Info] 正在使用 PyTorch 下載 MNIST 數據集...")
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                
                self.train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
                self.test_dataset = datasets.MNIST('./data', train=False, transform=transform)
                print(f"\t[Info] train data={len(self.train_dataset):7,}")
                print(f"\t[Info] test  data={len(self.test_dataset):7,}")
            
            self.num_classes = 10
            self.input_size = 28 * 28
        else:
            raise ValueError(f"不支持的數據集: {self.dataset_name}")
            
        # 創建測試數據加載器
        self.test_loader = DataLoader(self.test_dataset, batch_size=1000, shuffle=False)
        
    def _initialize_clients(self):
        """Initialize clients with arbitrary numbers of honest and sybil clients."""
        # Honest clients (if any)
        if self.num_honest_clients > 0:
            data_per_client = len(self.train_dataset) // self.num_honest_clients
            for i in range(self.num_honest_clients):
                start_idx = i * data_per_client
                end_idx = (i + 1) * data_per_client if i < self.num_honest_clients - 1 else len(self.train_dataset)
                client_data = Subset(self.train_dataset, range(start_idx, end_idx))
                client = HonestClient(
                    client_id=i, 
                    data=client_data, 
                    input_size=self.input_size,
                    num_classes=self.num_classes,
                    learning_rate=self.honest_lr
                )
                self.honest_clients.append(client)
        # Sybil clients (can be any number, even all clients)
        for i in range(self.num_sybil_clients):
            client = SybilClient(
                client_id=f"sybil_{i}",
                original_dataset=self.train_dataset,
                input_size=self.input_size,
                num_classes=self.num_classes,
                poison_ratio=self.poison_ratio,
                learning_rate=self.sybil_lr
            )
            self.sybil_clients.append(client)
            
    def get_global_model(self):
        """獲取或初始化全局模型"""
        if self.global_model is None:
            self.global_model = SimpleNN(self.input_size, num_classes=self.num_classes)
        return self.global_model
    
    def evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """評估模型在測試集上的性能"""
        model.eval()
        correct = 0
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        accuracy = correct / len(self.test_dataset)
        avg_loss = total_loss / len(self.test_loader)
        
        return accuracy, avg_loss
    
    def get_environment_info(self) -> Dict:
        """Get environment info, robust to any honest/sybil client combination."""
        total_clients = self.num_honest_clients + self.num_sybil_clients
        sybil_ratio = self.num_sybil_clients / total_clients if total_clients > 0 else 0
        return {
            'num_honest_clients': self.num_honest_clients,
            'num_sybil_clients': self.num_sybil_clients,
            'total_clients': total_clients,
            'sybil_ratio': sybil_ratio,
            'dataset': self.dataset_name,
            'num_classes': self.num_classes,
            'honest_lr': self.honest_lr,
            'sybil_lr': self.sybil_lr,
            'poison_ratio': self.poison_ratio
        }

def create_environment(config) -> FederatedLearningEnvironment:
    """根據配置創建聯邦學習環境"""
    return FederatedLearningEnvironment(
        num_honest_clients=config.NUM_HONEST_CLIENTS,
        num_sybil_clients=config.NUM_SYBIL_CLIENTS,
        dataset_name=config.DATASET_NAME,
        honest_lr=config.LEARNING_RATE_HONEST,
        sybil_lr=config.LEARNING_RATE_SYBIL,
        poison_ratio=config.POISON_RATIO
    ) 