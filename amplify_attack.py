import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple, Any
from environment import FederatedLearningEnvironment

# Only keep the SPoiL aggressive scenario
default_spoil_aggressive = {
    'attack_start_round': 2,
    'total_rounds': 15,
    'num_sybil_per_malicious': 100,
    'attack_method': 'label_flipping', 
    'flip_ratio': 1,
    'description': 'Aggressive SPoiL attack - high flip ratio, many Sybil'
}

ATTACK_SCENARIOS = {
    'spoil_aggressive': default_spoil_aggressive
}

class SybilVirtualDataAttackOrchestrator:
    """
    Orchestrator for SPoiL aggressive label flipping attack only.
    """
    def __init__(self, environment, num_sybil_per_malicious=8, amplification_factor=5.0):
        self.environment = environment
        self.attack_history = []
        self.current_round = 0
        self.attack_active = False
        self.attack_start_round = 2
        self.num_sybil_per_malicious = num_sybil_per_malicious
        self.amplification_factor = amplification_factor
        self.virtual_datasets = {}
        self.target_model = None
        self.attack_method = 'label_flipping'
        self.current_scenario = 'spoil_aggressive'

    def simple_label_flipping_attack(self, global_model, flip_ratio=1.0):
        sybil_models = []
        if len(self.environment.sybil_clients) == 0:
            return sybil_models
        for mal_idx, malicious_client in enumerate(self.environment.sybil_clients):
            original_data = []
            original_labels = []
            for data, labels in malicious_client.data_loader:
                original_data.append(data)
                original_labels.append(labels)
            if original_data:
                all_data = torch.cat(original_data, dim=0)
                all_labels = torch.cat(original_labels, dim=0)
                # Flip all labels to a random incorrect class (independent per Sybil)
                flipped_labels = all_labels.clone()
                for idx in range(len(all_labels)):
                    original_class = all_labels[idx].item()
                    choices = list(range(self.environment.num_classes))
                    choices.remove(original_class)
                    flipped_class = random.choice(choices)
                    flipped_labels[idx] = flipped_class
                adversarial_data = all_data.clone()
                if torch.rand(1).item() < 0.5:
                    noise_strength = 0.1
                    noise = torch.randn_like(all_data) * noise_strength
                    adversarial_data = torch.clamp(all_data + noise, 0, 1)
                for sybil_idx in range(self.num_sybil_per_malicious):
                    sybil_model = copy.deepcopy(global_model)
                    sybil_model.train()
                    optimizer = torch.optim.SGD(sybil_model.parameters(), lr=0.01, momentum=0.5)
                    criterion = torch.nn.CrossEntropyLoss()
                    poison_dataset = torch.utils.data.TensorDataset(adversarial_data, flipped_labels)
                    data_loader = torch.utils.data.DataLoader(poison_dataset, batch_size=32, shuffle=True)
                    # Adversarial training: maximize loss on validation/test set
                    for epoch in range(3):
                        for batch_idx, (data, target) in enumerate(data_loader):
                            if batch_idx >= 5:
                                break
                            optimizer.zero_grad()
                            output = sybil_model(data)
                            loss = criterion(output, target)
                            # Adversarial step: maximize loss on test set
                            test_loader = self.environment.test_loader
                            try:
                                test_data, test_target = next(self._test_iter)
                            except (AttributeError, StopIteration):
                                self._test_iter = iter(test_loader)
                                test_data, test_target = next(self._test_iter)
                            test_output = sybil_model(test_data)
                            test_loss = criterion(test_output, test_target)
                            total_loss = loss - test_loss
                            if torch.isnan(total_loss) or torch.isinf(total_loss):
                                continue
                            total_loss.backward()
                            # Gradient sign attack: reverse the gradients
                            for param in sybil_model.parameters():
                                if param.grad is not None:
                                    param.grad *= -1
                            torch.nn.utils.clip_grad_norm_(sybil_model.parameters(), max_norm=1.0)
                            optimizer.step()
                            if epoch == 2 and batch_idx == 4:
                                with torch.no_grad():
                                    for param in sybil_model.parameters():
                                        noise = torch.randn_like(param) * 0.01
                                        param.data += noise
                    with torch.no_grad():
                        for param in sybil_model.parameters():
                            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                                param.data.copy_(global_model.state_dict()[list(global_model.state_dict().keys())[0]])
                            param.data = torch.clamp(param.data, -10.0, 10.0)
                        # Gradient amplification: scale up Sybil model parameters
                        for param in sybil_model.parameters():
                            param.data *= self.amplification_factor
                    sybil_models.append(sybil_model)
        return sybil_models

    def execute_training_round(self):
        self.current_round += 1
        round_info = {
            'round': self.current_round,
            'attack_active': self.attack_active,
            'timestamp': datetime.now().isoformat(),
            'num_sybil_nodes': 0,
            'attack_method': 'label_flipping',
            'flip_ratio': 1
        }
        if not self.attack_active and self.current_round >= self.attack_start_round and len(self.environment.sybil_clients) > 0:
            self.attack_active = True
            print(f"Sybil attack started at round {self.current_round}")
        global_model = self.environment.get_global_model()
        honest_models = []
        for client in self.environment.honest_clients:
            local_model = client.train_local_model(global_model)
            honest_models.append(local_model)
        sybil_models = []
        if self.attack_active and len(self.environment.sybil_clients) > 0:
            sybil_models = self.simple_label_flipping_attack(global_model, flip_ratio=1.0)
            round_info['num_sybil_nodes'] = len(sybil_models)
        updated_global_model = self.federated_averaging_with_sybil(honest_models, sybil_models)
        self.environment.global_model = updated_global_model
        accuracy, loss = self.environment.evaluate_model(updated_global_model)
        round_info.update({
            'accuracy': accuracy,
            'loss': loss,
            'num_participants': len(honest_models) + len(sybil_models),
            'honest_clients': len(honest_models),
            'sybil_nodes': len(sybil_models),
            'sybil_ratio': len(sybil_models) / (len(honest_models) + len(sybil_models)) if (len(honest_models) + len(sybil_models)) > 0 else 0,
            'sybil_to_honest_ratio': len(sybil_models) / len(honest_models) if len(honest_models) > 0 else float('inf')
        })
        self.attack_history.append(round_info)
        if len(sybil_models) > 0:
            status = f"Label flipping attack (Sybil: {len(sybil_models)})"
        else:
            status = "Honest protocol (no attack)"
        print(f"Round {self.current_round} | {status} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
        return round_info

    def federated_averaging_with_sybil(self, honest_models, sybil_models):
        # Combine all models
        all_models = honest_models + sybil_models
        total_models = len(all_models)
        sample_size = min(10, total_models)
        if total_models == 0:
            return self.environment.global_model
        # Randomly sample clients for aggregation
        sampled_indices = random.sample(range(total_models), sample_size)
        sampled_models = [all_models[i] for i in sampled_indices]
        # Count how many are honest and how many are sybil
        num_honest = sum(1 for i in sampled_indices if i < len(honest_models))
        num_sybil = sample_size - num_honest
        print(f"[Aggregation] Sampled clients: {num_honest} honest, {num_sybil} sybil")
        # Uniform weight for each sampled model
        weight = 1.0 / sample_size
        global_model = copy.deepcopy(self.environment.global_model)
        for param in global_model.parameters():
            param.data.zero_()
        for model in sampled_models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                if not torch.isnan(local_param.data).any() and not torch.isinf(local_param.data).any():
                    global_param.data += weight * local_param.data
        with torch.no_grad():
            for param in global_model.parameters():
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    param.data.copy_(self.environment.global_model.state_dict()[list(self.environment.global_model.state_dict().keys())[0]])
                param.data = torch.clamp(param.data, -5.0, 5.0)
                if torch.rand(1).item() < 0.2:
                    scale_factor = torch.rand(1).item() * (1.2 - 0.8) + 0.8
                    param.data *= scale_factor
        return global_model

    def _normal_federated_averaging(self, models):
        if not models:
            return self.environment.global_model
        global_model = copy.deepcopy(models[0])
        for param in global_model.parameters():
            param.data.zero_()
        weight = 1.0 / len(models)
        for model in models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                global_param.data += weight * local_param.data
        return global_model

    def run_attack_simulation(self, total_rounds=15, attack_start_round=2, verbose=True):
        self.attack_start_round = attack_start_round
        for round_num in range(total_rounds):
            self.execute_training_round()
        return {
            'environment_info': self.environment.get_environment_info(),
            'attack_config': ATTACK_SCENARIOS['spoil_aggressive'],
            'history': self.attack_history,
            'final_accuracy': self.attack_history[-1]['accuracy'] if self.attack_history else 0,
            'timestamp': datetime.now().isoformat()
        }

def create_attack_orchestrator(environment, num_sybil_per_malicious=8, amplification_factor=5.0):
    return SybilVirtualDataAttackOrchestrator(environment, num_sybil_per_malicious, amplification_factor) 