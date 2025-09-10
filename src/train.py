import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_preprocessing import (
    load_general_peptides, load_amp_data, load_mic_data,
    get_batch, get_contrastive_batch, get_mic_batch,
    device, decode, max_len
)
from model import AMPTransformer

# Load all data
print("Loading general peptides...")
general_peptide_train, general_peptide_val = load_general_peptides()

print("Loading AMP data...")
amp_train, amp_val, non_amp_train, non_amp_val = load_amp_data()

print("Loading MIC data...")
mic_sequence_train, mic_sequence_val, mic_value_train, mic_value_val, mic_mean, mic_std = load_mic_data()

# Initializing Model:
model = AMPTransformer().to(device)

# Creating Optimizer:
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')

# General peptide training
general_train_losses = []
general_val_losses = []

# Creating Training Loop:
pbar = tqdm(range(5000), desc="training")

for step in pbar:

    x, y = get_batch('train', general_peptide_train, general_peptide_val)
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    general_train_losses.append(loss)

    pbar.set_postfix({'loss': loss.item()})

    with torch.no_grad():
        x, y = get_batch('val', general_peptide_train, general_peptide_val)
        _, loss = model(x, y)
        
    general_val_losses.append(loss)

print(f'Loss value after training: {loss.item()}')

# Creating Contrastive Loss Function:
contrastive_train_losses = []
contrastive_val_losses = []

lambda_weight = 0.1

def contrastive_loss(model, amp_batch, non_amp_batch, margin=1.0):
    
    # Creating AMP and non-AMP Representations:
    amp_repr = model.get_representation(amp_batch)
    non_amp_repr = model.get_representation(non_amp_batch)
    
    # Normalizing Probability Distributions:
    amp_repr = F.softmax(amp_repr, dim=-1)
    non_amp_repr = F.softmax(non_amp_repr, dim=-1)
    
    # Creating KL divergence:
    kl_div = F.kl_div(non_amp_repr.log(), amp_repr, reduction='batchmean')
    
    # Calculating Loss:
    loss = torch.max(torch.tensor(0.0).to(device), margin - kl_div) ** 2
    
    return loss

# Creating Contrastive Training Loop:
pbar = tqdm(range(5000), desc="training")

for step in pbar:

    # Contrastive Loss + Auto-Regressive Loss:
    sequences, targets, amps, non_amps = get_contrastive_batch('train', amp_train, amp_val, non_amp_train, non_amp_val)
    logits, arlm_loss = model(sequences, targets)
    contr_loss = contrastive_loss(model, amps, non_amps)
    total_loss = arlm_loss + lambda_weight * contr_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()

    contrastive_train_losses.append(total_loss)

    pbar.set_postfix({'loss': total_loss.item()})

    with torch.no_grad():
        sequences, targets, amps, non_amps = get_contrastive_batch('val', amp_train, amp_val, non_amp_train, non_amp_val)
        logits, arlm_loss = model(sequences, targets)
        contr_loss = contrastive_loss(model, amps, non_amps)
        total_loss = arlm_loss + lambda_weight * contr_loss

    contrastive_val_losses.append(total_loss)
    
print(f'Loss value after training: {total_loss.item()}')

# Creating MIC Prediction Training Loop:
mic_train_losses = []
mic_val_losses = []

batch_size = 32

pbar = tqdm(range(5000), desc="training")

for step in pbar:

    # Calculating MIC loss:
    x, y = get_mic_batch('train', mic_sequence_train, mic_sequence_val, mic_value_train, mic_value_val)
    MIC_predictions = model.predict_mic(x)
    mic_loss = F.mse_loss(MIC_predictions, y)
    
    optimizer.zero_grad(set_to_none=True)
    mic_loss.backward()
    optimizer.step()

    mic_train_losses.append(mic_loss)

    pbar.set_postfix({'loss': mic_loss.item()})

    with torch.no_grad():
        x, y = get_mic_batch('val', mic_sequence_train, mic_sequence_val, mic_value_train, mic_value_val)
        MIC_predictions = model.predict_mic(x)
        mic_loss = F.mse_loss(MIC_predictions, y)

    mic_val_losses.append(mic_loss)

print(f'Loss value after training: {mic_loss.item()}')

# Creating Novel Data:
n_peptides = 10
index = 0

while index < n_peptides:

    new_peptide = model.generate(torch.zeros([1, 1], dtype=torch.long).to(device))[0].tolist()

    display_sequence = new_peptide

    for _ in range(max_len - len(new_peptide)):
        new_peptide = new_peptide + [22]

    new_peptide = torch.tensor([new_peptide], dtype=torch.long).to(device)

    MIC_prediction = 10 ** (model.predict_mic(new_peptide) * mic_std + mic_mean)

    if (len(display_sequence) > max_len or len(display_sequence) <= 6 or MIC_prediction >= 20):
        continue

    index += 1

    print(f'{index}.\nSequence: {''.join(decode(display_sequence[1:-1]))}')

    print(f'MIC value: {MIC_prediction} ug/ml\n')