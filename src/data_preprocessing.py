# Import Statments:
import torch
import pandas as pd

# Adding Device Management:
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available and set as device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available and set as device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# Parameters
max_len = 27

# Creating Vocabulary for Amino Acids + Special Tokens:
amino_acids = ['<CLS>',
 'A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'Y',
 '<BED>',
 '<PAD>']

vocab_size = len(amino_acids)

aatoi = {aa:i for i,aa in enumerate(amino_acids)}
itoaa = {i:aa for i,aa in enumerate(amino_acids)}

encode = lambda p: [aatoi[aa] for aa in p]
decode = lambda l: [itoaa[i] for i in l]

def load_general_peptides():
    # Curating General Peptide Data from PeptideAtlas:
    with open('Generalized_Peptide_Data.txt', 'r') as f:
        general_peptides = f.read().splitlines()

    print(f'{len(general_peptides)} general peptides loaded | Sample peptide: {general_peptides[5]}')

    general_peptide_data = []

    # Converting Peptides to Fixed-Length with Special Tokens:
    for peptide in general_peptides:

        peptide = ['<CLS>'] + list(peptide) + ['<BED>']

        for _ in range(max_len - len(peptide)):
            peptide.append('<PAD>')

        general_peptide_data.append(peptide)

    print(f'Sample peptide: {''.join(general_peptide_data[5])}')
    print(f'Peptide Length: {len(general_peptide_data[5])} tokens')

    # Creating General Peptide Tensor:
    tokenized_peptides = []

    for peptide in general_peptide_data:
        tokenized_peptides.append(encode(peptide))

    general_peptide_tensor = torch.tensor(tokenized_peptides, dtype=torch.long).to(device)

    print(f'General peptide tensor shape: {list(general_peptide_tensor.shape)}')

    # Creating Training/Validation Split:
    train_val_split = int(0.9 * len(general_peptide_tensor))

    general_peptide_train = general_peptide_tensor[:train_val_split]
    general_peptide_val = general_peptide_tensor[train_val_split:]

    print(f'Number of training peptides: {len(general_peptide_train)}')
    print(f'Number of validation peptides: {len(general_peptide_val)}')

    return general_peptide_train, general_peptide_val

def load_amp_data():
    # Curating AMP Data from dbAMP:
    with open('AMP_Data.txt', 'r') as f:
        amps = f.read().splitlines()

    # Curating non-AMP Data from UniProt:
    with open('non-AMP_Data.txt', 'r') as f:
        non_amps = f.read().splitlines()

    print(f'{len(amps)} AMPs loaded | Sample AMP: {amps[5]}')
    print(f'{len(non_amps)} non-AMPs loaded | Sample non-AMP: {non_amps[5]}')

    amp_data = []
    non_amp_data = []

    # Converting AMPs to Fixed-Length with Special Tokens:
    for amp in amps:

        amp = ['<CLS>'] + list(amp) + ['<BED>']

        for _ in range(max_len - len(amp)):
            amp.append('<PAD>')

        amp_data.append(amp)

    # Converting non-AMPs to Fixed-Length with Special Tokens:
    for non_amp in non_amps:

        non_amp = ['<CLS>'] + list(non_amp) + ['<BED>']

        for _ in range(max_len - len(non_amp)):
            non_amp.append('<PAD>')

        non_amp_data.append(non_amp)

    print(f'Sample AMP: {''.join(amp_data[5])}')
    print(f'Sample non_AMP: {''.join(non_amp_data[5])}')
    print(f'AMP Length: {len(amp_data[5])} tokens')
    print(f'non_AMP Length: {len(non_amp_data[5])} tokens')

    # Creating AMP Tensor:
    tokenized_amps = []

    for amp in amp_data:
        tokenized_amps.append(encode(amp))

    amp_tensor = torch.tensor(tokenized_amps, dtype=torch.long).to(device)

    # Creating non-AMP Tensor:
    tokenized_non_amps = []

    for non_amp in non_amp_data:
        tokenized_non_amps.append(encode(non_amp))

    non_amp_tensor = torch.tensor(tokenized_non_amps, dtype=torch.long).to(device)

    amp_tensor = amp_tensor[:16384]
    non_amp_tensor = non_amp_tensor[:16384]

    print(f'AMP tensor shape: {list(amp_tensor.shape)}')
    print(f'non-AMP tensor shape: {list(non_amp_tensor.shape)}')

    # Creating Training/Validation Split:
    train_val_split = int(0.9 * len(amp_tensor))

    amp_train = amp_tensor[:train_val_split]
    amp_val = amp_tensor[train_val_split:]

    non_amp_train = non_amp_tensor[:train_val_split]
    non_amp_val = non_amp_tensor[train_val_split:]

    print(f'Number of training AMPs: {len(amp_train)}')
    print(f'Number of training non-AMPs: {len(non_amp_train)}')
    print(f'Number of validation AMPs: {len(amp_val)}')
    print(f'Number of validation non-AMPs: {len(non_amp_val)}')

    return amp_train, amp_val, non_amp_train, non_amp_val

def load_mic_data():
    # Curating MIC Value Data:
    df = pd.read_csv('MIC_Value_Data.csv')

    mic_sequence_data = []
    mic_value_data = df['value'].values

    # Converting MIC Data Sequences to Fixed-Length with Special Tokens:
    for sequence in df['sequence']:

        sequence = ['<CLS>'] + list(sequence) + ['<BED>']

        for _ in range(max_len - len(sequence)):
            sequence.append('<PAD>')

        mic_sequence_data.append(sequence)

    print(f'Sample sequence: {''.join(mic_sequence_data[5])}')
    print(f'sequence Length: {len(mic_sequence_data[5])} tokens')
    print(f'MIC value: {mic_value_data[5]} ug/ml')

    # Creating MIC Sequence Tensor:
    tokenized_sequences = []

    for sequence in mic_sequence_data:
        tokenized_sequences.append(encode(sequence))

    mic_sequence_tensor = torch.tensor(tokenized_sequences, dtype=torch.long).to(device)
    mic_value_tensor = torch.tensor(mic_value_data, dtype=torch.float32).to(device)

    # Normalizing MIC Value Tensor:
    mic_mean = mic_value_tensor.mean()
    mic_std = mic_value_tensor.std()
    mic_value_tensor = (mic_value_tensor - mic_mean) / mic_std

    print(f'MIC sequence tensor shape: {list(mic_sequence_tensor.shape)}')
    print(f'MIC value tensor shape: {list(mic_value_tensor.shape)}')

    # Creating Training/Validation Split:
    train_val_split = int(0.9 * len(mic_sequence_data))

    mic_sequence_train = mic_sequence_tensor[:train_val_split]
    mic_sequence_val = mic_sequence_tensor[train_val_split:]

    mic_value_train = mic_value_tensor[:train_val_split]
    mic_value_val = mic_value_tensor[train_val_split:]

    print(f'Number of training sequences: {len(mic_sequence_train)}')
    print(f'Number of training values: {len(mic_value_train)}')
    print(f'Number of validation sequences: {len(mic_sequence_val)}')
    print(f'Number of validation values: {len(mic_value_val)}')

    return mic_sequence_train, mic_sequence_val, mic_value_train, mic_value_val, mic_mean, mic_std

# Creating Batches of General Peptide Data:
def get_batch(split, general_peptide_train, general_peptide_val, batch_size=1):
    data = general_peptide_train if split == 'train' else general_peptide_val

    ix = torch.randint(len(data), (batch_size,))

    x = torch.stack([data[i] for i in ix])
    padding_tensor = torch.tensor([22], dtype=torch.long).to(device)
    y = torch.stack([torch.cat([data[i][1:], padding_tensor]) for i in ix])

    return x, y

# Creating Batches of AMP and non-AMP Data:
def get_contrastive_batch(split, amp_train, amp_val, non_amp_train, non_amp_val, batch_size=32):
    amp_data = amp_train if split == 'train' else amp_val
    non_amp_data = non_amp_train if split == 'train' else non_amp_val
    
    ix = torch.randint(len(amp_data), (batch_size//2,))
    
    amp_sequences = torch.stack([amp_data[i] for i in ix])
    non_amp_sequences = torch.stack([non_amp_data[i] for i in ix])
    
    # Combining AMP and non-AMP Sequences:
    all_sequences = torch.cat([amp_sequences, non_amp_sequences], dim=0)
    
    # Creating Targets for Autoregressive Training:
    targets = torch.cat([
        all_sequences[:, 1:], 
        torch.full((batch_size, 1), 22).to(device)  
    ], dim=1)
    
    return all_sequences, targets, amp_sequences, non_amp_sequences

# Creating Batches of MIC Data:
def get_mic_batch(split, mic_sequence_train, mic_sequence_val, mic_value_train, mic_value_val, batch_size=1):
    sequence_data = mic_sequence_train if split == 'train' else mic_sequence_val
    value_data = mic_value_train if split == 'train' else mic_value_val
    
    ix = torch.randint(len(sequence_data), (batch_size,))
    
    x = torch.stack([sequence_data[i] for i in ix])
    y = torch.stack([value_data[i] for i in ix])
    
    return x, y