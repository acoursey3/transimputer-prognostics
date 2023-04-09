import torch
import numpy as np

import numpy as np
import torch

def mask_random(batched_data):
    masked = batched_data.clone()
    percent_missing = np.random.randint(30, 60) / 100

    # Find indices of the padded values
    padded_indices = (masked == 0).all(dim=2)

    # Generate random masks for true values only
    true_masks = torch.rand(batched_data.shape) > percent_missing
    true_masks[padded_indices] = False

    # Mask the true values and leave padded values as 0
    masked[true_masks] = -1

    return masked

def mask_intermittently(batched_data):
    masked = batched_data.clone()
    n_cols = np.random.randint(0, batched_data.shape[2])
    cols = np.random.choice(list(range(n_cols)), n_cols, replace=False)
    checkerboard_mask = np.ones(masked.shape[1])
    checkerboard_mask[0::2] = -1

    masked[:, :, cols] = (masked[:, :, cols] * np.expand_dims(np.expand_dims(checkerboard_mask, 0), 2))
    masked[masked < 0] = -1
    return masked

# Written by ChatGPT
def mask_blocks(batched_data):
    x = batched_data.clone()
    batch_size, seq_len, num_features = x.shape

    # Create a mask tensor to indicate which elements of the tensor should be masked
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    # Iterate over each element in the batch
    for i in range(batch_size):
        # Get the sequence for the current example
        seq = x[i]

        # Determine the length of the non-padded sequence
        non_padded_len = torch.nonzero(seq[:,2]).shape[0]

        # Randomly select the number of blocks to mask (1-5)
        num_blocks = torch.randint(1, 3, size=(1,)).item()

        # Iterate over each block to mask
        for j in range(num_blocks):
            # Randomly select the length of the block (5-20)
            block_len = torch.randint(0, int(non_padded_len/num_blocks)+1, size=(1,)).item()

            if non_padded_len - block_len + 1 < 1:
                continue
            # Randomly select the start index of the block
            start_index = torch.randint(0, non_padded_len - block_len + 1, size=(1,)).item()
                
            # Mask the selected block
            mask[i, start_index:start_index + block_len] = True

    # Apply the mask to the tensor and replace masked values with -1
    x[mask.unsqueeze(-1).expand_as(x)] = -1

    return x

def mask_input(batched_data, only_mask=True):
    min_size = 5
    
    masked_data = []
    batch_size = batched_data.shape[0]
    for i in range(batch_size):
        
        sample = batched_data[i]
        eps = np.random.rand()

        if torch.nonzero(sample[:,2]).shape[0] < min_size:
            masked_data.append(sample)
            continue
            
        if eps > 0.66:
            masked = mask_random(sample.unsqueeze(0))
        elif eps > 0.33:
            masked = mask_intermittently(sample.unsqueeze(0))
        else:
            masked = mask_blocks(sample.unsqueeze(0))
            
        masked_data.append(masked.squeeze(0))
            
    
    return torch.stack(masked_data)