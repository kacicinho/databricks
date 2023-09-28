from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v1'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, disable_tqdm=True)
model = AutoModel.from_pretrained(MODEL_NAME)


def embed(data: np.array, batch_size: int) -> List[np.array]:
    """ returns the embeddings list from synopsys
    list has a length of len(data) and is comprise of np.arrays of shape [1,768]
    """
    if len(data) < batch_size:
        nb_batchs = 1
    else:
        nb_batchs = len(data) // batch_size

    batchs = np.array_split(data, nb_batchs)
    mean_pooled_list = []

    for batch in batchs:
        mean_pooled_list.append(transform(batch))
    mean_pooled_tensor = torch.tensor(len(data), dtype=float)
    mean_pooled_tensor = torch.cat(mean_pooled_list, out=mean_pooled_tensor)  # concatenation

    return list(mean_pooled_tensor.numpy())


def transform(data):
    """
    Create a len(data)  * 768 tensor.
    """
    data = list(data)

    token_dict = tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt")

    # removes unnecessary grad component of the tensor for faster computation
    with torch.no_grad():
        token_embed = model(**token_dict)

    # each of the 512 token has a vector of size 768
    attention_mask = token_dict['attention_mask']
    embeddings = token_embed.last_hidden_state

    # average pooling of masked embeddings
    mean_pooled = mean_pooling(embeddings, attention_mask)

    return mean_pooled


def mean_pooling(embeddings, attention_mask):
    """
    compute the "mean" of the embeddings
    attention_mask provide information on what to take into account
    (we don't want to use padding tokens, as an example)

    token_embeddings shape is [nb_of_synopsys, 512, 768], and we want a mean across dimension 1 (of size 512)

    """

    # match attention_mask shape to token_embeddings shape
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    mask_embeddings = embeddings * input_mask_expanded
    summed = torch.sum(mask_embeddings, 1)

    # min parameter avoids a division by zero
    counts = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return summed / counts
