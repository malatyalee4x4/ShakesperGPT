import os
import torch
import urllib.request

SHAKESPERE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"

DATASET_PATH = "data/shakespeare.txt"

def download_shakespeare():
    if os.path.exists(DATASET_PATH):
        print("Shakespeare dataset already exists.")
        return

    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(SHAKESPERE_URL, DATASET_PATH)
    print("Shakespeare dataset downloaded.")
    


class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        
        self.char_to_id ={}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index
        
        self.id_to_char ={}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char 
            
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {self.characters}")
        
    def encode(self, text:str) -> list:
        ids=[]
        for char in text:
            ids.append(self.char_to_id[char])
        return ids
    
    def decode(self, ids:list) -> str:
        
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        text=""
        for id in ids:
            text+=self.id_to_char[id]
        return text
    
    
    
def load_data(train_split: float = 0.9):
    download_shakespeare()
    
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        text = file.read()
    
    tokenizer = CharacterTokenizer(text)
    
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)
    
    split_index = int(len(data) * train_split)
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data,tokenizer

def get_batch(data:torch.Tensor,block_size:int,batch_size:int):
    
    max_start=len(data) - block_size - 1
    positions=torch.randint(max_start,(batch_size,))
    #print(f"positions:",+positions)
    
    x_list=[]
    y_list=[]
    
    for pos in positions:
        x_list.append(data[pos: pos+block_size])
        y_list.append(data[pos+1:pos+block_size+1])
        
    x=torch.stack(x_list)
    y=torch.stack(y_list)

    return x,y


if __name__ == "__main__":
    
    train_data, test_data, tokenizer = load_data()
    x,y=get_batch(data=train_data,block_size=256,batch_size=5)
    print(x.shape)
    print(y.shape)