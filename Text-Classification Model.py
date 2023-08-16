Mat_category = {
    10000: 'API',
    12000: 'Excipients',
    18000: 'Building Supplies -I',
    19000: 'Electrical Items-I',
    20000: 'Primary Packing',
    21000: 'Machinery Supplies-I',
    22000: 'Engineering-I',
    23000: 'Secondary Packing',
    30000: 'Semi Finished - 1',
    31000: 'Semi Finished - 2',
    32000: 'Laboratory Supplies',
    33000: 'R&D Materials',
    34000: 'Brand samples',
    35000: 'Office Supplies',
    36000: 'Building Supplies',
    37000: 'Machinery Supplies',
    38000: 'Gas & Fuel',
    39000: 'Engineering',
    40000: 'Filters',
    41000: 'Electrical Items',
    42000: 'Computer H/W Exps',
    43000: 'Factory Maintenance',
    44000: 'Vehicle Maintenance',
    45000: 'Tubings',
    46000: 'Consumables',
    47000: 'Welfare Expenses',
    48000: 'Safety & Protective',
    49000: 'Assets',
    50000: 'Connectors',
    80000: 'Injectables',
    81000: 'Ophthalmic',
    82000: 'Oral solutions',
    90000: 'Services',
    90001: 'Freight'
}




text=input('Enter your text:')


import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertModel


# Loading pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)
model.eval()

# Function to generate text embedding
def generate_text_embedding(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        text_embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return text_embedding

# Loading category embeddings from pickle file
input_file = r"C:\Users\Amber.mishra\PycharmProjects\Text-Classification-Project1\category_embeddings07.pkl"
with open(input_file, 'rb') as file:
    category_embeddings = pickle.load(file)


# Generating text embedding for the input text
text_embedding = generate_text_embedding(text)

# Calculating similarity and assigning predicted category
predicted_category = None
max_similarity = -1

for category, embeddings in category_embeddings.items():
    similarities = torch.cosine_similarity(torch.stack(embeddings), text_embedding.unsqueeze(0))
    category_similarity = torch.max(similarities).item()

    if category_similarity > max_similarity:
        max_similarity = category_similarity
        predicted_category = category

print(Mat_category[predicted_category])
