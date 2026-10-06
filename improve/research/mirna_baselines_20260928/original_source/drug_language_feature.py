import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from rdkit import Chem


# Step 1: Load ChemBERTa model and tokenizer (from local path)
def load_model_and_tokenizer(local_path="./ChemBERTa-zinc-base-v1"):
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    return model, tokenizer

# Step 2: Verify SMILES format
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


# Step 3: Encode SMILES
def encode_smiles(smiles_list, tokenizer):
    inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs


# Step 4: Extract feature vectors
def extract_features(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)
    return features


# Step 5: Extract drug features from Excel file
def process_excel(file_path, model, tokenizer, output_path="./2.construct_network/drug_features-chemBertA.csv"):
    df = pd.read_csv(file_path)
    df["is_valid"] = df["smiles"].apply(validate_smiles)
    valid_df = df[df["is_valid"] == True]
    invalid_df = df[df["is_valid"] == False]

    if len(invalid_df) > 0:
        print(f" SMILES invalid，next：\n{invalid_df[['drug_cid', 'smiles']]}")

    smiles_list = valid_df["smiles"].tolist()
    inputs = encode_smiles(smiles_list, tokenizer)
    features = extract_features(model, inputs)


    feature_df = pd.DataFrame(features.numpy())
    feature_df.insert(0, "drug_cid", valid_df["drug_cid"].tolist())
    feature_df.to_csv(output_path, index=False, encoding="utf-8-sig")




if __name__ == "__main__":
    input_file = "./2.construct_network/drug_smiles.csv"
    output_file = "./2.construct_network/drug_features_test.csv"
    local_model_path = "./ChemBERTa-zinc-base-v1"
    model, tokenizer = load_model_and_tokenizer(local_path=local_model_path)
    process_excel(input_file, model, tokenizer, output_file)