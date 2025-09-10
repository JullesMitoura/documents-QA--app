import requests

api_url = "https://documents-qa-app-h3dremaacre3fqat.eastus2-01.azurewebsites.net"

# 1. Criar índice
create_payload = {
    "index_name": "meu_indice",
    "vector_dimension": 1536
}
resp = requests.post(f"{api_url}/create-index", json=create_payload)
print("Create index:", resp.json())

# 2. Fazer upload de documento
files = {"file": open("meu_doc.pdf", "rb")}
data = {
    "index_name": "meu_indice",
    "processing_mode": "quality",  # ou "normal"
    "additional_information": "Teste",
    "library_name": "default"
}
resp = requests.post(f"{api_url}/upload-document", files=files, data=data)
print("Upload:", resp.json())

# 3. Perguntar algo sobre o documento
ask_payload = {
    "question": "Qual é o título do documento?",
    "index_name": "meu_indice",
    "top_k": 3
}
resp = requests.post(f"{api_url}/ask", json=ask_payload)
print("Resposta:", resp.json())

# # 4. Deletar índice
# delete_payload = {"index_name": "meu_indice"}
# resp = requests.delete(f"{api_url}/delete-index", json=delete_payload)
# print("Delete index:", resp.json())
