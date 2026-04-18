from load_documents import preprocess_cs_documents, preprocess_en_documents

docs = preprocess_cs_documents(lst_path="documents_cs.lst", num_workers=4)
print(f"Loaded {len(docs)} Czech documents.")
print(docs["000004485"])