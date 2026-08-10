import os
import shutil

backend_dir = r"C:\Users\ARAVIND\Desktop\local-rag\backend"
for root, dirs, files in os.walk(backend_dir):
    for d in dirs:
        if d == "__pycache__":
            pyc_dir = os.path.join(root, d)
            try:
                shutil.rmtree(pyc_dir)
                print("Removed:", pyc_dir)
            except Exception as e:
                print("Error removing", pyc_dir, e)
