import json

class Configurator:
    def __init__(self, json_file_path="sentences.json"):
        self.data = self.load_config(json_file_path)

    def load_config(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Unable to parse JSON in file: {file_path}")
            return None

    def get_sentence_pair(self, index):
        if str(index) in self.data:
            return self.data[str(index)]['sentences']
        else:
            print("Invalid index.")
            return None
    
    def get_yorum(self, index):
        if str(index) in self.data:
            return self.data[str(index)]['Yorum']
        else:
            print("Invalid index.")
            return None
