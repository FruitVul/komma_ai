from tensorflow.python.keras.models import load_model
import os

package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
dictionary_path = os.path.join(package_path, "dictionary.json")
model_path = os.path.join(package_path, "baseline.h5")

def main():
    model = load_model(model_path)

    model_json = model.to_json()
    with open("../model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")



if __name__ =="__main__":
    main()