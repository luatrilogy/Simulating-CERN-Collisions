import json

def append_to_json_array(filename=str, new_data=dict):
    try:
        with open(filename, "r+") as file:
            file_data = json.load(file)

            if isinstance(file_data, list):
                file_data.append(new_data)
            else:
                print(f"Error: The root element of {filename} is not a list")
                return
            
            file.seek(0)
            json.dump(file_data, file, indent=4)
            file.truncate()

    except FileNotFoundError:
        with open(filename, "w") as file:
            json.dump([new_data], file, indent=4)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format")

def clear_json_file_array(filename):
    with open(filename, "w") as file:
        json.dump([], file)