import json

def create_id_cap_mapping():
    data = open('annotations/captions_train2014.json')
    predicted_data = open('data/image_id_caption.json')
    data = json.load(data)
    predicted_data = json.load(predicted_data)

    id_caption = {}
    for i in data['images'] :
        file_name = i['file_name']
        id = i['id']
        for image_name,caption in predicted_data.items():
            if image_name == file_name:
                id_caption[id] = caption

    out_file = open('data/id_caption.json', "w") 
    json.dump(id_caption,out_file)

    return