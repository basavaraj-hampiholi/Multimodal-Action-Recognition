import os
import argparse
import skvideo.io as io
import skimage.io as imageio


root = '/storage/hampiholi/datasets/nvgesture/'

def load_split_nvgesture(file_with_split = './nvgesture_train_correct.lst',list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'rb') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.decode().split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split

def load_data_from_file(train_list, sensor, args):
    image_tmpl='frame{:05d}.jpg'
    for l, line in enumerate(train_list):
        path = args.datadir+line[sensor][2:] + ".avi"
        start_frame = line[sensor+'_start']
        end_frame = line[sensor+'_end']

        vid = io.vread(path)[start_frame:end_frame,:,:,:]
        folder_path = path[:-4]
        print('processing.....',l, folder_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i,img in enumerate(vid):
            imageio.imsave(folder_path+'/'+image_tmpl.format(i), img)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Extract frames from videos containing an actions.')
    parser.add_argument('--datadir', type=str, help='provide absolute path to root data directory', default='/path/to/root data directory')

    args = parser.parse_args()

    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    file_lists = dict()
    file_lists["test"] = "./meta/nvgesture_test_correct.lst"
    file_lists["train"] = "./meta/nvgesture_train_correct.lst"
    train_list = list()
    test_list = list()

    load_split_nvgesture(file_with_split = file_lists["train"], list_split = train_list)
    load_split_nvgesture(file_with_split = file_lists["test"], list_split = test_list)

    for sensors in ["color","depth"]:
        load_data_from_file(train_list, sensors, args)
        load_data_from_file(test_list, sensors, args)

# Example run: python readdata.py --datadir '/storage/hampiholi/datasets/nvgesture/'