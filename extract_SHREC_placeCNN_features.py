import os
import os.path as osp
from multiprocessing import Pool
import sys
current_dir = os.getcwd()
places365_dir = osp.join(current_dir, 'places365')
if not current_dir in sys.path:
    sys.path.append(current_dir)
from typing import List, Tuple
from functools import partial
from places365.run_placesCNN_unified import ExtractPlaceCNNFeatureParams, extract_placeCNN_feature, load_model
from tqdm import tqdm


def create_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)

# Initialize some path paramenters
current_dir = os.getcwd()
data_dir = '/mnt/DATA/nvtu/SceneIBR2018_Dataset'
create_folder(data_dir)
# Processed data folder path
processed_data_path = osp.join(current_dir, 'PROCESSED_DATA')
create_folder(processed_data_path)
# Attribute feature and prediction folder
attribute_folder_path = osp.join(processed_data_path, 'Attributes')
attribute_feat_path = osp.join(attribute_folder_path, 'feat')
attribute_pred_path = osp.join(attribute_folder_path, 'pred')
create_folder(attribute_feat_path)
create_folder(attribute_pred_path)
# Category feature and prediction folder
category_folder_path = osp.join(processed_data_path, 'Categories')
category_feat_path = osp.join(category_folder_path, 'feat')
category_pred_path = osp.join(category_folder_path, 'pred')
create_folder(category_feat_path)
create_folder(category_pred_path)
# Raw feature folder
raw_feat_path = osp.join(processed_data_path, 'Raw')
create_folder(raw_feat_path)
# CAMs folder
CAMs_folder_path = osp.join(processed_data_path, 'CAMs')
create_folder(CAMs_folder_path)


def parse_training_metadata() -> Tuple[List[str], List[List[str]]]:
    # Load training metadata
    training_metadata_file_path = osp.join(data_dir, 'SceneIBR2018_Image_Training.cla')
    training_metadata = [line.rstrip().split() for line in open(training_metadata_file_path, 'r').readlines()]

    # Parsing metadata to create training list
    num_classes, total_training_images = list(map(int, training_metadata[1]))
    print(f"Total number of classes: {num_classes}")
    print(f"Total number of training images: {total_training_images}")

    # Start parsing
    classes = []
    images_in_class = [] 
    num_lines = len(training_metadata)
    for line_index in range(3, num_lines):
        num_items = len(training_metadata[line_index])
        if num_items == 3: # Starting information of one class
            cls, _, num_images_per_cls = training_metadata[line_index]
            _iter = line_index + 1
            end_iter = _iter + int(num_images_per_cls)
            image_list = []
            extension = '.JPEG'
            while _iter < end_iter:
                image_name = training_metadata[_iter][0] + extension
                image_list.append(image_name)
                _iter += 1
            line_index = _iter + 1
            # classes[i] has its training items in images_in_class[i]
            images_in_class.append(image_list)
            classes.append(cls)
    return classes, images_in_class


def create_output_folder_for_classes(classes: List[str]) -> List[str]:
    # Create output for each classes
    classes_output_path = []
    for cls in classes:
        cls_attr_feat_path = osp.join(attribute_feat_path, cls)
        cls_attr_pred_path = osp.join(attribute_pred_path, cls)
        cls_cate_feat_path = osp.join(category_feat_path, cls)
        cls_cate_pred_path = osp.join(category_pred_path, cls)
        cls_raw_feat_path = osp.join(raw_feat_path, cls)
        cls_CAMs_path = osp.join(CAMs_folder_path, cls)
        paths = [cls_attr_feat_path, cls_attr_pred_path, cls_cate_feat_path, cls_cate_pred_path, cls_raw_feat_path, cls_CAMs_path]
        classes_output_path.append(paths)
        for path in paths:
            create_folder(path)
    return classes_output_path

    
def generate_input_params(classes_output_path: List[str], images_in_class: List[List[str]]) -> List[ExtractPlaceCNNFeatureParams]:
    training_data_path = osp.join(data_dir, 'Images', 'Scene10,000')
    image_extension = '.JPEG'
    feat_extension = '.npy'
    pred_extension = '.txt'
    input_params = []
    for index, image_list in enumerate(images_in_class):
        image_path_list = [osp.join(training_data_path, image_name) for image_name in image_list]
        for image_path in image_path_list:
            cls_attr_feat_path, cls_attr_pred_path, cls_cate_feat_path, cls_cate_pred_path, cls_raw_feat_path, cls_CAMs_path = classes_output_path[index]
            attr_feat_path = image_path.replace(training_data_path, cls_attr_feat_path).replace(image_extension, feat_extension)
            attr_pred_path = image_path.replace(training_data_path, cls_attr_pred_path).replace(image_extension, pred_extension)
            cate_feat_path = image_path.replace(training_data_path, cls_cate_feat_path).replace(image_extension, feat_extension)
            cate_pred_path = image_path.replace(training_data_path, cls_cate_pred_path).replace(image_extension, pred_extension)
            raw_feat_path = image_path.replace(training_data_path, cls_raw_feat_path).replace(image_extension, feat_extension)
            CAMs_path = image_path.replace(training_data_path, cls_CAMs_path)
            params = ExtractPlaceCNNFeatureParams(
                image_file_path = image_path,
                raw_feat_output_path = raw_feat_path,
                output_attribute_feat = True, 
                attribute_feat_output_path = attr_feat_path,
                attribute_pred_output_path = attr_pred_path,
                output_category_feat = True,
                category_feat_output_path = cate_feat_path,
                category_pred_output_path = cate_pred_path,
                output_CAMs = True,
                CAMs_output_path = CAMs_path
            )
            input_params.append(params)
    return input_params


if __name__ == '__main__':
    classes, images_in_class = parse_training_metadata()
    classes_output_path = create_output_folder_for_classes(classes)
    input_params = generate_input_params(classes_output_path, images_in_class)
    total_training_images = len(input_params)

    # Start extracting features
    model = load_model()
    func = partial(extract_placeCNN_feature, model=model)
    pool = Pool()
    for _ in tqdm(pool.imap_unordered(func, input_params), total=total_training_images):
        pass