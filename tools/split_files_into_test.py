from shutil import copyfile
import anafora
import os
split_files_path = 'path/to/train-all-data'
split_train_path = 'path/to/train-new-data'
split_test_input_path = 'path/to/test-input'
split_test_label_path = 'path/to/test-label'
text_directory_files = anafora.walk(split_files_path, xml_name_regex=".*((?<![.].{3})|[.]xml)$")
i = 0
for text_files in text_directory_files:
    text_subdir_path, text_doc_name, text_file_names = text_files
    for text_file_name in text_file_names:
        old_xml_file_path = os.path.join(split_files_path, text_subdir_path, text_file_name)
        if i % 5 == 4:
            if text_file_name.endswith("xml"):
                new_xml_dir_path = os.path.join(split_test_label_path, text_subdir_path)
            else:
                new_xml_dir_path = os.path.join(split_test_input_path, text_subdir_path)
        else:
            new_xml_dir_path = os.path.join(split_train_path, text_subdir_path)
        if not os.path.exists(new_xml_dir_path):
            os.makedirs(new_xml_dir_path, 0o0755)
        new_xml_file_path = os.path.join(new_xml_dir_path, text_file_name)
        copyfile(old_xml_file_path, new_xml_file_path)
    i += 1


