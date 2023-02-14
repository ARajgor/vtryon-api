## import
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import shutil
import subprocess
import time
import cv2
import boto3
## from custom py files
from tryon_utils.openpose_json import generate_pose_keypoints
from tryon_utils.cloth_mask import cloth_masking
from tryon_utils.image_mask import make_body_mask

application = Flask(__name__)
application.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}

bucket_name = 'wiw-img'
filename_cloth = '004325_1.jpg'


# check extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in application.config['ALLOWED_EXTENSIONS']


def upload_file(file_name):
    object_name = file_name
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(f'result/TOM/test/try-on/{file_name}', bucket_name,
                                     f'result_img/{object_name}')
    print("result file uploaded")
try:
    print()
except:
    print()
def save_file(file_name):
    object_name = file_name
    s3_client = boto3.resource('s3')
    try:
        r = s3_client.Bucket(bucket_name).download_file('upload_img/{}'.format(file_name),
                                                        'data/test/image/{}'.format(object_name))
        print("file downloaded")
        msg = True
    except:
        msg = "file is not downloaded"
        print(msg)

    return msg


def images_generation(filename_person):
    # ..... Resize/Crop Images 192 x 256 (width x height) ..... #
    img_p = cv2.imread("data/test/image/" + filename_person)
    # print("img_p: ", img_p)
    person_resize = cv2.resize(img_p, (192, 256))
    # save resized person image
    cv2.imwrite("data/test/image/" + filename_person, person_resize)

    img_c = cv2.imread("data/test/cloth/" + filename_cloth)
    cloth_resize = cv2.resize(img_c, (192, 256))
    # save resized cloth image
    cv2.imwrite("data/test/cloth/" + filename_cloth, cloth_resize)

    # ..... Cloth Masking ..... #
    image_path = "data/test/cloth/" + filename_cloth
    res_path = "data/test/cloth-mask/" + filename_cloth
    clothmask = cloth_masking(image_path, res_path)

    # ..... Image parser ..... #
    cmd_parse = "python tryon_utils/inference.py --loadmodel tryon_utils/checkpoints/inference.pth --img_path " + filename_person + " --output_path img/ --output_name " + filename_person
    subprocess.call(cmd_parse, shell=True)

    # ..... Person Image Masking ..... #
    # img_file = "000010_0.jpg", seg_file = "000010_0.png"
    seg_file = filename_person.replace(".jpg", ".png")
    img_mask = make_body_mask(filename_person, seg_file)

    # ..... Generate Pose Keypoints .....#
    pose_keypoints = generate_pose_keypoints(filename_person)

    # ..... Write test sample pair txt file ..... #
    with open("data/test_samples_pair.txt", "w") as text_file:
        text_file.write(str(filename_person) + " " + str(filename_cloth))


@application.route('/api', methods=['POST'])
def api():
    filename_person = request.json['file']
    filename_person.lower()
    preds = request.json['preds']
    print("filename: ", filename_person)
    print("preds value:", preds)

    save_file(filename_person)

    print("data/test/image/" + filename_person)

    # resize and generate all images and poses keypoints
    images_generation(filename_person)

    # ..... Run Geometric Matching Module(GMM) Model ..... #
    cmd_gmm = "python tryon_utils/test.py --name GMM --stage GMM --workers 4 --datamode test --data_list test_samples_pair.txt --checkpoint tryon_utils/checkpoints/GMM/gmm_final.pth"
    subprocess.call(cmd_gmm, shell=True)
    # time.sleep(10)
    # move generated files to data/test/
    # result/GMM/test/warp-cloth/004325_1.jpg
    warp_cloth = "result/GMM/test/warp-cloth/" + filename_person
    warp_mask = "result/GMM/test/warp-mask/" + filename_person
    shutil.copyfile(warp_cloth, "data/test/warp-cloth/" + filename_person)
    shutil.copyfile(warp_mask, "data/test/warp-mask/" + filename_person)

    # ..... Run Try-on Module(TOM) Model ..... #
    cmd_tom = "python tryon_utils/test.py --name TOM --stage TOM --workers 4 --datamode test --data_list test_samples_pair.txt --checkpoint tryon_utils/checkpoints/TOM/tom_final.pth"
    subprocess.call(cmd_tom, shell=True)
    upload_file(filename_person)
    return jsonify({"action": "success"})


if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5001)

