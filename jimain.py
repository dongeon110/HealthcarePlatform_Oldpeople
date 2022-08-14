from calendar import day_abbr
from flask import Flask,render_template,request,redirect,session,url_for,Blueprint,send_file,escape,jsonify
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS, cross_origin
import argparse
from io import BytesIO, StringIO
from PIL import Image
import json
from hyerim.hyrimmain import result
from instagram import getfollowedby, getname
from datetime import timedelta
import torch
from machine129 import translabel, meals, mypagefunction
from hyerim import model_data
from operator import itemgetter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
import os
import sys
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from pathlib import Path
from models.common import DetectMultiBackend
import io



HOME_URL = "/home"
SURVEY_URL = "/survey"
DETECTION_URL = "/predict"
SESSION_LIFETIME = 2000


# Flask, User 정보는 SQLite
app = Flask(__name__, static_url_path='/static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=SESSION_LIFETIME) # 로그인 지속시간을 정합니다. 현재 1분
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True # 사용자에게 정보 전달완료하면 teadown. 그 때마다 커밋=DB반영


db = SQLAlchemy(app)
cors = CORS(app)

class User(db.Model):
    """ Create user table"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    date = db.Column(db.String(80))
    sex = db.Column(db.String(80))
    name = db.Column(db.String(80))
    password1 = db.Column(db.String(80))
    password2 = db.Column(db.String(80))
    
    def __init__(self, name, date, sex, username, password1,password2):
        self.username = username
        self.name = name
        self.password1 = password1
        self.password2 = password2
        self.date = date
        self.sex = sex




@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept")
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  response.headers.add('Access-Control-Allow-Credentials', 'true')
  return response

##HOME, LOGIN, LOGOUT START POINT ##
@app.route('/',methods=['GET','POST'])
def home():
    if session.get('logged_in') :
        home_login = True
    else : 
        home_login = False
    
    return render_template('door.html', switch = home_login)



@app.route(HOME_URL,methods=['GET','POST'])
def main():
    """ Session control"""
    if not session.get('logged_in'):
        return render_template('door.html')
    else:
        if request.method == 'POST' :
            session['logged_in'] = True
            return render_template('home.html')
        return render_template('home.html')
    
@app.route('/login/', methods=['GET', 'POST'])
def login():
    """Login Form"""
    if request.method == 'GET':
        return render_template('Login.html')
    else:
        name = request.form['username']
        passw = request.form['password']
        try:
            data = User.query.filter_by(username=name, password1=passw).first()
            if data is not None:
                session['logged_in'] = True
                session['name'] = name
                return redirect('/home')
            else:
                return '로그인 실패'
        except:
            return '로그인 실패'

 
@app.route('/register/', methods=['GET', 'POST'])
def register():
    """Register Form"""
    if request.method == 'POST':
        new_user = User(name=request.form['name'], 
                        username=request.form['username'], 
                        date = request.form['date'],
                        sex = request.form.get('sex',False),
                        password1=request.form['password1'],
                        password2=request.form['password2'])
        db.session.add(new_user)
        db.session.commit()
        return render_template('home.html')
    return render_template('register.html')

 
@app.route("/logout")
def logout():
    """Logout Form"""
    session['logged_in'] = False
    return redirect(url_for('home'))


@app.route('/survey')
def survey():
    return render_template('질병예측.html')

@app.route('/survey/result',methods=['GET','POST'])
def hello_world():
    mydict=request.form
    if request.method == 'POST':
        #mydict에서 문항 다 입력하고 거기서 
        survey_data = pd.DataFrame({
        'LQ_3EQL' : float(mydict['LQ_3EQL']),
        'N_MUFA' : float(mydict['N_MUFA']),
        'TOTAL_SLP_WD' : float(mydict['TOTAL_SLP_WD']),
        'N_PHOS' : float(mydict['N_PHOS']),
        'BD1_11' : float(mydict['BD1_11']),
        'HE_WT' : float(mydict['HE_WT']),
        'BP1' : float(mydict['BP1']),
        'N_FAT' : float(mydict['N_FAT']),
        'BH1' : float(mydict['BH1']),
        'AGE' : float(request.form['AGE']),
        'N_CAROT' : float(mydict['N_CAROT']),
        'BS13' : float(mydict['BS13']),
        'N_INTK' : float(mydict['N_INTK']),
        'N_EN' : float(mydict['N_EN']),
        'LQ_5EQL' : float(mydict['LQ_5EQL']),
        'BE3_85' : float(mydict['BE3_85']),
        'N_WATER' : float(mydict['N_WATER']),
        'HE_HT' : float(mydict['HE_HT']),
        'N_CHOL' : float(mydict['N_CHOL']),
        'N_NA' : float(mydict['N_NA']),
        'N_PROT' : float(mydict['N_PROT']),
        'SEX' : int(request.form['SEX']), #request.form['SEX']
        'BP7' : float(mydict['BP7']),
        'LQ_2EQL' : float(mydict['LQ_2EQL']),
        'HE_DBP' : float(mydict['HE_DBP']),
        'LQ_1EQL' : float(mydict['LQ_1EQL']),
        'BE3_81' : float(mydict['BE3_81']),
        'N_SFA' : float(mydict['N_SFA']),
        'HE_RPLS' : float(mydict['HE_RPLS']),
        'N_SUGAR' : float(mydict['N_SUGAR']),
        'SM_PRESNT' : float(mydict['SM_PRESNT']),
        'L_BR_FQ' : float(mydict['L_BR_FQ']),
        'HE_FH' : float(mydict['HE_FH']),
        'LQ_4EQL' : float(mydict['LQ_4EQL']),
        'MH_STRESS' : float(mydict['MH_STRESS']),
        'LQ4_00' : float(mydict['LQ4_00'])
        }, index=np.arange(1))
        
    result, data = model_data.model(survey_data, err9=True, err9value = 21) # 천식 에러 이슈면 err9 True, 강제로 값 21퍼센트
    ag = pd.read_csv('hyerim/data/data_ag.csv', index_col=0)
    current_user = session.get('name')
    result_db = meals.meals()
    #result_db.dis_results_input(dis_results_dic = result.to_dict(), id = str(current_user))
    #result_db.db_close()
    table_name = 'dis_results'
    result['user_id'] = str(current_user)
    result_db.insert_df(df = result, table = table_name, id = str(current_user))
    result_db.db_close()

    return render_template('detect_result.html', data=data, ag = ag, survey_data = survey_data, result = result)

@app.route(DETECTION_URL, methods=["GET","POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        




        # FILE = Path(__file__).resolve()
        # ROOT = FILE.parents[0]  # YOLOv5 root directory
        # if str(ROOT) not in sys.path:
        #     sys.path.append(str(ROOT))  # add ROOT to PATH
        # ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        source = "static/image1.jpg"
        img.save(source, format="JPEG")
        


        data = 'data/coco128.yaml' # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.25  # confidence threshold
        iou_thres=0.45  # NMS IOU threshold
        max_det=1000  # maximum detections per image
        device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False  # show results
        save_txt=True  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=False  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project='static/detect'  # save results to project/name
        # name='exp'  # save results to project/name
        name = session.get('name')
        exist_ok=True  # existing project/name ok, do not increment
        line_thickness=8  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=False  # use FP16 half-precision inference
        dnn=False  # use OpenCV DNN for ONNX inference

        db_disfood = meals.meals()
        current_user = session.get('name')
        machine129=db_disfood.dis_food(id = current_user, conf=50) # machine129 I want to change box colors
        db_disfood.db_close()

        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        weights = 'yolov5s.pt'
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        # model = torch.hub.load('../yolov5', 'custom', path='yolov5s.pt', source='local') 
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt) # LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str('가나다'))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    save_txt_reset = True
                    # save_txt reset
                    if save_txt_reset:
                        with open(f'{txt_path}.txt', 'w') as f:
                            f.write('')

                    # Write results
                    detected_foods = []
                    detected_bad_foods = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                            machine129_dic = {'jabgogbab':'잡곡밥', 'baechugimchi':'배추김치', 'doenjangjjigae':'된장찌개',
                            'gimchijjigae':'김치찌개', 'myeolchibokk-eum':'멸치볶음', 'Sigeumchinamul':'시금치나물', 
                            'gajinamul':'가지나물', 'gosalinamul':'고사리나물', 'miyeoggug':'미역국',
                            'gimbab':'김밥', 'bulgogi':'불고기', 'aehobagbokk-eum':'애호박볶음',
                            'musaengchae':'무생채', 'jabchae':'잡채', 'mugug':'무국', 
                            'godeung-eogu-i':'고등어구이', 'ssalbab':'쌀밥', 'gyelanhulai':'계란후라이',
                            'gyelanmal-i':'계란말이', 'jeyukbokk-eum':'제육볶음', 'siraegiguk':'시래기국', 
                            'ddangkongjorim':'땅콩조림'}

                            label_acc = label.split(' ')

                            if len(machine129) == 0:
                                annotator.box_label(xyxy, machine129_dic[label_acc[0]] + ' ' + label_acc[1], color=colors(c, True))
                                detected_foods.append(label_acc[0])
                            else:
                                if c in machine129:
                                    annotator.box_label(xyxy, machine129_dic[label_acc[0]], color=colors(0, True), safe129 = 'warn')
                                    detected_foods.append(machine129_dic[label_acc[0]])
                                    detected_bad_foods.append(machine129_dic[label_acc[0]])
                                else:
                                    annotator.box_label(xyxy, machine129_dic[label_acc[0]], color=colors(8, True), safe129 = 'safe')
                                    detected_foods.append(machine129_dic[label_acc[0]])
                
                detected_foods_set = list(set(detected_foods))
                detected_foods_str = ""
                detected_foods_nut = ""
                db = meals.meals()
                for detected_food_name in detected_foods_set:
                    detected_foods_str += detected_food_name + ', '
                    df = db.select_query(f"SELECT * FROM Food WHERE food_name = '{detected_food_name}';")
                    calorie = df.head(1)['calorie_kcal'][0]
                    carbo = df.head(1)['carbohydrate_g'][0]
                    protein = df.head(1)['protein_g'][0]
                    fat = df.head(1)['fat_g'][0]
                    detected_foods_nut += detected_food_name + f' 칼로리:{calorie} 탄수화물:{carbo} 단백질:{protein} 지방:{fat}, '

                detected_foods_str = detected_foods_str.rstrip(', ')
                detected_foods_nut = detected_foods_nut.rstrip(', ')
                session['detected_foods'] = detected_foods_str
                session['detected_foods_nut'] = detected_foods_nut
                
                detected_bad_foods_set = list(set(detected_bad_foods))
                session['detected_bad_foods'] = detected_bad_foods_set

                # Stream results
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


        db_meals = meals.meals()
        txt_label_path = f"static/detect/{name}/labels/image1.txt"
        current_user = session.get('name')
        db_meals.label_to_meals(path = txt_label_path, id=str(current_user))
        db_meals.db_close()

        img = f"detect/{name}/image1.jpg"
        return img, detected_bad_foods_set, detected_foods_set

    return render_template("detect.html")

@app.route('/predict/result', methods=["GET","POST"] )
def food_result():

    img, bad_food, food = predict()
    current_user = session.get('name') # ID
    db = meals.meals() # DB
    youngyangso = db.meals_to_3nutrient(id=current_user, recent = 1) # 식단의 탄단지 / 칼로리 (dic형태)

    #return render_template('객체-탐지-결과-페이지-2.html', img =img, bad_food = bad_food, food = food, youngyangso = youngyangso)
    return render_template('객체탐지결과페이지.html', img =img, bad_food = bad_food, food = food, youngyangso = youngyangso)


@app.route('/mypage',methods=['GET','POST'])
def mypage():
    # 로그인 유무 여부는 html에서 처리
    
    current_user = session.get('name')
    #현재 사용자 ID, 넘겨줘야 할 변수 1
    
    db = meals.meals()
    db2 = mypagefunction.db_class()
    
    # 1. 최근식단의 [탄, 단, 지] 값
    youngyangso = db.meals_to_3nutrient(id=current_user, recent = 1)

    # 2. 50이 넘는 [질병명, 질병확률]
    over_50 = db2.over_disease(id=current_user, conf=50, percent=True)

    # 3. 피해야 할 [음식명]
    bad_food = db.dis_food(id=current_user, conf=0.5, recent = 1, str=True)

    # 4. 90 이 넘는 [질병명]
    over_90 = db2.over_disease(id=current_user, conf=90, percent=False)

    # 5. meals table의 [date_time
    df = db.select_query(f"SELECT * FROM meals WHERE user_id = '{current_user}' ORDER BY date_time DESC;")
    meals_date = df.head(1)['date_time'][0]
    db.db_close()
    db2.db_close()

    
    # 1. 최근식단의 [탄, 단, 지] 값 // youngyangso
    # 2. 50이 넘는 [질병명, 질병확률] // over_50
    # 3. 피해야 할 [음식명] // bad_food
    # 4. 90이 넘는 [질병명] // over_90
    # 5. meals table의 [date_time] // meals_date
    
    
    #질병 확률 labels, value START #
    
    over_50.sort(key=itemgetter(1), reverse=True)  # or newlist = sorted(category, key=itemgetter(3))
    # 내림차순 정렬
    
    disease_labels = []
    disease_data = []
    backgroundcolor_level = []

    for disease_name, disease_percent in over_50 :
        if disease_percent >= 50 and disease_percent < 90 :
            backgroundcolor_level.append('#3CB371') 
        elif disease_percent >=90 and disease_percent <= 100 :
            backgroundcolor_level.append('#bd0404') #8B0000
        else :
            pass
        disease_labels.append(disease_name)
        disease_data.append(disease_percent)



  
    #질병 확률 labels, value END #
    
    # 3대 영양소 in Donut chart, 총 칼로리 kcal 따로 빼기. #
    nutrient_labels = []
    nutrient_data = []
    kcal_labels = []
    kcal_values = []
    
    for index, (key, elem) in enumerate(youngyangso.items()) :
        if index == 0 :
            nutirent_kcal = {key, elem}
        else :
            nutrient_labels.append(key)
            nutrient_data.append(elem)
    
    kcal_values.append(list(youngyangso.values())[0])
    kcal_labels.append(list(youngyangso.keys())[0])

    

    return render_template('mypage1.html', labels = disease_labels, data = disease_data, data3 = bad_food,
                           data4 = over_90, data5 = meals_date, data6 = current_user, nutrient_data = nutrient_data, nutrient_labels = nutrient_labels,
                           kcal_values = kcal_values, kcal_labels = kcal_labels, backgroundcolor_level = backgroundcolor_level, data_50 = over_50)
    #return render_template('mypage.html', data = df_dict)
    
@app.route('/hello',methods=['GET','POST'])
def example():
    return render_template('mypage1.html')


if __name__ == '__main__'  :
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    args = parser.parse_args()
#    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True, autoshape=True)  # force_reload = recache latest code
    # model = torch.hub.load('../yolov5', 'custom', path='yolov5s.pt', source='local') 
#    model = torch.hub.load(r'yolov5', 'yolov5s', path=r'yolov5s.pt', source='local')

    # model.eval()
    db.create_all()
    app.secret_key = "123"
    # secret_key는 서버상에 동작하는 어플리케이션 구분하기 위해 사용하고 복잡하게 만들어야 합니다.
    
    app.run(debug=True) 
    