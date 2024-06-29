import torch
import yaml
import os
import threading
import time
import datetime
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import mysql.connector
import L76X
import math
import hdlogging
import sys
from tkinter import filedialog, dialog
import subprocess
import time
import asyncio

from skimage import transform
from test import normPRED
from models.myu2net import U2NETP
from models.myu2net import U2NET
from PIL import Image, ImageDraw, ImageFont

event = threading.Event()
global db, cursor, i
global notfirst_load
notfirst_load = False
# 全局变量
weights_path = r"/home/jetauto/yolov5-master -yan2024_6_6/runs/train/ghost_res2net/weights/best0.pt"
yaml_path = r"/home/jetauto/yolov5-master -yan2024_6_6/models/yolov5ghost_RES.yaml"
global detect_flag

hex_color = "#{:02x}{:02x}{:02x}".format(236, 222, 254)
hex_purple_color = "#{:02x}{:02x}{:02x}".format(210, 220, 250)


class Diary:
    def __init__(self, parent_window):

        global detect_flag

        parent_window.update()
        parent_window.destroy()  # 销毁主界面
        self.window = tk.Tk()  # 初始框的声明
        self.window.title('检测日志系统')
        #self.window.attributes("-fullscreen", True)  # 允许更改大小
        self.window.geometry("1024x600")
        window_width = 1200
        window_height = 640
        self.WIDE, self.HEIGHT =window_width, window_height
        self.scr_w, self.scr_h = int(self.WIDE / 800 * 650 * 107 / 1000), int((self.HEIGHT / 600 * 560 - self.HEIGHT / 600 * 325) * 100 / 1000)
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        # 设置窗口是否可变长、宽，True：可变，False：不可变
        self.window.resizable(width=False, height=False)
        self.frm1 = tk.Frame(self.window)
        self.frm2 = tk.Frame(self.window)
        self.frm3 = tk.Frame(self.window)
        self.createpage()

    def createpage(self):
        # 界面内的所有组件都根据屏幕的实际大小成比例缩放扩大
        self.frm1.config(bg=hex_color, height=500, width=1160, relief=tk.RAISED)
        self.frm1.place(x=20, y=135)
        self.frm2.config(bg=hex_color, height=80, width=1160, relief=tk.RAISED)
        self.frm2.place(x=20, y=50)
        self.frm3.config(bg=hex_color, height=40, width=1160, relief=tk.RAISED)
        self.frm3.place(x=20, y=5)

        # frm3下的Label
        tk.Label(self.frm3, text='日志管理页面', bg=hex_color, font=("song ti", 20)).place(x=480, y=2.5)

        # frm2下的Button
        tk.Button(self.frm2, text='查找日志', font=("song ti", 12), command=self.log_select,borderwidth=5,highlightthickness=2,bg=hex_purple_color).place(x=690, y=20, width=180, height=50)
        tk.Button(self.frm2, text='返回主页', font=('song ti', 12), command=self.back, borderwidth=5,highlightthickness=2,bg=hex_purple_color).place(x=290, y=20, width=180, height=50)
        # frm1下的控件
        tk.Label(self.frm1, bg='white', text='检测日志管理页面', fg='black', font=('song ti', 20, 'bold')).place(x=5, y=5,height=490,width=1150)

    def log_select(self):
        file_path = filedialog.askopenfilename(title=u'查看日志', initialdir=(
            os.path.expanduser('/home/jetauto/yolov5-master -yan2024_6_6/logs')))  # 设置文件夹路径
        if file_path is not None:
            self.scr = scrolledtext.ScrolledText(self.frm1, width=self.scr_w+38, height=self.scr_h+6, # weight 104  height 25
                    font=("song ti", 12))  # 滚动文本框（宽，高（这里的高应该是以行数为单位），字体样式）
            tail_lines = []

            self.scr.place(x=2, y=2)  # 滚动文本框在页面的位置
            # 打开路径下的日志文件，并将其插入到tk中来
            with open(file_path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    self.scr.insert(tk.END, line)
                    tail_lines.append(line)

                last_line = tail_lines[-1]

                if 'Defect Ratio' in last_line:
                    # 从最后一行中提取ratio值
                    ratio_value = float(last_line.split(':')[-1].strip())
                    # 将ratio值插入到文本框中
                    self.scr.insert(tk.END, f"缺陷总数/实时检测时间比值: {ratio_value:.2f}\n")
                    print("找到 ratio 值。")

                else:
                    self.scr.insert(tk.END, '未找到 Defect Ratio 值。\n')
                    print("未找到 ratio 值。")

                    # if 'D11' in last_line2:
                    #     # 从最后一行中提取ratio值
                    #     ratio_value = float(last_line2.split(':')[-1].strip())

                image_path = '/home/jetauto/yolov5-master -yan2024_6_6/1217.png'  # 替换为你的图片路径  cha
                image_path2 = '/home/jetauto/yolov5-master -yan2024_6_6/13.png'  # 替换为你的图片路径  you
                image_path3 = '/home/jetauto/yolov5-master -yan2024_6_6/0818.png'  # 替换为你的图片路径  liang
                image_path4 = '/home/jetauto/yolov5-master -yan2024_6_6/1102.png'  # 替换为你的图片路径

                if ratio_value <= 10:
                    img = Image.open(image_path2)
                    draw = ImageDraw.Draw(img)
                    #font = ImageFont.truetype("song ti", 12)  # 设置字体和字号
                    text = f"91            1          1            1           Great"
                elif 10 < ratio_value < 100:
                    img = Image.open(image_path3)
                    draw = ImageDraw.Draw(img)
                    #font = ImageFont.truetype("song ti", 12)  # 设置字体和字号
                    text = f"82            3           1           2           Good"
                elif 100 < ratio_value < 500:
                    img = Image.open(image_path4)
                    draw = ImageDraw.Draw(img)
                    #font = ImageFont.truetype("song ti", 12)  # 设置字体和字号
                    text = f"53            5           4           5            Bad"
                else:
                    img = Image.open(image_path)
                    draw = ImageDraw.Draw(img)
                    #font = ImageFont.truetype("song ti", 12)  # 设置字体和字号
                    text = f"31            11          13         12          Worth"
               

                text_width, text_height = draw.textsize(text)
                position = (170, 440)  # 文本显示位置，根据需求调整
                draw.text(position, text, fill="Black")

                img.show()
                self.scr.insert(tk.END, '本次检测结束!\n')
                self.scr.insert(tk.END, '本次路段1缺陷检测已完毕！缺陷等级为：差  （等级分为：优、良、中、差）\n')
                self.scr.insert(tk.END, '请道路维修部门尽快对路段1进行养护！\n!')


    def back(self):
        StartPage(self.window)  # 显示主窗口 销毁本窗口


class StartPage:
    def __init__(self, parent_window):
        global i
        global start_time, end_time, defect_count
        duration = None
        start_time = None
        end_time = None
        defect_count = 0
        i = 0
        parent_window.update()
        parent_window.destroy()

        self.window = tk.Tk()
        self.window.title('Smart Vision')
        self.window.geometry("1024x600")
        window_width = 1200
        window_height = 640
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))
        self.window.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
        self.window.resizable(width=True, height=True)
        self.frm1 = tk.Frame(self.window)
        self.frm2 = tk.Frame(self.window)
        self.frm3 = tk.Frame(self.window)
        self.frm4 = tk.Frame(self.window)
        self.device = ''
        self.model = None
        #self.model = self.load_model(yaml_path, weights_path)
        self.createpage()
        self.filenames = []
        self.pic_filelist = []
        self.imgt_list = []
        self.image_labellist = []
        self.model_list = []
        self.model_list1 = []
        self.x = None

    def createpage(self):

        menu = tk.Menu(self.window)
        self.window.config(menu=menu)
        self.frm1.config(bg=hex_color, height=500, width=870, relief=tk.RAISED)
        self.frm1.place(x=20, y=135)
        self.frm2.config(bg=hex_color, height=80, width=1160, relief=tk.RAISED)
        self.frm2.place(x=20, y=50)
        self.frm3.config(bg=hex_color, height=40, width=1160, relief=tk.RAISED)
        self.frm3.place(x=20, y=5)
        self.frm4.config(bg=hex_color, height=500, width=285, relief=tk.RAISED)
        self.frm4.place(x=897, y=135)

        self.scr_ = scrolledtext.ScrolledText(self.frm4, width=29, height=12, font=("song ti", 12))
        self.scr_.place(x=8, y=270)
        self.uart_display_label = tk.Label(self.frm4, text="道路损伤评估结果：", font=("song ti", 12), bg=hex_color)
        self.uart_display_label.place(x=3, y=242, width=150, height=25)
        self.scr_GPS = scrolledtext.ScrolledText(self.frm4, width=29, height=12, font=("song ti", 12))
        self.scr_GPS.place(x=8, y=33)
        self.uart_GPS_label = tk.Label(self.frm4, text="GPS定位信息：", font=("song ti", 12), bg=hex_color)
        self.uart_GPS_label.place(x=3, y=5, width=110, height=25)
        self.result = tk.Label(self.frm1, text="语义分割结果显示", fg='black', font=("song ti", 20))
        self.result.place(x=510, y=5, width=350, height=320)

        tk.Label(self.frm3, text='智驭坦途--道路缺陷侦测系统', bg=hex_color, font=("song ti", 20)).place(x=430, y=2.5)
        #tk.Label(self.frm3, text='Smart Vision', bg=hex_color, font=("fang song", 12)).place(x=1000, y=4)
        tk.Button(self.frm2, text='系统初始化', font=("song ti", 12), command=self.sys_init, bg=hex_purple_color,
                borderwidth=5, highlightthickness=2).place(x=45, y=20, width=150, height=40)
        tk.Button(self.frm2, text='实时检测', font=("song ti", 12), command=self.detect, borderwidth=5,
                highlightthickness=2, bg=hex_purple_color).place(x=225, y=20, width=150, height=40)
        tk.Button(self.frm2, text='停止检测', font=("song ti", 12), command=self.stop_detect, borderwidth=5,
                highlightthickness=2, bg=hex_purple_color).place(x=405, y=20, width=150, height=40)
        tk.Button(self.frm2, text='读图检测', font=("song ti", 12), command=self.readImage, borderwidth=5,
                highlightthickness=2, bg=hex_purple_color).place(x=585, y=20, width=150, height=40)
        tk.Button(self.frm2, text='检测日志', font=("song ti", 12), command=lambda: Diary(self.window), borderwidth=5,highlightthickness=2, bg=hex_purple_color).place(x=765, y=20, width=150,height=40)
        tk.Button(self.frm2, text='退出系统', font=("song ti", 12), command=self.end, borderwidth=5,
                highlightthickness=2, bg=hex_purple_color).place(x=945, y=20, width=150, height=40)


        self.video = tk.Label(self.frm1, bg='gray', text='检测结果显示', fg='black', font=("song ti", 20))
        self.video.place(x=10, y=5, height=490, width=490)


        self.scr = scrolledtext.ScrolledText(self.frm1, width=39, height=7.5, font=("song ti", 12))
        self.scr.place(x=510, y=358)
        self.uart_status_label = tk.Label(self.frm1, text="系统运行状态：", font=("song ti", 12), bg=hex_color)
        self.uart_status_label.place(x=505, y=330, width=120, height=25)

        self.window.mainloop()

    def end(self):
        self.window.destroy()
        cv2.destroyAllWindows()
        sys.exit(0)  # 全部关闭
        time.sleep(2)

    def get_current_frame(self):
        # 返回最后一帧图像
        global last_frame
        return last_frame

    def stop_detect(self):
        global detect_flag
        detect_flag = False
        subprocess.run(['rosservice', 'call', '/lidar_app/set_running', 'data: 0'], check=True)
        #subprocess.run(['bash', '-c', 'source ~/.bashrc && rosservice call /lidar_app/set_running "data: 0"'], check=True)
        self.scr.insert(tk.END, "停止检测！\n")
        self.scr.see(tk.END)
        #hdlogging.info('停止检测！')
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.scr_GPS.insert(tk.END, f"{current_time} - GPS实时定位关闭！\n")
        self.scr_GPS.see(tk.END)
        
        #create_and_run_script('script3.sh', [
        #'rosservice call /lidar_app/set_running "data: 0"'
        #])
        

    def event(self):
        global detect_flag
        self.if_exit = False

        while detect_flag:
            self.cap = cv2.VideoCapture(0)
            time.sleep(0.2)
            if not self.cap.isOpened():
                print("Error opening video stream or file")
                break

            while self.cap.isOpened() and detect_flag:
                ret, frame = self.cap.read()
                if ret:
                    height, width, _ = frame.shape
                    size = min(height, width)
                    start_x = (width - size) // 2
                    start_y = (height - size) // 2
                    end_x = start_x + size
                    end_y = start_y + size
                    frame = frame[start_y:end_y, start_x:start_x + size]
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    imgg = Image.fromarray(cv2image)

                    gps_thread = threading.Thread(target=self.gps_display)
                    #gps_thread.setDaemon(True)
                    gps_thread.start()

                    result, detection_texts, ratio = self.detect_and_segment(frame)
                    result = result.resize((490, 490), Image.ANTIALIAS)
                    imgtk = ImageTk.PhotoImage(image=result)
                    self.video.configure(image=imgtk)
                    self.video.image = imgtk

                    self.scr_.insert(tk.END, "\n".join(detection_texts) + "\n")
                    self.scr_.see(tk.END)
                    detection_text_str = "\n".join(detection_texts)  # 使用换行符将列表中的每个元素连接成一个字符串
                    if detection_text_str:
                        hdlogging.info(detection_text_str)
                    # 语义分割
                    seg_image = self.meg_image(cv2image)
                    imgtk2 = ImageTk.PhotoImage(image=seg_image)
                    self.result.configure(image=imgtk2)
                    self.result.image = imgtk2

                    self.insert_sql(np.array(result), np.array(seg_image))

                        #self.insert_sql(frame)
                        #self.gps_display() # 为了防止打印太快，在插入数据库时打印坐标信息

                else:
                    break

            if self.cap:
                self.cap.release()
                self.cap = None

        cv2.destroyAllWindows()

        self.video = tk.Label(self.frm1, bg='gray', text='检测结果显示', fg='black', font=("song ti", 20))
        self.video.place(x=10, y=5, height=490, width=490)
        self.result = tk.Label(self.frm1, text="语义分割结果显示", font=("song ti", 20))
        self.result.place(x=510, y=5, width=350, height=320)

    def load_model(self, yaml_path, weights_path):
        if not os.path.exists(yaml_path):
            print(f"Error: 配置文件路径不存在: {yaml_path}")
            return None
        if not os.path.exists(weights_path):
            print(f"Error: 权重文件路径不存在: {weights_path}")
            return None

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        print("加载模型中，请稍候...")
        #self.scr.insert(tk.END, "加载模型中，请稍候...\n")
        #self.scr.see(tk.END)
        model = torch.hub.load('./', 'custom', path=weights_path, source='local')
        print("模型已成功加载！")
        #self.scr.insert(tk.END, "模型已成功加载！\n")
        #self.scr.see(tk.END)
        return model


    def sys_init(self):
        global detect_flag
        detect_flag = False
        self.scr.insert(tk.END, "初始化系统...\n")
        #hdlogging.info('初始化系统...')
        self.con_sql()
        self.gps_init()
        self.scr.insert(tk.END, "已完成系统初始化！\n")

        if self.model is None:
            self.scr.insert(tk.END, "系统首次模型加载较慢，请稍候...\n")

            self.model = self.load_model(yaml_path, weights_path)
            #time.sleep(1)
            self.scr.insert(tk.END, "模型加载完成!\n")
            self.scr.see(tk.END)
            #hdlogging.info('模型加载完成！')

        self.load_net()

        event.set()
        self.T = threading.Thread(target=self.event)
        self.T.setDaemon(True)
        self.T.start()

    def readImage(self):
        global i
        if self.model is None:
            self.scr.insert(tk.END, "系统首次模型加载较慢，请稍候...\n")

            self.model = self.load_model(yaml_path, weights_path)
            #time.sleep(1)
            self.scr.insert(tk.END, "模型加载完成!\n")
            self.scr.see(tk.END)

        self.file = filedialog.askopenfilename(title='选择图片', filetypes=[('image', '*.jpg *.png *.jpeg *.bmp')])
        if self.file:
            img_open = Image.open(self.file)
            img_open = img_open.resize((490, 490), Image.ANTIALIAS)

            self.scr.insert(tk.END, '图片已打开！正在处理。。。\n')

            #results = detect(self.model, self.file)  # 检测
            results = self.model(self.file)

            rendered_image = results.render()[0]

            rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(rendered_image)
            result_image = result_image.resize((490, 490), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=result_image)
            self.video.configure(image=imgtk)
            self.video.image = imgtk
            self.scr.insert(tk.END, '图片处理已完成！\n')
            if self.file:
                if i == 0:
                    self.load_net()
            image = Image.open(self.file).resize((350, 320), Image.ANTIALIAS)
            cv_img = cv2.imread(self.file)
            imgtk2 = ImageTk.PhotoImage(image=image)

            seg_image = self.meg_image(cv_img)  # 分割

            imgtk2 = ImageTk.PhotoImage(image=seg_image)
            self.result.configure(image=imgtk2)
            self.result.image = imgtk2
            self.scr.insert(tk.END, '语义分割处理已完成！\n') 
            i = 1
        else:
            print("No file selected.")

    def detect(self):
        global detect_flag
        global notfirst_load       
        detect_flag = True

        if self.model is None:
            self.scr.insert(tk.END, "系统首次模型加载较慢，请稍候...\n")

            self.model = self.load_model(yaml_path, weights_path)
            #time.sleep(1)
            self.scr.insert(tk.END, "模型加载完成!\n")
            self.scr.see(tk.END)
        #self.model = self.load_model(yaml_path, weights_path)

        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.scr_GPS.insert(tk.END, f"{current_time} - GPS实时定位开启！\n")
        self.scr_GPS.see(tk.END)
        self.scr.insert(tk.END, "开始实时检测！\n")
        self.scr.see(tk.END)
        #subprocess.run(['bash', '-c', 'source ~/.bashrc && rosservice call /lidar_app/set_running "data: 1"'], check=True)
        #create_and_run_script('script3.sh', [
        #'rosservice call /lidar_app/enter "{}"',
        #'rosservice call /lidar_app/set_running "data: 1"'
        #])

        #threading.Thread(target=self.event).start()
        self.T = threading.Thread(target=self.event)
        self.T.setDaemon(True)
        self.T.start()
        if notfirst_load:
            #subprocess.run(['rosservice', 'call', '/lidar_app/enter', '{}'], check=True)
            subprocess.run(['rosservice', 'call', '/lidar_app/set_running', 'data: 1'], check=True)
        #if_first_load = if_first_load+1
        notfirst_load = True

    def detect_and_segment(self, frame):
        global defect_count
        start_time = datetime.now()
        results = self.model([frame])
        rendered_image = results.render()[0]
        class_names = self.model.names
        # starttime = datetime.datetime.now()
        # 提取检测结果信息
        detection_texts = []
        end_time = datetime.now()
        detection_time = (end_time - start_time).total_seconds()

        for *box, conf, cls in results.xyxy[0]:
            defect_count += 1  # 增加缺陷计数
            # 将defect_ratio添加到detection_texts
            # if cls == D11
            detection_texts.append(f"Class: {class_names[int(cls)]}, Confidence: {conf:.2f}, ADD:{defect_count:.2f}")

        if detection_time != 0:
            defect_ratio = defect_count / detection_time
        else:
            defect_ratio = 0

        # 如果有缺陷数和检测时间，就将defect_ratio添加到detection_texts
        if defect_count != 0 and detection_time != 0:
            detection_texts.append(f"Defect Ratio: {defect_ratio:.2f}")

        # self.scr.insert(tk.END, f"缺陷总数/实时检测时间比值: {defect_ratio:.2f}\n")
        # self.scr.see(tk.END)

        return Image.fromarray(rendered_image), detection_texts, defect_ratio

    def meg_image(self, image):
        output_size = 320
        image = transform.resize(image, (output_size, output_size), mode='constant')

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        tmpImg = tmpImg.transpose((2, 0, 1))
        input_tensor = torch.from_numpy(np.ascontiguousarray(tmpImg)).unsqueeze(0).type(torch.FloatTensor).cuda()
        with torch.no_grad():
            _, _, _, _, d0, _, _, _ = self.net(input_tensor)
        result = d0[:, 0, :, :]
        pred = normPRED(result)
        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        result = Image.fromarray(predict_np * 255).convert('RGB')
        result = result.resize((350, 320), Image.ANTIALIAS)
        return result

    def load_net(self):
        self.scr.insert(tk.END, '正在加载语义分割模型...\n')
        self.net = U2NETP(3, 1).cuda().eval()
        self.net.load_state_dict(torch.load('u2net/best.pth'))
        self.scr.insert(tk.END, '语义分割模型加载完成！\n')

    def con_sql(self):
        try:
            global db, cursor
            db = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    # password="1820540722",
                    database="CrackData"
                    )
            cursor = db.cursor()
            self.scr.insert(tk.END, '数据库连接成功！\n')
        except mysql.connector.Error as err:
            self.scr.insert(tk.END, f'数据库连接失败:{err}！\n')
        #except:
         #   self.scr.insert(tk.END, '数据库连接失败！\n')

    def close_sql(self):
        db.close()
        cursor.close()
        self.scr.insert(tk.END, '数据库连接已关闭！\n')

    def insert_sql(self, crack_image, seg_image):

        # 设置保存图像的目录和分割后图像的目录
        save_dir = "Crack_System/Crack"
        segmented_dir = "Crack_System/Mask"
        current_timestamp = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')
        crack_path = os.path.join(save_dir, current_timestamp + ".jpg")
        # Add watermark with GPS coordinates to the crack_image
        watermark = f"Latitude: {self.x.Lat}, Longitude: {self.x.Lon}"
        img_with_watermark = cv2.putText(crack_image, watermark, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite(crack_path, img_with_watermark)

        #cv2.imwrite(crack_path, crack_image)
        mask_path = os.path.join(segmented_dir, current_timestamp + ".png")
        cv2.imwrite(mask_path, seg_image)  # 先保存检测图像，代替予以分割图像
        #cv2.imwrite(mask_path, mask_image)
        # 保存数据到数据库
        sql = "INSERT INTO Crack (CrackPath, MaskPath) VALUES (%s, %s)"
        val = (crack_path, mask_path)
        cursor.execute(sql, val)
        db.commit()


    def gps_init(self):
        # try:
        self.x=L76X.L76X()
        self.x.L76X_Set_Baudrate(9600)
        self.x.L76X_Send_Command(self.x.SET_NMEA_BAUDRATE_115200)
        time.sleep(0.1)
        self.x.L76X_Set_Baudrate(115200)

        # !!! time
        self.x.L76X_Send_Command(self.x.SET_POS_FIX_1S);

        #Set output message
        self.x.L76X_Send_Command(self.x.SET_NMEA_OUTPUT);
        self.scr_GPS.insert(tk.END, "GPS已完成初始化\n")
        self.scr_GPS.see(tk.END)
        #x.L76X_Exit_BackupMode();


    def gps_display(self):
        self.x.L76X_Gat_GNRMC()
        if(self.x.Status == 1):
            print('Already positioned')
        else:
            print('No positioning')

        self.x.L76X_Baidu_Coordinates(self.x.Lat, self.x.Lon)
        #print('Baidu coordinate %f'%self.x.Lat_Baidu,end='')
        #print(',%f'%self.x.Lon_Baidu)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.scr_GPS.insert(tk.END, f"{current_time}\n")
        self.scr_GPS.insert(tk.END, '经度：%f\n'%self.x.Lon)
        self.scr_GPS.insert(tk.END, '纬度：%f\n'%self.x.Lat)
        self.scr_GPS.see(tk.END)
        #hdlogging.info('经度：%f\n'%self.x.Lon, '纬度：%f\n'%self.x.Lat)
        #hdlogging.info('纬度：%f\n'%self.x.Lat)

def create_and_run_script(script_name, commands):
    # 创建并写入 shell 脚本
    with open(script_name, 'w') as file:
        file.write("#!/bin/bash\n")
        file.write("export LC_ALL=C\n")  # 设置locale
        file.write("export LANG=C\n")    # 设置locale
        file.write("source ~/.bashrc\n")  # 确保加载 bash 配置文件
        file.write("conda init\n")
        file.write("conda activate yds\n")
        for command in commands:
            file.write(f"{command}\n")

    # 赋予脚本执行权限
    subprocess.run(['chmod', '+x', script_name], check=True)

    # 在新终端中运行 shell 脚本
    #subprocess.Popen(['gnome-terminal', '--', f"./{script_name}"])
    with open('script_output.log', 'a') as out, open('script_error.log', 'a') as err:
        subprocess.Popen([f"./{script_name}"], stdout=out, stderr=err, shell=True)
    #subprocess.Popen(['xterm', '-hold', '-e', f"./{script_name}"])

if __name__ == '__main__':

    # 根据特定条件执行
    time.sleep(16)

    hdlogging.init("logMain", "DEBUG")
    window = tk.Tk()
    StartPage(window)
    window.mainloop()

