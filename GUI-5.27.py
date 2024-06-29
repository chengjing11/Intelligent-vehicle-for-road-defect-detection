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
event = threading.Event()
global db, cursor, i

# 全局变量
weights_path = r"/home/jetauto/yolov5-master -yan/runs/train/ghost_res2net/weights/best.pt"
yaml_path = r"/home/jetauto/yolov5-master -yan/models/yolov5ghost_RES.yaml"
global detect_flag, realtime_flag





def detect(model, image_path):
    results = model(image_path)
    return results


class StartPage:
    def __init__(self, parent_window):
        global i
        i = 0
        parent_window.destroy()
        self.window = tk.Tk()
        self.window.title('智驭坦途--道路缺陷侦测系统')
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
        #self.model = None
        self.model = self.load_model(yaml_path, weights_path)
        self.createpage()
        self.filenames = []
        self.pic_filelist = []
        self.imgt_list = []
        self.image_labellist = []
        self.model_list = []
        self.model_list1 = []
        self.x = None

    def createpage(self):
        hex_color = "#{:02x}{:02x}{:02x}".format(236, 222, 254)
        hex_purple_color = "#{:02x}{:02x}{:02x}".format(210, 220, 250)
        menu = tk.Menu(self.window)
        self.window.config(menu=menu)
        self.frm1.config(bg=hex_color, height=500, width=870, relief=tk.RAISED, highlightbackground='red')
        self.frm1.place(x=20, y=135)
        self.frm2.config(bg=hex_color, height=80, width=1160, relief=tk.RAISED, highlightbackground='green')
        self.frm2.place(x=20, y=50)
        self.frm3.config(bg=hex_color, height=40, width=1160, relief=tk.RAISED, highlightbackground='blue')
        self.frm3.place(x=20, y=5)
        self.frm4.config(bg=hex_color, height=500, width=285, relief=tk.RAISED, highlightbackground='yellow')
        self.frm4.place(x=897, y=135)

        self.scr_ = scrolledtext.ScrolledText(self.frm4, width=29, height=12, font=("song ti", 12))
        self.scr_.place(x=8, y=270)
        self.uart_display_label = tk.Label(self.frm4, text="道路损伤评估结果：", font=("song ti", 12), bg=hex_color)
        self.uart_display_label.place(x=3, y=242, width=150, height=25)
        self.scr_GPS = scrolledtext.ScrolledText(self.frm4, width=29, height=12, font=("song ti", 12))
        self.scr_GPS.place(x=8, y=33)
        self.uart_GPS_label = tk.Label(self.frm4, text="GPS定位信息：", font=("song ti", 12), bg=hex_color)
        self.uart_GPS_label.place(x=3, y=5, width=110, height=25)
        self.result = tk.Label(self.frm1, text="检测结果显示", fg='black', font=("song ti", 20))
        self.result.place(x=510, y=5, width=350, height=320)

        tk.Label(self.frm3, text='智驭坦途--道路缺陷侦测系统', bg=hex_color, font=("song ti", 20)).place(x=480, y=2.5)
        tk.Button(self.frm2, text='系统初始化', font=("song ti", 12), command=self.sys_init, bg=hex_purple_color,
                  borderwidth=5, highlightthickness=2).place(x=30, y=20, width=180, height=40)
        tk.Button(self.frm2, text='实时检测', font=("song ti", 12), command=self.detect, borderwidth=5,
                  highlightthickness=2, bg=hex_purple_color).place(x=260, y=20, width=180, height=40)
        tk.Button(self.frm2, text='停止检测', font=("song ti", 12), command=self.stop_detect, borderwidth=5,
                  highlightthickness=2, bg=hex_purple_color).place(x=490, y=20, width=180, height=40)
        tk.Button(self.frm2, text='读图检测', font=("song ti", 12), command=self.readImage, borderwidth=5,
                  highlightthickness=2, bg=hex_purple_color).place(x=720, y=20, width=180, height=40)
        tk.Button(self.frm2, text='退出系统', font=("song ti", 12), command=self.window.destroy, borderwidth=5,
                  highlightthickness=2, bg=hex_purple_color).place(x=950, y=20, width=180, height=40)

        self.video = tk.Label(self.frm1, bg='gray', text='原图像显示', fg='black', font=("song ti", 20))
        self.video.place(x=10, y=5, height=490, width=490)


        self.scr = scrolledtext.ScrolledText(self.frm1, width=39, height=7.5, font=("song ti", 12))
        self.scr.place(x=510, y=358)
        self.uart_status_label = tk.Label(self.frm1, text="系统运行状态：", font=("song ti", 12), bg=hex_color)
        self.uart_status_label.place(x=505, y=330, width=120, height=25)

        self.window.mainloop()

    def stop_detect(self):
        global detect_flag
        detect_flag = False
        self.scr.insert(tk.END, "停止检测！\n")
        self.scr.see(tk.END)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.scr_GPS.insert(tk.END, f"{current_time} - GPS实时定位关闭！\n")
        self.scr_GPS.see(tk.END)

    def event(self):
        global detect_flag, realtime_flag
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
                    imgg = imgg.resize((490, 490), Image.ANTIALIAS)
                    imgtk = ImageTk.PhotoImage(image=imgg)

                    self.video.configure(image=imgtk)
                    self.video.image = imgtk

                    gps_thread = threading.Thread(target=self.gps_display)
                    gps_thread.start()

                    if realtime_flag:
                        result, detection_texts = self.detect_and_segment(frame)
                        result = result.resize((350, 320), Image.ANTIALIAS)
                        imgtk = ImageTk.PhotoImage(image=result)
                        self.result.configure(image=imgtk)
                        self.result.image = imgtk

                        self.scr_.insert(tk.END, "\n".join(detection_texts) + "\n")
                        self.scr_.see(tk.END)

                        # self.insert_sql(frame, np.array(seg_image))
                        self.insert_sql(frame)

                else:
                    break

            if self.cap:
                self.cap.release()
                self.cap = None

        cv2.destroyAllWindows()

        self.video = tk.Label(self.frm1, bg='gray', text='原图像显示', fg='black', font=("song ti", 20))
        self.video.place(x=10, y=5, height=490, width=490)
        self.result = tk.Label(self.frm1, text="检测结果显示", font=("song ti", 20))
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
        global detect_flag, realtime_flag
        detect_flag = False
        realtime_flag = False
        self.scr.insert(tk.END, "初始化系统...\n")
        self.con_sql()
        self.gps_init()
        self.scr.insert(tk.END, "已完成系统初始化！\n")

        if self.model is None:
            self.scr.insert(tk.END, "系统首次模型加载较慢，请稍候...\n")
            
            self.model = self.load_model(yaml_path, weights_path)
            #time.sleep(1)
            self.scr.insert(tk.END, "模型加载完成!\n")
            self.scr.see(tk.END)

        event.set()
        self.T = threading.Thread(target=self.event)
        self.T.setDaemon(True)
        self.T.start()

    def readImage(self):
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
            img = ImageTk.PhotoImage(img_open)

            self.video.config(image=img)
            self.video.image = img
            self.scr.insert(tk.END, '图片已加载！正在检测缺陷。。。\n')

            results = detect(self.model, self.file)
            rendered_image = results.render()[0]
            rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(rendered_image)
            result_image = result_image.resize((350, 320), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=result_image)
            self.result.configure(image=imgtk)
            self.result.image = imgtk
            self.scr.insert(tk.END, '缺陷检测已完成！\n')

        else:
            print("No file selected.")

    def detect(self):
        global realtime_flag, detect_flag
        realtime_flag = True
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

        threading.Thread(target=self.event).start()

    def detect_and_segment(self, frame):
        results = self.model([frame])
        rendered_image = results.render()[0]
        class_names = self.model.names

        # 提取检测结果信息
        detection_texts = []
        for *box, conf, cls in results.xyxy[0]:
            detection_texts.append(f"Class: {class_names[int(cls)]}, Confidence: {conf:.2f}")
        return Image.fromarray(rendered_image), detection_texts


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

    def insert_sql(self, crack_image):
        #self.gps_display() # 为了防止打印太快，在插入数据库时打印坐标信息
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
        cv2.imwrite(mask_path, crack_image)  # 先保存检测图像，代替予以分割图像
        #cv2.imwrite(mask_path, mask_image)
        # 保存数据到数据库
        sql = "INSERT INTO Crack (CrackPath, MaskPath) VALUES (%s, %s)"
        val = (crack_path, mask_path)
        cursor.execute(sql, val)
        db.commit()
    

    def gps_init(self):
        self.scr_GPS.insert(tk.END, "GPS初始化中...\n")
        self.scr_GPS.see(tk.END)
        # try:
        self.x=L76X.L76X()
        self.x.L76X_Set_Baudrate(9600)
        self.x.L76X_Send_Command(self.x.SET_NMEA_BAUDRATE_115200)
        time.sleep(0.1)
        self.x.L76X_Set_Baudrate(115200)

        # !!! time
        self.x.L76X_Send_Command(self.x.SET_POS_FIX_400MS);

        #Set output message
        self.x.L76X_Send_Command(self.x.SET_NMEA_OUTPUT);

        #x.L76X_Exit_BackupMode();


    def gps_display(self):
        self.x.L76X_Gat_GNRMC()
        if(self.x.Status == 1):
            print('Already positioned')
        else:
            print('No positioning')
        #print('Time %d:'%self.x.Time_H,end='')
        #print('%d:'%self.x.Time_M,end='')
        #print('%d'%self.x.Time_S)

        #print('Lon = %f'%self.x.Lon,end='')
        #print(' Lat = %f'%self.x.Lat)
        self.x.L76X_Baidu_Coordinates(self.x.Lat, self.x.Lon)
        print('Baidu coordinate %f'%self.x.Lat_Baidu,end='')
        #print(',%f'%self.x.Lon_Baidu)
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.scr_GPS.insert(tk.END, f"{current_time}\n")
        self.scr_GPS.insert(tk.END, '经度：%f\n'%self.x.Lon)
        self.scr_GPS.insert(tk.END, '纬度：%f\n'%self.x.Lat)
        self.scr_GPS.see(tk.END)

if __name__ == '__main__':
    window = tk.Tk()
    StartPage(window)
    window.mainloop()
