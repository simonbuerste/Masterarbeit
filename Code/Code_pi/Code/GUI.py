from PIL import Image
from PIL import ImageTk
import tkinter as tki
import numpy as np
import threading
import datetime
import imutils
import cv2
import os

#import tkFont
from tkinter import ttk
from ttkthemes import themed_tk as tk

from ModulA import modul_a
from FuzzyARTMAP import FuzzyARTMAP

# Parameters for the Pipeline are set here
params = {
    'no_classes':           10,
    'modul_b_alpha':        0.2,
    'modul_b_rho':          0.85,
    'modul_b_s':            1.05,
    'modul_b_epsilon':      0.001
}

class PhotoGui:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.plot_img = None
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.features_np_normed = None
        #self.font = tkFont.Font(family='Arial', size=10)
        
        # initialize the root window and image panel
        bck_color = '#f0f0f0'
        btn_width = 25
        
        self.root = tk.ThemedTk()# tk.Tk()
        self.root.get_themes()
        self.root.set_theme("arc") #radiance")# plastik")# clearlooks")
        self.root.configure(background=bck_color)
        self.panel = None
        
        tki.Grid.rowconfigure(self.root, [0 ,1, 2, 3, 4, 5], weight=1)
        tki.Grid.columnconfigure(self.root, [0, 1], weight=1)
        
        # initialize the class name
        self.class_name = tki.StringVar()
        
        # initialize the list for class names
        self.class_name_list = []
        
        # create a button, that when pressed, will take the current
        # frame and save it to file
        self.btn_classify = ttk.Button(self.root, text="Klassifizieren!", command=self.takeSnapshot)
        self.btn_classify.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")
        self.btn_classify.configure(width=btn_width)
        
        # Set a Entry-Field for Class Name
        self.entry_class_name = ttk.Entry(self.root, state='disable')
        self.entry_class_name.grid(row=2, column=0, padx=10, pady=10, sticky="nswe")
        self.entry_class_name.configure(width=26)
        
        # create a button, that when pressed, reads the
        # class name
        self.btn_class_name = ttk.Button(self.root, text="Klassenname einlesen!", state='disable',
                                        command=lambda: self.class_name.set(self.entry_class_name.get()))
        self.btn_class_name.grid(row=3, column=0, padx=10, pady=10, sticky="nswe")
        self.btn_class_name.configure(width=btn_width)
        
        # create a button, that when pressed, consolidate the
        # networks representation
        self.btn_consolidate = ttk.Button(self.root, text="Konsolidieren!", command=self.consolidate)
        self.btn_consolidate.grid(row=4, column=0, padx=10, pady=10, sticky="nswe")
        self.btn_consolidate.configure(width=btn_width)
        
        # create a button, that can be pressed when prediction of
        # network is wrong and corect group name can be given
        self.btn_wrong_pred = ttk.Button(self.root, text="Falsche Klasse!", state='disable', command=self.train_new_sample)
        self.btn_wrong_pred.grid(row=5, column=0, padx=10, pady=10, sticky="nswe")
        self.btn_wrong_pred.configure(width=btn_width)
        
        # Show the snapshotted Image from Video
        self.snapshot_img = ttk.Label(self.root)
        self.snapshot_img.grid(row=0, column=1, padx=10, pady=10)#, sticky="nswe")
        self.snapshot_img.configure(background=bck_color)
        
        # Set a Label for the Prediction of the Network
        self.label_pred = ttk.Label(self.root)
        self.label_pred.grid(row=1, column=1, padx=10, pady=10)#, sticky="nswe")
        self.label_pred.configure(background=bck_color)
        
        # Set up a Listbox for the Matching-Values
        self.listBox = ttk.Treeview(self.root, columns=("one"), height=7)
        self.listBox.heading("#0", text="Klassenname", anchor=tki.W)
        self.listBox.column("#0", minwidth=0, width=100) # stretch=True)
        self.listBox.heading("one", text="Matchingwerte", anchor=tki.W)
        self.listBox.column("one", minwidth=0, width=100) # stretch=True) 
        self.listBox.grid(row=2, column=1, padx=10, pady=10, rowspan=4, sticky="nswe")
        
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        #self.thread = threading.Thread(target=self.videoLoop, args=())
        #self.thread.start()
        self.videoLoop()
        
        # set the title and icon of the GUI
        icon = tki.Image("photo", file='/home/pi/Desktop/Code/Download.gif')
        #icon = tki.PhotoImage(file='/home/pi/Code_MA/Code/Download.gif')
        self.root.call('wm', 'iconphoto', self.root._w, icon)
        #self.root.iconphoto(True, icon)
        self.root.wm_title("Demonstrator Lifelong Learning")
        # set a callback to handle when the window is closed
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        
        # The Feature Extraction Module A is called and created/downloaded with the corresponding image size
        self.modulA = modul_a(image_size=224)

        # Incremental Classifier in Module B is initialized with desired Parameters
        self.modulB = FuzzyARTMAP(alpha=params["modul_b_alpha"], rho=params["modul_b_rho"],
                                  n_classes=params["no_classes"], s=params["modul_b_s"],
                                  epsilon=params["modul_b_epsilon"])
        
    def videoLoop(self):
        # keep looping over frames until we are instructe to stop
        if not self.stopEvent.is_set():
            # grab the frame from the video stream and resize it to
            # have a defined img size
            self.frame = self.vs.read()
            # self.frame = imutils.resize(self.frame, width=224)
            
            # OpenCV represents images in BGR order: however PIL
            # represents images in RGB order, so we need to swap
            # the channels, then convert to PIL and ImageTk format
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            
            # if the panel is None, we need to initialize it
            if self.panel is None:
                self.panel = ttk.Label(image=image)
                self.panel.image = image
                self.panel.grid(row=0, column=0, padx=10, pady=10)
            # otherwise, simply update the panel
            else:
                self.panel.configure(image=image)
                self.panel.image = image
        
        self.panel.after(10, self.videoLoop)
                
    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        # ts = datetime.datetime.now()
        # filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        # p = os.path.sep.join((self.outputPath, filename))
        
        # save the file
        # cv2.imwrite(p, self.frame.copy())
        # print("[INFO] saved {}".format(filename))
        
        image_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image_rgb = Image.fromarray(image_rgb)
        self.plot_img = ImageTk.PhotoImage(image_rgb)
        self.snapshot_img.configure(image=self.plot_img)
                
        features = self.modulA.predict(np.expand_dims(self.frame.copy(), axis=0), steps=1)
        features_np = np.array(features)
        self.features_np_normed = features_np/np.sum(features_np)
        
        prediction, matching, label_class = self.modulB.test(self.features_np_normed)
        if prediction == -1:
            self.label_pred.config(text=("Klasse unbekannt!\nGeben Sie bitte einen Klassennamen ein"))
        else:
            self.label_pred.config(text="Prädizierte Klasse: {}".format(str(self.class_name_list[int(prediction[0])])))
        
        # Just write Matching-Values and their class names to GUI if available
        if np.all(label_class != -1):
            self.listBox.delete(*self.listBox.get_children())
            # Sort negative Matching Array for Descending Order
            idx_descending = np.argsort(-matching)
            #for j, m in zip(label_class, np.around(matching, 2)):
            for i, idx in enumerate(idx_descending):
                if i == 0 and prediction != -1:
                    self.listBox.insert("", "end", text=str(np.asarray(self.class_name_list)[label_class[idx]]),
                                        values=np.around(matching[idx], 2), tags='winner')
                else:
                    self.listBox.insert("", "end", text=str(np.asarray(self.class_name_list)[label_class[idx]]),
                                        values=np.around(matching[idx], 2), tags='looser')
            
            self.listBox.tag_configure('winner', background='#00ff0a', font='bold')

        if prediction == -1:
            self.train_new_sample()
        else:
            self.btn_wrong_pred.configure(state='normal')
            
    def train_new_sample(self):    
        # Enable the Button and Entry Field for Class Name if desired
        self.entry_class_name.configure(state='normal')
        self.btn_class_name.configure(state='normal')
        # Wait for Class Name to be submitted via Button
        self.btn_class_name.wait_variable(self.class_name)
        # Disable Button and Entry Field for Class Name
        self.entry_class_name.configure(state='disabled')            
        self.btn_class_name.configure(state='disabled')
                    
        label = []
        tmp_class_name = self.class_name.get()
        if tmp_class_name in self.class_name_list:
            label.append(self.class_name_list.index(tmp_class_name))
        else:
            label.append(len(np.unique(self.class_name_list)))
            self.class_name_list.append(tmp_class_name)
        self.modulB.train(self.features_np_normed, np.asarray(label))
        
    def consolidate(self):
        # Consolidate Module B
        self.modulB.consolidation()
                    
    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
