import cv2
import depthai as dai
import numpy as np
import time
import blobconverter
import matplotlib.pyplot as plt
import pyaudio
import mouse
import pygame
import wave
from tkinter import * # Detection de la souris
from pydub.playback import play
from pydub import AudioSegment
from datetime import timedelta

duration = 5  # en secondes
sample_rate = 44100  # Hz
volume_increase_rate = 0.02  # Taux d'augmentation du volume par échantillon

def faire_cercle_horiz(img):
    # Define the dimensions of the image
    verti = img.shape[0]
    horiz = img.shape[1]

    # Create a black image with the specified dimensions
    mask = np.zeros((verti, horiz, 3), np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #Conversion de l'image en nuance de gris

    # Colonne 10 Ligne 10
    h_circle = int(horiz/20)
    v_circle = int(verti/12)
    x_centre = int(h_circle/2)
    y_centre = int(v_circle/2)
    rayon = int(x_centre*0.8)
    x = x_centre
    y = y_centre

    while y < verti:
        while x < horiz:
            # Create white circle
            cv2.circle(mask, (x,y), rayon, (255,255,255),-1)

            x += h_circle
        x = x_centre
        y += v_circle
    
    visionhori = np.copy(mask)
    visionhori[:90,:610] = 0
    visionhori[260:,:610] = 0
    mask = cv2.GaussianBlur(mask, (9, 9), 2)
    visionhori = cv2.GaussianBlur(visionhori, (9, 9), 2)
    return visionhori

def generate_stereo_sound(duration, sample_rate, frequencies):
    # Calcul de la durée de chaque fréquence
    tres_loin = 50
    loin = 100
    proche = 200
    
    frequency_duration = duration / len(frequencies)

    # Initialisation des canaux gauche et droit
    left_channel = np.array([], dtype=np.float32)
    right_channel = np.array([], dtype=np.float32)
    # Seuil des niveau de gris pour définir la fréquence à émettre
    for nuance in frequencies:
        if nuance <= tres_loin:
            frequency = 200
        elif nuance > tres_loin and nuance <= loin:
            frequency = 500
        elif nuance > loin and nuance < proche:
            frequency = 800
        else:
            frequency = 1100
        
        # Paramètres du signal pour la fréquence actuelle
        t = np.linspace(0, frequency_duration, int(sample_rate * frequency_duration), endpoint=False)
        current_left_channel = np.sin(2 * np.pi * frequency * t)
        current_right_channel = np.sin(2 * np.pi * frequency * t + np.pi/2)  # Décalage de phase de pi/2

        # Concaténation des canaux actuels avec les canaux existants
        left_channel = np.concatenate((left_channel, current_left_channel))
        right_channel = np.concatenate((right_channel, current_right_channel))
    # Augmentation progressive du volume de gauche à droite
    T = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    volume = np.linspace(0, 1, len(T))
    right_channel *= volume
    left_channel *= volume[::-1]  # Inverser l'ordre pour l'augmentation de droite à gauche

    # Normaliser les canaux
    right_channel /= np.max(np.abs(right_channel))
    left_channel /= np.max(np.abs(left_channel))

    # Créer le tableau de données audio en format stéréo
    stereo_sound = np.column_stack((left_channel, right_channel))

    # Enregistrement du son
    write_wave_file(stereo_sound, sample_rate)

def write_wave_file(data, sample_rate, filename='stereo.wav'):
    # Enregistrement du fichier wave
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(2)  # Stéréo
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes((data * 32767).astype(np.int16).tobytes())

def traitement_image(img): 
    img = cv2.resize(img, dsize=(600, 350), interpolation=cv2.INTER_CUBIC)
    circle = faire_cercle_horiz(img)
    # Détection des cercles
    cercleshori = cv2.HoughCircles(circle, cv2.HOUGH_GRADIENT, dp=1, minDist=10,param1=25, param2=15, minRadius=5, maxRadius=15)

    grishori=[]
    liste=[]
    i=0

    if cercleshori is not None:
        cercleshori = np.uint16(np.around(cercleshori))

        # Afficher les coordonnées des cercles détectés
        for i, cercle in enumerate(cercleshori[0, :]):
            x, y, rayon = cercle[0], cercle[1], cercle[2]
            valeur_gris = img[y - rayon:y + rayon, x - rayon:x + rayon]
            valeur_gris = int(np.mean(valeur_gris)) if not np.isnan(np.mean(valeur_gris)) else 0
            grishori.append(valeur_gris)
            
            
            # Colorier le cercle avec la valeur moyenne de gris
            cv2.circle(circle, (x, y), rayon, (valeur_gris, valeur_gris, valeur_gris), -1)
            liste.append([x,y,valeur_gris])
            cv2.imwrite('visionhori.png', circle)

            
    else:
        print("Aucun cercle détecté.")

    liste= sorted(liste, key=lambda x: x[0])
    actuel=[]
    ysignal=[]
    xsignal=[]
    for i in range (0,len(liste)):
        if len(actuel)<6:
            actuel.append(liste[i][2])
        if len(actuel)==6:
            #print(actuel)
            xsignal.append(liste[i][0])
            ysignal.append( max(actuel))
            actuel=[]

    inc = int(len(ysignal)/5)
    signal_final = []
    for i in range(0,len(ysignal),inc):
        signal_final.append(np.max([ysignal[i],ysignal[i+1],ysignal[i+2],ysignal[i+3]]))

    generate_stereo_sound(duration, sample_rate, signal_final)

    pygame.init()
    fstereo= 'stereo.wav' # Emplacement du fichier audio créer
    sound = pygame.mixer.Sound(fstereo) #tr
    channel = pygame.mixer.find_channel()
    channel.play(sound)



# Custom JET colormap with 0 mapped to `black` - better disparity visualization
jet_custom = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_custom[0] = [0, 0, 0]

blob = dai.OpenVINO.Blob(blobconverter.from_zoo(name="deeplab_v3_mnv2_256x256", zoo_type="depthai", shaves=6))
# for name,tensorInfo in blob.networkInputs.items(): print(name, tensorInfo.dims)
INPUT_SHAPE = blob.networkInputs['Input'].dims[:2]
TARGET_SHAPE = (400,400)

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def get_multiplier(output_tensor):
    class_binary = [[0], [1]]
    class_binary = np.asarray(class_binary, dtype=np.uint8)
    output = output_tensor.reshape(*INPUT_SHAPE)
    output_colors = np.take(class_binary, output, axis=0)
    return output_colors

class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0
    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1
    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

class HostSync:
    def __init__(self):
        self.arrays = {}
    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({'msg': msg})
        # Try finding synced msgs
        ts = msg.getTimestamp()
        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                time_diff = abs(obj['msg'].getTimestamp() - ts)
                # 20ms since we add rgb/depth frames at 30FPS => 33ms. If
                # time difference is below 20ms, it's considered as synced
                if time_diff < timedelta(milliseconds=33):
                    synced[name] = obj['msg']
                    # print(f"{name}: {i}/{len(arr)}")
                    break
        # If there are 3 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 3: # color, depth, nn
            def remove(t1, t2):
                return timedelta(milliseconds=500) < abs(t1 - t2)
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if remove(obj['msg'].getTimestamp(), ts):
                        arr.remove(obj)
                    else: break
            return synced
        return False

def crop_to_square(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]

# Start defining a pipeline
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# Color cam: 1920x1080
# Mono cam: 640x400
cam.setIspScale(2,3) # To match 400P mono cameras
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# For deeplabv3
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setPreviewSize(*INPUT_SHAPE)
cam.setInterleaved(False)

# NN output linked to XLinkOut
isp_xout = pipeline.create(dai.node.XLinkOut)
isp_xout.setStreamName("cam")
cam.isp.link(isp_xout.input)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlob(blob)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)
cam.preview.link(detection_nn.input)

# NN output linked to XLinkOut
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Left mono camera
left = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
# Right mono camera
right = pipeline.create(dai.node.MonoCamera)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
left.out.link(stereo.left)
right.out.link(stereo.right)

# Create depth output
xout_disp = pipeline.create(dai.node.XLinkOut)
xout_disp.setStreamName("disparity")
stereo.disparity.link(xout_disp.input)

# Pipeline is defined, now we can connect to the device
with dai.Device() as device:
    cams = device.getConnectedCameras()
    depth_enabled = dai.CameraBoardSocket.LEFT in cams and dai.CameraBoardSocket.RIGHT in cams
    if not depth_enabled:
        raise RuntimeError("Unable to run this experiment on device without depth capabilities! (Available cameras: {})".format(cams))
    device.startPipeline(pipeline)
    # Output queues will be used to get the outputs from the device
    q_color = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_disp = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    fps = FPSHandler()
    sync = HostSync()
    disp_frame = None
    disp_multiplier = 255 / stereo.initialConfig.getMaxDisparity()

    frame = None
    depth = None
    depth_weighted = None
    frames = {}

    while True:
        msgs = False
        if q_color.has():
            msgs = msgs or sync.add_msg("color", q_color.get())
        if q_disp.has():
            msgs = msgs or sync.add_msg("depth", q_disp.get())
        if q_nn.has():
            msgs = msgs or sync.add_msg("nn", q_nn.get())

        if msgs:
            fps.next_iter()
            # get layer1 data
            layer1 = msgs['nn'].getFirstLayerInt32()
            # reshape to numpy array
            lay1 = np.asarray(layer1, dtype=np.int32).reshape(*INPUT_SHAPE)
            output_colors = decode_deeplabv3p(lay1)

            # To match depth frames
            output_colors = cv2.resize(output_colors, TARGET_SHAPE)

            frame = msgs["color"].getCvFrame()
            frame = crop_to_square(frame)
            frame = cv2.resize(frame, TARGET_SHAPE)
            frames['frame'] = frame
            frame = cv2.addWeighted(frame, 1, output_colors,0.5,0)
            cv2.putText(frame, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
            frames['colored_frame'] = frame

            disp_frame = msgs["depth"].getFrame()
            disp_frame = (disp_frame * disp_multiplier).astype(np.uint8)
            disp_frame = crop_to_square(disp_frame)
            disp_frame = cv2.resize(disp_frame, TARGET_SHAPE)

            # Colorize the disparity
            frames['depth'] = cv2.applyColorMap(disp_frame, jet_custom)

            multiplier = get_multiplier(lay1)
            multiplier = cv2.resize(multiplier, TARGET_SHAPE)
            depth_overlay = disp_frame * multiplier
            frames['cutout'] = cv2.applyColorMap(depth_overlay, jet_custom)
            # You can add custom code here, for example depth averaging

            if len(frames) == 4:
                show = np.concatenate((frames['colored_frame'], frames['cutout'], frames['depth']), axis=1)
                cv2.imshow( "depth", disp_frame)

            if cv2.waitKey(1) == ord('q'):
                exit()

            if mouse.is_pressed("left"):
                traitement_image(disp_frame)
