#------------------------------IMPORTANCION DE LIBRERIAS---------------------------------------------#
import tkinter as tk
from tkinter import ttk, Canvas
from PIL import Image, ImageTk
import dlib
import cv2
import os
from test1 import test
import time
import random
import socket
import threading
import queue
import pygame
from util import load_embeddings, recognize_with_embeddings
from datetime import datetime, timedelta
from mongo_connection_1 import *
from collections import deque
import numpy as np
import mediapipe as mp
import subprocess
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pymongo.errors import PyMongoError
#---------------------------------------------------------------------------------------#


# def launch_external_script():
#     try:
#         # Ejecutar el script externo de manera asíncrona usando Popen.
#         external_process = subprocess.Popen(["python", "BioDataRetrieverV2.py"])
#         print("Script externo lanzado de manera asíncrona.")
#     except Exception as e:
#         print(f"Error al lanzar el script externo: {e}")

# # Llamar a la función en el momento adecuado (por ejemplo, al inicio del programa)
# launch_external_script()

external_process = None  # Inicializa la variable 'external_process' con None. 
#Esta variable se utilizará para almacenar un proceso externo que se inicia desde el
#  programa principal. Un ejemplo de esto podría ser un programa de verificación de huellas
#  dactilares que se ejecuta como un proceso separado. Almacenar el proceso permite al programa
#  principal interactuar con él, como terminar el proceso cuando ya no se necesita.

# Obtener la ruta absoluta al archivo del predictor
current_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_dir, "utils", "shape_predictor_68_face_landmarks.dat")

# Inicializar el detector de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# Inicializar el rastreador de OpenCV
tracker = cv2.TrackerCSRT_create()  #TRACKER

# ----------------------- CONSTANTES DE COLORES ----------------------- #
COLOR_RED = (0, 0, 255)  # Define el color rojo en formato BGR (Blue, Green, Red) utilizado por
#OpenCV. (0, 0, 255) representa el color rojo.
COLOR_YELLOW = (0, 255, 255)  # Define el color amarillo en formato BGR utilizado por OpenCV.
# (0, 255, 255) representa el color amarillo.
COLOR_GREEN = (0, 255, 0)  # Define el color verde en formato BGR utilizado por OpenCV. 
#(0, 255, 0) representa el color verde.

#--------------------------------- FUNCIONES AUXILIARES ---------------------------------#

def draw_feedback(frame, color, face_detected):
    """
    Dibuja un círculo con efecto 3D en la esquina superior izquierda del frame según el color proporcionado.

    Args:
        frame (numpy.ndarray): El frame de la imagen donde se dibujará el círculo.
        color (tuple): Una tupla que representa el color del círculo en formato BGR (Blue, Green, Red).

    Returns:
        numpy.ndarray: El frame de la imagen con el círculo dibujado.

    Detalles:
        - La función dibuja un círculo con un efecto 3D en la esquina superior izquierda del frame.
        - Primero, se dibuja una sombra para dar el efecto 3D.
        - Luego, se dibuja el círculo principal con el color proporcionado.
        - Finalmente, se dibuja un borde blanco alrededor del círculo para mejorar el efecto visual.
    """
    if face_detected:
        indicator_center_x = 50
        indicator_center_y = 50
        circle_radius = 20

        # Dibuja la sombra del círculo
        cv2.circle(frame, (indicator_center_x + 2, indicator_center_y + 2), circle_radius, (50, 50, 50), -1, cv2.LINE_AA)

        # Dibuja el círculo principal
        cv2.circle(frame, (indicator_center_x, indicator_center_y), circle_radius, color, -1, cv2.LINE_AA)

        # Dibuja el borde del círculo
        cv2.circle(frame, (indicator_center_x, indicator_center_y), circle_radius, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def is_face_in_center(face, frame_shape, margin_ratio_width=0.5, margin_ratio_height=0.5):
    """
    Verifica si una cara está centrada dentro de un margen especificado en el frame.

    Args:
        face (dlib.rectangle): Un rectángulo que delimita la posición de la cara detectada en el frame.
        frame_shape (tuple): Las dimensiones del frame en formato (altura, anchura).
        margin_ratio_width (float): El ratio del margen de la anchura respecto al tamaño del frame. Por defecto es 0.5.
        margin_ratio_height (float): El ratio del margen de la altura respecto al tamaño del frame. Por defecto es 0.5.

    Returns:
        bool: True si la cara está dentro del margen central especificado, False en caso contrario.

    Detalles:
        - La función calcula los márgenes centralizados del frame en base a los ratios proporcionados.
        - Se verifica si todas las coordenadas de la cara detectada están dentro de estos márgenes centralizados.
    """
    frame_height, frame_width = frame_shape[:2]
    margin_width = int(frame_width * margin_ratio_width)
    margin_height = int(frame_height * margin_ratio_height)
    
    center_x1 = (frame_width // 2) - (margin_width // 2)
    center_y1 = (frame_height // 2) - (margin_height // 2)
    center_x2 = center_x1 + margin_width
    center_y2 = center_y1 + margin_height
    
    face_x1, face_y1, face_x2, face_y2 = face.left(), face.top(), face.right(), face.bottom()
    
    return (center_x1 < face_x1 < center_x2 and center_x1 < face_x2 < center_x2 and
            center_y1 < face_y1 < center_y2 and center_y1 < face_y2 < center_y2)


def draw_center_area(frame, margin_width_ratio=0.5, color=(255, 255, 255), thickness=1):
    """
    Dibuja líneas verticales semitransparentes en el área central del frame para indicar la región de interés.

    Args:
        frame (numpy.ndarray): El frame de la imagen donde se dibujarán las líneas.
        margin_width_ratio (float): El ratio del margen de la anchura respecto al tamaño del frame. Por defecto es 0.5.
        color (tuple): Una tupla que representa el color de las líneas en formato BGR (Blue, Green, Red). Por defecto es blanco (255, 255, 255).
        thickness (int): El grosor de las líneas. Por defecto es 1.

    Returns:
        numpy.ndarray: El frame de la imagen con las líneas semitransparentes dibujadas.

    Detalles:
        - La función calcula las coordenadas del área central en el frame en base al ratio de margen proporcionado.
        - Dibuja dos líneas verticales semitransparentes a los lados del área central para indicar la región de interés.
        - Utiliza una copia del frame original para crear una superposición semitransparente.
    """
    frame_height, frame_width = frame.shape[:2]
    margin_width = int(frame_width * margin_width_ratio)

    center_x1 = (frame_width // 2) - (margin_width // 2)
    center_x2 = center_x1 + margin_width

    # Draw semi-transparent lines
    overlay = frame.copy()
    cv2.line(overlay, (center_x1, 0), (center_x1, frame_height), color, thickness)
    cv2.line(overlay, (center_x2, 0), (center_x2, frame_height), color, thickness)
    alpha = 0.1  # Adjust transparency here (0 is fully transparent, 1 is fully opaque)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame
def ajustar_brillo(img, face):
    try:
        # Extraer coordenadas del rostro y asegurar que estén dentro de los límites de la imagen
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y = max(0, x), max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        # Región del rostro y conversión a escala de grises para calcular el brillo promedio
        rostro = img[y:y+h, x:x+w]
        rostro_gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
        brillo_promedio = np.mean(rostro_gray)

        # Parámetros de ajuste
        UMBRAL_BAJO = 65
        UMBRAL_ALTO = 150
        FACTOR_BRILLO_BAJO = 1.2
        FACTOR_BRILLO_ALTO = 0.8
        BETA_BRILLO_BAJO = 30
        BETA_BRILLO_ALTO = -40

        # Ajuste de brillo según el brillo promedio
        if brillo_promedio < UMBRAL_BAJO:
            alpha = FACTOR_BRILLO_BAJO
            beta = BETA_BRILLO_BAJO
        elif brillo_promedio > UMBRAL_ALTO:
            # Si el brillo es alto, convertir toda la imagen a escala de grises
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.convertScaleAbs(img_gray, alpha=FACTOR_BRILLO_ALTO, beta=BETA_BRILLO_ALTO)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # Convertir de vuelta a color (3 canales)
            alpha = 1.0
            beta = 0
        else:
            alpha = 1.0
            beta = 0

        # Aplicar ajuste de brillo en la imagen completa y limitar los valores
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = np.clip(img, 0, 255)

        # Aplicar filtro bilateral para reducir ruido sin distorsionar bordes
        img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)

        return img
    except Exception as e:
        print(f"Error al ajustar el brillo: {e}")
        return img

"""_create_gradient_
    Crea un gradiente horizontal en un lienzo dado, utilizando los colores RGB de inicio y fin especificados.

    Args:
        canvas (tkinter.Canvas): El lienzo donde se dibujará el gradiente.
        start_color (str): El color de inicio del gradiente en formato hexadecimal (por ejemplo, '#RRGGBB').
        end_color (str): El color de fin del gradiente en formato hexadecimal (por ejemplo, '#RRGGBB').
        width (int): El ancho del lienzo sobre el cual se aplicará el gradiente.

    Returns:
        None: Esta función no retorna ningún valor, modifica el lienzo directamente.

    Raises:
        Exception: Cualquier excepción que ocurra durante la creación del gradiente.
    """
def create_gradient(canvas, start_color, end_color, width):
    """ Create a horizontal gradient with the given start and end RGB colors """
    (r1, g1, b1) = canvas.winfo_rgb(start_color)
    (r2, g2, b2) = canvas.winfo_rgb(end_color)
    
    r_ratio = float(r2 - r1) / width
    g_ratio = float(g2 - g1) / width
    b_ratio = float(b2 - b1) / width

    for i in range(width):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = "#%4.4x%4.4x%4.4x" % (nr, ng, nb)
        canvas.create_line(i, 0, i, 10, tags=("gradient",), fill=color)

    canvas.lower("gradient")  


def load_resized_image(path, size):
   
    image = Image.open(path)
    image = image.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(image)

def update_time(time_label, root):
    current_time = datetime.now().strftime('%I:%M:%S %p')
    time_label.config(text=current_time)
    root.after(1000, update_time, time_label, root)
    
dias_espanol = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo"
}


def update_date(date_label):
    day_of_week = datetime.now().strftime("%A")
    date_str = datetime.now().strftime("%d/%m/%Y").upper()
    day_of_week_es = dias_espanol[day_of_week]
    current_date = f"{day_of_week_es}\n\n{date_str}"
    date_label.config(text=current_date)

"""
Funciones de Reproducción de Sonidos

Este módulo contiene funciones para reproducir diferentes sonidos asociados con eventos específicos en una aplicación. 
Cada función utiliza el módulo pygame.mixer para cargar y reproducir un archivo de audio específico.

Funciones:

- play_success_sound(): Reproduce el sonido de éxito.
- play_error_sound(): Reproduce el sonido de error.
- play_notification_sound(): Reproduce el sonido de notificación.
- play_normal_sound(): Reproduce el sonido de entrada normal.
- play_falta_sound(): Reproduce el sonido de falta.
- play_nota_mala_sound(): Reproduce el sonido de nota mala.
- play_retardo_sound(): Reproduce el sonido de retardo.
- play_error_escaneo(): Reproduce el sonido de error de escaneo.
- play_ya_scaneado(): Reproduce el sonido de "ya escaneado".
- play_sa_normal(): Reproduce el sonido de salida normal.
- play_sa_retardo(): Reproduce el sonido de salida con retardo.
- play_sa_notamala(): Reproduce el sonido de salida con nota mala.
- play_sa_falta(): Reproduce el sonido de salida con falta.

Importante:
Asegúrese de que los archivos de audio estén en la ruta correcta 
especificada en cada función. Ya que a veces eso produce que el sistema
no funcione
"""

# sonidos
def play_success_sound():
        success_sound = pygame.mixer.Sound('RECURSOS/audios/correcto.wav')  
        success_sound.play()

   
def play_error_sound():
        error_sound = pygame.mixer.Sound('RECURSOS/audios/Error.wav')  
        error_sound.play()
        
def play_notification_sound():
    notification_sound = pygame.mixer.Sound('RECURSOS/audios/correcto.wav')
    notification_sound.play()
    return notification_sound


# sonidos para los estatus de entrada y salida
def  play_normal_sound():
    normal_sound = pygame.mixer.Sound('RECURSOS/audios/normal.wav')
    normal_sound.play()
    
def play_falta_sound(): 
    falta_sound = pygame.mixer.Sound('RECURSOS/audios/falta.wav')
    falta_sound.play()
    
def play_nota_mala_sound():
    nota_mala_sound = pygame.mixer.Sound('RECURSOS/audios/nota_mala.wav')
    nota_mala_sound.play()

def play_retardo_sound():
    retardo_sound = pygame.mixer.Sound('RECURSOS/audios/retardo.wav')
    retardo_sound.play()

def play_error_escaneo():
    error_escaneo = pygame.mixer.Sound('RECURSOS/audios/error_escanear.wav')
    error_escaneo.play()

def play_ya_scaneado():
    ya_scaneado= pygame.mixer.Sound('RECURSOS/audios/Ya_escaneado.wav')
    ya_scaneado.play()

### SALIDAS ###
def play_sa_normal():
    salida_normal= pygame.mixer.Sound('RECURSOS/audios/SALIDA_NORMAL-.wav')
    salida_normal.play()

def play_sa_retardo():
    salida_retado= pygame.mixer.Sound('RECURSOS/audios/SALIDA_CON_RETARDO.wav')
    salida_retado.play()

def play_sa_notamala():
    salida_notamala= pygame.mixer.Sound('RECURSOS/audios/SALIDA_CON_NOTA_MALA.wav')
    salida_notamala.play()

def play_sa_falta():
    salida_falta= pygame.mixer.Sound('RECURSOS/audios/SALIDA_CON_FALTA.wav')
    salida_falta.play()




# CLASE PRINCIPAL
class App:
    def __init__(self, root, parent_frame, section2_frame, section4_frame):
        self.root = root
        self.detector = dlib.get_frontal_face_detector()
        self.tracker = dlib.correlation_tracker()
        self.tracking_face = False

        self.main_frame = parent_frame
        self.webcam_label = tk.Label(self.main_frame)
        self.webcam_label.grid(row=0, column=0, sticky='nswe')
        self.add_webcam(self.webcam_label)
        self.section2_frame = section2_frame
        self.section4_frame = section4_frame
        self.db = get_db()
        
        self.db = get_db()
        if self.db is None:
            # Mostrar mensaje en la sección 2 si no hay conexión
            self.update_section2()  # Cambiamos esto para usar update_section2 en lugar de show_db_error_message
        else:
            self.update_section2()
        
        self.update_section2()
        self.message_queue = queue.Queue()
        self.result_queue = queue.Queue()  
        self.start_socket_server()
        self.metodo_verificacion = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock() 
        self.spoofing_warning_count = 0   



        resized_image1 = load_resized_image('RECURSOS/imagenes/H.png', (90, 100))
        self.image_label1 = tk.Label(self.section4_frame, image=resized_image1, bg='#D3D3D3')
        self.image_label1.grid(row=0, column=0, sticky='nswe')
        self.image_label1.image = resized_image1  

        resized_image2 = load_resized_image('RECURSOS/imagenes/R.png', (90, 90))
        self.image_label2 = tk.Label(self.section4_frame, image=resized_image2, bg='#D3D3D3')
        self.image_label2.grid(row=0, column=2, sticky='nswe')
        self.image_label2.image = resized_image2  

        self.overlay_message_text = None
        self.positioning_label = None
        self.clean_frame = None
        
        self.db_dir = './db_1'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)


          # Cargar las incrustaciones una vez al inicio
        print(f"Directorio de incrustaciones: {self.db_dir}")
        self.embeddings_dict = load_embeddings(self.db_dir)
        print(f"Incrustaciones cargadas: {len(self.embeddings_dict)} archivos") 


        self.log_path = './log.txt'
        self.most_recent_capture_arr = None

        self.last_capture_time = 0 
        self.face_detected_time = None  

        self.capturing = False  # Estado del cronómetro

        
       
        """_initialize_attributes_

            Inicializa los atributos necesarios para el funcionamiento de la clase.
            Propósito:
            Configurar e inicializar los parámetros y estados necesarios para las 
            funcionalidades de la clase, tales como la detección facial, el escaneo 
            y el manejo de mensajes.
        """
    def initialize_attributes(self):
            self.is_warning_message_active = False
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh()
            self.scan_position = 0
            self.scan_direction = 1
            self.scan_speed = 9  # Velocidad del escaneo en píxeles por frame
            self.line_thickness = 2
            self.direction = 1
            self.fade_speed = 0.05  # Velocidad de desvanecimiento
            self.line_color = (0, 255, 0)  # Verde intenso
            self.recognition_error_count=0
            self.error_count = 0 
            self.scan_error_count= 0
            self.message_active = False 
            self.tracker = cv2.TrackerKCF_create()  # Usar TrackerKCF para un buen equilibrio entre velocidad y precisión
            self.tracking_face = False  # Para controlar si se está rastreando una cara
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.clean_frame = None
            



    """_check_initialized_
        Verifica si los atributos necesarios están inicializados y, si no lo están, los inicializa.

        Propósito:
        Garantizar que los atributos esenciales de la clase estén configurados antes de su uso. 
        Si los atributos no están presentes, los inicializa llamando a `initialize_attributes()`.

        Funcionalidad:
        - Verifica la existencia del atributo 'is_warning_message_active'.
        - Si el atributo no existe, llama a `initialize_attributes()` para inicializar todos los atributos necesarios.
        - Imprime un mensaje de depuración indicando que los atributos no estaban inicializados y que se han inicializado.

    """
    def check_initialized(self):
            if not hasattr(self, 'is_warning_message_active'):
                self.initialize_attributes()
                print("[DEBUG] Attributes were not initialized. Initializing now.")

  
   # ------------------------------------ PARA AVISOS TRABAJADOR 2 MODULOS ------------------------------------ #
    """procesar_rfc_
        Procesa un RFC buscando información relacionada en la base de datos.

        Args:
            rfc (str): El RFC que se va a procesar.

        Funcionalidad:
        - Imprime el RFC a procesar.
        - Busca información relacionada en la colección 'avisos' de la base de datos.
        - Imprime la información encontrada o un mensaje si no se encuentra información.

        Returns:
            None
        """
    def procesar_rfc(self, rfc):
        try:
            # Aquí va la lógica para procesar el RFC
            print(f"Procesando RFC: {rfc}")

            # Verificar la conexión a la base de datos
            if not self.db:
                print("Conexión a la base de datos no establecida.")
                return

            # Verificar que la colección 'avisos' existe
            if 'avisos' not in self.db.list_collection_names():
                print("La colección 'avisos' no existe en la base de datos.")
                return

            # Realizar la búsqueda del RFC en la colección 'avisos'
            info_rfc = self.db.get_collection('avisos').find_one({'RFC': rfc})
            if info_rfc:
                print("Información encontrada:", info_rfc)
            else:
                print("No se encontró información para el RFC proporcionado.")

        except PyMongoError as e:
            print(f"Error al acceder a la base de datos: {str(e)}")
        except Exception as e:
            print(f"Error inesperado: {str(e)}")


    """_ register_facial_entry_
        Registra una entrada de reconocimiento facial en un archivo de log y muestra un mensaje de administrador si existe.

        Args:
            name (str): Nombre o RFC de la persona registrada.
            entry_type (str): Tipo de entrada, como 'Entrada' o 'Salida'.
            success (bool): Indica si la entrada fue exitosa o fallida.

        Funcionalidad:
        - Registra la entrada con un timestamp, un ID único, el método de reconocimiento facial, el nombre, el tipo de entrada y el resultado en un archivo de log.
        - Verifica si el RFC tiene un mensaje de administrador asociado en la base de datos.
        - Si existe un mensaje de administrador, reproduce un sonido de notificación y muestra el mensaje.

        Returns:
            None
    """ 
    def register_facial_entry(self, name, entry_type, success):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = "exitoso" if success else "fallido"
       # entry_id = str(uuid.uuid4())
        log_entry = f"{timestamp} -  Método: Reconocimiento Facial, Nombre: {name}, Tipo: {entry_type}, Resultado: {result}\n"

        with open(self.log_path, 'a') as f:
            f.write(log_entry)
        
         # Comprobar si el nombre (que es el RFC) tiene un mensaje de administrador asociado
        admin_message = get_admin_message_by_rfc(self.db, name)
        if admin_message and admin_message != "No se encontró el RFC en la base de datos." and not admin_message.startswith("Error"):
            
            notification_sound = play_notification_sound()
            pygame.time.wait(int(notification_sound.get_length() * 10))
            
            self.show_admin_message(admin_message)

    
    
    def show_warning_message(self, message, duration=7, color='gray'):
        """
        Muestra un mensaje de advertencia en una ventana emergente.
        
        Args:
            message (str): Mensaje a mostrar
            duration (int): Duración en segundos
            color (str): Color del fondo
        """
        self.is_warning_message_active = True
        top = tk.Toplevel(self.main_frame)

        # Centrar dinámicamente
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        window_width = 500
        window_height = 200
        top_x = (screen_width - window_width) // 2
        top_y = (screen_height - window_height) // 2
        top.geometry(f"{window_width}x{window_height}+{top_x}+{top_y}")

        top.grab_set()
        top.focus_force()
        top.attributes('-topmost', True)
        top.overrideredirect(True)
        top.configure(bg=color, padx=0, pady=0)

        frame = tk.Frame(top, bg='light gray', padx=10, pady=10, relief='raised', bd=5)
        frame.pack(expand=True, fill='both')

        # Panel superior
        top_panel = tk.Canvas(frame, bg='#2C2C2C', height=40)
        top_panel.pack(side='top', fill='x')

        # Botones simulados
        buttons = [('red', 10), ('yellow', 40), ('green', 70)]
        for color, x in buttons:
            top_panel.create_oval(x, 10, x+20, 30, fill=color, outline=color)

        aviso_label = top_panel.create_text(250, 20, 
                                        text='AVISO IMPORTANTE', 
                                        font=('Arial', 16, 'bold'), 
                                        fill='light gray')

        msg_label = tk.Label(frame, text=message, 
                            font=('Arial', 18), 
                            wraplength=350,
                            bg='light gray',
                            fg='black',
                            bd=0)
        msg_label.pack(pady=(20, 10), padx=20, expand=True)

        countdown_label = tk.Label(frame, text="", 
                                font=('Arial', 12),
                                bg='light gray', 
                                fg='black')
        countdown_label.pack(side='bottom', anchor='e', padx=10, pady=(0, 10))

        def update_countdown(seconds_left):
            if seconds_left > 0:
                countdown_label.config(text=f"Cerrando en {seconds_left} s")
                self.root.after(1000, update_countdown, seconds_left - 1)
            else:
                self.close_message(top)

        def blink(count=0):
            if count < 4:
                current_color = frame.cget("bg")
                next_color = 'lightcoral' if current_color == 'light gray' else 'light gray'
                frame.config(bg=next_color)
                msg_label.config(bg=next_color)
                countdown_label.config(bg=next_color)
                new_text_color = 'red' if current_color == 'light gray' else 'light gray'
                top_panel.itemconfig(aviso_label, fill=new_text_color)
                self.root.after(500, blink, count + 1)

        update_countdown(duration)
        blink()

        # Iniciar el cronómetro con 7 segundos
        update_countdown(7)

        # Iniciar el parpadeo
        blink()

    def close_message(self, top):
        """
        Cierra la ventana emergente con una animación de desvanecimiento.
        """
        def fade_out():
            self.is_warning_message_active = False
            alpha = top.attributes("-alpha")
            if alpha > 0:
                alpha -= 0.1
                top.attributes("-alpha", alpha)
                top.after(50, fade_out)
            else:
                
                top.destroy()

        top.attributes("-alpha", 1)
        fade_out()
    
      # ------------------------------------ TERMINA AVISOS TRABAJADOR ------------------------------------ #
    def show_admin_message(self, message):
         self.show_warning_message(message)


    # ------------------------------------ PARA AVISOS GENERALES 1 MODULO -------------------------------- #


    def register_fingerprint_entry(self, name, entry_type, success):
        # Método para registrar la entrada por reconocimiento de huella dactilar
        with open(self.log_path, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result = "exitoso" if success else "fallido"
          #  entry_id = str(uuid.uuid4())

            log_entry = f"{timestamp} - Método: Huella Dactilar, Nombre: {name}, Tipo: {entry_type}, Resultado: {result}\n"
            f.write(log_entry)
            
            with open(self.log_path, 'a') as f:
                f.write(log_entry)
        
         # Comprobar si el nombre (que es el RFC) tiene un mensaje de administrador asociado
        admin_message = get_admin_message_by_rfc(self.db, name)
        if admin_message and admin_message != "No se encontró el RFC en la base de datos." and not admin_message.startswith("Error"):
            notification_sound = play_notification_sound()
            pygame.time.wait(int(notification_sound.get_length() * 10))
            
            self.show_admin_message(admin_message)
            
    """_ update_section2_
        Actualiza la información mostrada en la sección 2 del marco de la interfaz.

        Funcionalidad:
        - Verifica si hay un mensaje de error activo en la sección 2 y, si es así, no realiza ninguna actualización para evitar sobrescribirlo.
        - Obtiene la información del campo 'Avisogeneral' desde la base de datos.
        - Si no se encuentra información, muestra un mensaje indicando que no hay mensaje disponible.
        - Limpia el contenido actual de la sección 2 y actualiza con la nueva información obtenida.
        - En caso de errores específicos, como falta de campo en la base de datos, muestra un mensaje de error específico.
        - Maneja cualquier otro error genérico mostrando un mensaje de error detallado.

        Excepciones:
        - KeyError: Si el campo requerido no está disponible en la base de datos.
        - Exception: Para manejar cualquier otro error que pueda surgir durante la actualización.

        Returns:
            None
    """
    def update_section2(self):
        try:
            if self.db is None:
                info = "Base de datos desconectada.\nPor favor, revise la conexión y reinicie la aplicación."
                bg_color = '#FFEBEE'
                text_color = 'red'
            else:
                if any(isinstance(widget, tk.Label) and 
                    widget.cget("text").startswith("Error mostrando el mensaje") 
                    for widget in self.section2_frame.winfo_children()):
                    return

                info = get_info(self.db, 'Avisogeneral') or "No hay mensaje disponible."
                bg_color = '#EFEFEF'
                text_color = 'black'

            for widget in self.section2_frame.winfo_children():
                widget.destroy()

            container = tk.Frame(self.section2_frame, bg=bg_color, width=600, height=380)
            container.pack_propagate(False)
            container.pack(expand=True, fill='both')

            # Contar líneas y caracteres aproximados por línea
            approx_chars_per_line = 50  # Ajustar según necesidad
            text_lines = info.split('\n')
            num_lines = sum(len(line) // approx_chars_per_line + 1 for line in text_lines)

            if num_lines > 4:
                font_size = 16
                text_align = 'left'  
            elif num_lines > 3:
                font_size = 18
                text_align = 'left'
            else:
                font_size = 20
                text_align = 'center'

            info_label = tk.Label(container,
                                text=info,
                                bg=bg_color,
                                fg=text_color,
                                anchor='center',
                                justify=text_align,
                                font=('Roboto', font_size),
                                wraplength=550)
            info_label.place(relx=0.5, rely=0.5, anchor='center')

        except Exception as e:
            for widget in self.section2_frame.winfo_children():
                widget.destroy()

            container = tk.Frame(self.section2_frame, bg='#FFEBEE', width=600, height=380)
            container.pack_propagate(False)
            container.pack(expand=True, fill='both')

            error_label = tk.Label(container,
                                text=f"Error al obtener la información:\n{str(e)}",
                                bg='#FFEBEE',
                                fg='red',
                                anchor='center',
                                justify='center',
                                font=('Roboto', 20),
                                wraplength=550)
            error_label.place(relx=0.5, rely=0.5, anchor='center')

    def show_db_error_message(self):
        # Limpiar la sección 2
        for widget in self.section2_frame.winfo_children():
            widget.destroy()
        
        # Crear un frame contenedor con fondo blanco
        container = tk.Frame(self.section2_frame, bg='white', width=600, height=380)
        container.pack_propagate(False)
        container.pack(expand=True, fill='both')
        
        # Mostrar mensaje de error
        error_label = tk.Label(
            container,
            text="Base de datos desconectada.\nPor favor, revise la conexión.",
            bg='white',
            fg='red',
            font=('Arial', 18),
            wraplength=550,
            justify='center'
        )
        error_label.place(relx=0.5, rely=0.5, anchor='center')



 # ---------------------------------- MANEJO DE SOCKET PAR LA HUELLA -----------------------------
    """_ start_socket_server_
        Inicia un servidor de socket en un nuevo hilo.

        Funcionalidad:
        - Define la dirección del host y el puerto en los que el servidor de socket escuchará las conexiones entrantes.
        - Inicia un nuevo hilo para ejecutar la función `run_server`, que maneja las operaciones del servidor de socket.
        - El hilo se ejecuta en modo daemon, lo que significa que se cerrará automáticamente cuando el programa principal termine.

        Args:
            None

        Returns:
            None
    """        
    def start_socket_server(self):
        host = '127.0.0.1'
        port = 12345
        threading.Thread(target=self.run_server, args=(host, port), daemon=True).start()
        
    """_run_server_
        Ejecuta el servidor de socket y maneja las conexiones entrantes.

        Funcionalidad:
        - Crea y configura un socket del servidor para aceptar conexiones en la dirección y puerto especificados.
        - Configura el socket para reutilizar la dirección, permitiendo reinicios rápidos.
        - En un bucle infinito, escucha y acepta conexiones entrantes.
        - Para cada conexión aceptada, se inicia un nuevo hilo usando `self.executor` para manejar al cliente.

        Args:
            host (str): La dirección del host donde el servidor escuchará.
            port (int): El puerto en el que el servidor escuchará.

        Returns:
            None
    """
    def run_server(self, host, port):
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
                    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reusar la dirección del socket
                    server_socket.bind((host, port))
                    server_socket.listen(5)
                    logging.info(f'Servidor escuchando en {host}:{port}')
                    while True:
                        client_socket, addr = server_socket.accept()
                        logging.info(f'Conexión aceptada de {addr}')
                        self.executor.submit(self.handle_client, client_socket, addr)
            except Exception as e:
                logging.error(f'Ocurrió un error en el servidor: {e}')
                logging.info('Reiniciando servidor...')

    """
        Maneja la comunicación con un cliente conectado.

        Funcionalidad:
        - Recibe datos del cliente, decodifica y procesa los mensajes recibidos.
        - Mantiene un contador de toques del lector para determinar si se necesita una acción adicional.
        - Encola mensajes recibidos para su procesamiento posterior.
        - Envía una respuesta de confirmación al cliente después de recibir un mensaje.
        - Gestiona los errores y excepciones de la conexión del socket.
        - Finaliza cualquier proceso externo asociado y cierra el socket del cliente cuando se termina la conexión.

        Args:
            client_socket (socket.socket): El socket del cliente conectado.
            addr (tuple): La dirección del cliente (IP, puerto).

        Returns:
            None
    """
    def handle_client(self, client_socket, addr):
        external_process = None
        client_socket.settimeout(35)  
        
        def cleanup():
            try:
                if external_process is not None:
                    external_process.terminate()
                    external_process.wait()
                client_socket.close()
                logging.info(f'Conexión con el cliente {addr} cerrada y recursos liberados')
            except Exception as e:
                logging.error(f'Error al limpiar recursos para {addr}: {e}')

        try:
            while True:
                try:
                    datos = client_socket.recv(1024)
                    if not datos:
                        logging.info(f'Conexión cerrada por el cliente {addr}')
                        break
                        
                    mensaje_completo = datos.decode("ascii").strip()
                    logging.debug(f'Mensaje recibido de {addr}: {mensaje_completo}')

                    if mensaje_completo == "CERRAR_CONEXION":
                        logging.info(f'Cerrando conexión con {addr}')
                        break

                    # Encolar el mensaje para su procesamiento
                    self.message_queue.put(mensaje_completo)

                    # Enviar respuesta al cliente
                    try:
                        response = "Mensaje recibido"
                        client_socket.sendall(response.encode("ascii"))
                    except Exception as e:
                        logging.warning(f'Error al enviar respuesta a {addr}: {e}')
                        break

                except socket.timeout:
                    logging.warning(f'Timeout en la conexión con {addr}')
                    break
                except ConnectionResetError:
                    logging.warning(f'Conexión reiniciada por el cliente {addr}')
                    break
                except socket.error as e:
                    if e.errno in [10053, 10054]:  # Códigos de error específicos de Windows
                        logging.warning(f'Conexión interrumpida con {addr}: {e}')
                    else:
                        logging.error(f'Error de socket con el cliente {addr}: {e}')
                    break
                except Exception as e:
                    logging.error(f'Error procesando datos del cliente {addr}: {e}')
                    break

        except Exception as e:
            logging.error(f'Error general manejando el cliente {addr}: {e}')
        finally:
            cleanup()
  
    
    """_check_for_messages_
        Verifica y procesa mensajes de la cola de mensajes.

        Funcionalidad:
        - Establece el método de verificación como 'huella'.
        - Procesa todos los mensajes en la cola de mensajes.
        - Muestra un cuadro de mensaje en caso de error en la captura de huella.
        - Procesa otros mensajes de acuerdo a la acción recibida y el ID de huella.

        Lógica de Mensajes:
        - Si el mensaje es "FalloCapturaHuella", muestra un mensaje de error, reproduce un sonido de error y registra la entrada fallida.
        - Para otros mensajes, divide el mensaje en acción e ID de huella y llama a `self.process_message` para procesarlo.

        Programación:
        - Utiliza `self.root.after(100, self.check_for_messages)` para verificar la cola de mensajes cada 100 ms.

        Returns:
            None
    """
    def check_for_messages(self):
        self.metodo_verificacion = 'huella'
        while not self.message_queue.empty():
            mensaje_completo = self.message_queue.get_nowait()
            
            # Ignorar mensajes de keep-alive silenciosamente
            if mensaje_completo == "KEEP_ALIVE":
                continue
                
            print(f"Mensaje recibido: {mensaje_completo}")  # Solo imprime mensajes no keep-alive

            # Procesar otros mensajes
            partes = mensaje_completo.split(": ")
            accion = partes[0]
            idHuella = partes[1] if len(partes) > 1 else ""

            # Procesar el mensaje de acuerdo a la acción recibida
            self.process_message(accion, idHuella)

        # Programar la siguiente verificación
        self.root.after(100, self.check_for_messages)



    """_ process_message_
        Procesa mensajes de acciones relacionadas con la asistencia de empleados usando huella digital.

        Args:
            accion (str): La acción recibida para procesar (por ejemplo, "Asistencia tomada", "Escaneo fallido").
            idHuella (str): El identificador de la huella digital del empleado.

        Funcionalidad:
        - Procesa diferentes tipos de acciones basadas en el tipo de horario (abierto o cerrado) del empleado.
        - Muestra mensajes de éxito o error basados en la acción y estado del proceso de asistencia.
        - Reproduce sonidos asociados a cada estado del proceso de asistencia.
        - Registra la entrada o el fallo en el sistema.

        Nota:
        - Esta función solo procesará el mensaje si no hay otro mensaje activo.
    """
    def process_message(self, accion, idHuella):
        print(f"Procesando mensaje: Acción: {accion}, ID de Huella: {idHuella}")  # Añade este mensaje de depuración
        if not self.message_active:  # Solo procesar el mensaje si no hay un mensaje activo
            if accion == "Asistencia tomada":
                success, schedule_type = get_employee_schedule_type(self.db, idHuella)
                print(f"Tipo de Horario: {schedule_type}")  # Añade este mensaje de depuración
                if success:
                    if schedule_type == 'Abierto':
                        mensaje = add_open_schedule_check(self.db, idHuella, "entrada")
                        print(f"Mensaje de registro: {mensaje}")  # Añade este mensaje de depuración
                        self.msg_box_huella('Registro de Asistencia', mensaje, 'éxito')
                        self.register_fingerprint_entry(idHuella, 'Entrada Exitosa', True)
                        if mensaje == f"Entrada registrada con éxito {idHuella}. ¡Bienvenido de nuevo!":
                            play_normal_sound()
                        elif mensaje == f"Bienvenido {idHuella}, llegaste a tiempo. Asistencia tomada.":
                            play_normal_sound()
                        elif mensaje == f"Hasta luego {idHuella}, salida registrada a tiempo.":
                            play_sa_normal()
                    elif schedule_type == 'Cerrado':
                        result = verificar_y_actualizar_horario_fechas(self.db, idHuella)
                        if isinstance(result, tuple):
                            estatus, action_type = result
                            self.handle_status_messages(idHuella, estatus, action_type)
                        else:
                            self.msg_box_huella('Error', result, 'error')
                else:
                    self.msg_box_huella('Error', 'Tipo de horario no encontrado.', 'error')
            elif accion == "Escaneo fallido":
                play_error_sound()
                self.msg_box_huella('Error', 'El escaneo ha fallado. Por favor, intenta nuevamente.', 'error')
                self.register_fingerprint_entry(idHuella, 'Entrada Fallida', False)
            elif accion == "Usuario no registrado, o int?ntelo de nuevo":
                play_error_sound()
                self.msg_box_huella('Usuario no registrado', 'Por favor, regístrese o intente nuevamente.', 'error')
                self.register_fingerprint_entry(idHuella, 'Usuario no registrado', False)
            else:
                print("Acción no reconocida.")


    """_ handle_status_messages_
        Maneja los mensajes de estado y reproduce los sonidos correspondientes basados en el estatus y tipo de acción.

        Args:
            idHuella (str): El identificador de la huella digital del empleado.
            estatus (str): El estatus de la asistencia (por ejemplo, "NORMAL", "RETARDO").
            action_type (str): El tipo de acción realizada (por ejemplo, "entrada", "salida").

        Funcionalidad:
        - Muestra mensajes específicos para cada combinación de estatus y tipo de acción.
        - Reproduce sonidos específicos para cada combinación de estatus y tipo de acción.
        - Registra la entrada en el sistema basada en el éxito del proceso.

        Nota:
        - Utiliza un mapa de mensajes y tipos de mensajes para determinar el contenido y tipo de mensaje a mostrar.
    """
    
    def handle_status_messages(self, idHuella, estatus, action_type):
        status_messages = {
            "NORMAL": {
                "entrada": f"Bienvenido {idHuella}, llegaste a tiempo, asistencia tomada.",
                "salida": f"Hasta luego {idHuella}, salida registrada a tiempo."
            },
            "RETARDO": {
                "entrada": f"¡CASI! {idHuella}, llegaste un poco tarde, asistencia tomada con retardo.",
                "salida": f"¡CUIDADO! {idHuella}, has salido tarde."
            },
            "NOTA MALA": {
                "entrada": f"¡UPSS! {idHuella}, esta vez tienes nota mala, llegaste tarde.",
                "salida": f"¡ALERTA! {idHuella}, has salido mucho más tarde de lo previsto."
            }
        }
        message_types = {
            "NORMAL": "éxito",
            "RETARDO": "retardo",
            "NOTA MALA": "fueraderango"
        }
        message = status_messages.get(estatus, {}).get(action_type, "Ya escaneado o fuera de rango.")
        message_type = message_types.get(estatus, "error")
        print(f"Estatus: {estatus}, Acción: {action_type}, Mensaje: {message}")

        if message == "Ya escaneado o fuera de rango.":
            play_ya_scaneado()
        elif message == status_messages["NORMAL"]["entrada"]:
            play_normal_sound()
        elif message == status_messages["NORMAL"]["salida"]:
            play_sa_normal()
        elif message == status_messages["RETARDO"]["entrada"]:
            play_retardo_sound()
        elif message == status_messages["RETARDO"]["salida"]:
            play_sa_retardo()
        elif message == status_messages["NOTA MALA"]["entrada"]:
            play_nota_mala_sound()
        elif message == status_messages["NOTA MALA"]["salida"]:
            play_sa_notamala()

        self.msg_box_huella('Registro de Asistencia', message, message_type)
        entry_success = estatus in ["NORMAL", "RETARDO"]
        entry_type = 'Entrada Exitosa' if entry_success else 'Entrada Fallida'
        self.register_fingerprint_entry(idHuella, entry_type, entry_success)

#-----------------------------MANEJO DE MENSAJES DEL LECTOR DE HUELLA ----------------------
    """
    Muestra un mensaje en una ventana emergente dentro de la sección 2 de la interfaz de usuario, con un fondo 
    de color correspondiente al tipo de mensaje, y reproduce un sonido si es necesario.

    Args:
        title (str): El título del mensaje.
        message (str): El contenido del mensaje.
        message_type (str): El tipo de mensaje ('error', 'éxito', 'retardo', 'fueraderango').

    Returns:
        None

    Detalles:
        - La función verifica si ya hay un mensaje activo para evitar superposiciones.
        - Se asigna un color de fondo basado en el tipo de mensaje.
        - Si el mensaje es de tipo 'error', se reproduce un sonido de error.
        - Se limpia el contenido actual de `section2_frame` antes de mostrar el nuevo mensaje.
        - El mensaje se muestra en un contenedor configurado para no cambiar de tamaño y se posiciona en el centro de la sección.
        - El fondo del `image_label1` se cambia al color correspondiente y se restablece después de 5 segundos.
        - Se programa una llamada para limpiar el mensaje después de 5 segundos.
    """
    def msg_box_huella(self, title, message, message_type):
        print(f"Mostrando mensaje: {title} - {message} - Tipo: {message_type}")  # Añade este mensaje de depuración
        if self.message_active:
            return  # Si ya hay un mensaje activo, no hacer nada

        colors = {
            'error': '#FF0000',       # Rojo
            'éxito': '#008000',       # Verde
            'retardo': '#FFFF00',     # Amarillo
            'fueraderango': '#8A2BE2' # Morado
        }
        background_color = colors.get(message_type, '#D3D3D3')

        if message_type == 'error':
            play_error_sound()

        self.message_active = True  # Establecer el flag de mensaje activo

        # Limpiar los widgets existentes en section2_frame
        for widget in self.section2_frame.winfo_children():
            widget.destroy()

        # Configurar el contenedor de mensajes con tamaño fijo
        msg_container = tk.Frame(self.section2_frame, bg='white', borderwidth=0, relief="groove", width=800, height=380)
        msg_container.pack_propagate(False)  # Para asegurar que el tamaño no cambie
        msg_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        msg_container.place(relx=0.5, rely=0.5, anchor='center')

        try:
            # Formatear y mostrar el mensaje
            current_time = datetime.now().strftime("%H:%M:%S")
            current_day = dias_espanol[datetime.now().strftime("%A")]
            current_date = datetime.now().strftime("%d / %m / %Y")
            full_message = f"{title}\n{message}\n\nHora: {current_time}\nDía: {current_day}\nFecha: {current_date}"

            message_label = tk.Label(msg_container, text=full_message, bg='white', font=('Arial', 18), wraplength=400, justify=tk.CENTER)
            message_label.pack(expand=True, fill='both', padx=20, pady=20)

            # Cambiar el color de fondo de image_label1 y restablecerlo después de un tiempo
            self.image_label1.config(bg=background_color)
            self.root.after(5000, lambda: self.image_label1.config(bg='#D3D3D3'))

            # Restablecer el fondo del section2_frame después de 5 segundos
            self.root.after(5000, self.clear_message)
        except Exception as e:
            print(f"Error mostrando el mensaje: {e}")
            # Mostrar un mensaje de error básico si algo falla
            message_label = tk.Label(msg_container, text="Error mostrando el mensaje", bg='white', font=('Arial', 18), wraplength=480, justify=tk.CENTER)
            message_label.pack(expand=True, fill='both', padx=20, pady=20)
            # Restaurar el fondo de image_label1 en caso de error
            self.image_label1.config(bg='#D3D3D3')
            self.message_active = False  # Restablecer el flag de mensaje activo en caso de error

    def clear_message(self):
        # Limpia los widgets existentes en section2_frame
        for widget in self.section2_frame.winfo_children():
            widget.destroy()
        # Llama a update_section2 para actualizar la sección
        self.update_section2()
        self.message_active = False  # Restablecer el flag de mensaje activo





  #--------------------- MANEJO DE ERRORES DE LA HUELLA BIOMETRICA --------------------

    """_show_error_message_
        Muestra un mensaje de error en section2_frame.

        Args:
            title (str): Título del mensaje.
            message (str): Contenido del mensaje.
            duration (int): Duración del mensaje en milisegundos.
            background_color (str): Color de fondo para la indicación del error.
        
        Funcionalidad:
        - Muestra un mensaje de error en la interfaz con una duración específica.
        - Limpia los mensajes anteriores antes de mostrar el nuevo mensaje.
        - Cambia el color de fondo del label image_label1 para indicar un error.
        - Restablece el estado de la interfaz después de la duración especificada.
    """
    def show_error_message(self, title, message, duration=2000, background_color='#FF0000'):
        if self.message_active:
            return  # Si ya hay un mensaje activo, no hacer nada

        self.message_active = True  # Establecer el flag de mensaje activo

        try:
            # Limpiar los widgets existentes en section2_frame
            for widget in self.section2_frame.winfo_children():
                widget.destroy()

            # Configurar el contenedor de mensajes con tamaño fijo
            msg_container = tk.Frame(self.section2_frame, bg='white', borderwidth=2, relief="groove", width=800, height=380)
            msg_container.pack_propagate(False)  # Para asegurar que el tamaño no cambie
            msg_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            msg_container.place(relx=0.5, rely=0.5, anchor='center')

            # Formatear y mostrar el mensaje
            current_time = datetime.now().strftime("%H:%M:%S")
            current_day = dias_espanol[datetime.now().strftime("%A")]
            current_date = datetime.now().strftime("%d / %m / %Y")
            full_message = f"{title}\n{message}\n\nHora: {current_time}\nDía: {current_day}\nFecha: {current_date}"

            message_label = tk.Label(msg_container, text=full_message, bg='white', font=('Arial', 18), wraplength=480, justify=tk.CENTER)
            message_label.pack(expand=True, fill='both', padx=20, pady=20)

            # Cambiar el color de fondo de image_label1 y restablecerlo después de un tiempo
            self.image_label1.config(bg=background_color)
            self.root.after(duration, lambda: self.image_label1.config(bg='#D3D3D3'))

            # Restablecer el fondo del section2_frame después del tiempo especificado
            self.root.after(duration, self.clear_message)
        except Exception as e:
            print(f"Error mostrando el mensaje: {e}")
            # Mostrar un mensaje de error básico si algo falla
            error_msg_container = tk.Frame(self.section2_frame, bg='white', borderwidth=2, relief="groove", width=800, height=380)
            error_msg_container.pack_propagate(False)
            error_msg_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            error_message_label = tk.Label(error_msg_container, text="Error mostrando el mensaje", bg='white', font=('Arial', 18), wraplength=480, justify=tk.CENTER)
            error_message_label.pack(expand=True, fill='both', padx=20, pady=20)
            self.image_label1.config(bg='#D3D3D3')
            self.message_active = False  # Restablecer el flag de mensaje activo en caso de error

    def mostrar_usuario_no_registrado(self):
        self.metodo_verificacion = 'huella'
        self.show_error_message(
            title="Usuario no registrado",
            message="Por favor, regístrese o intente nuevamente.",
            duration=2000,  # Duración ajustada
            background_color='#FF0000'
        )

    def mostrar_error(self):
        self.show_error_message(
            title="Error",
            message="El escaneo ha fallado. Por favor, intenta nuevamente.",
            duration=2000,  # Duración ajustada
            background_color='#FF0000'
        )

    def clear_message(self):
        # Limpia los widgets existentes en section2_frame
        for widget in self.section2_frame.winfo_children():
            widget.destroy()
        # Llama a update_section2 para actualizar la sección
        self.update_section2()
        self.message_active = False  # Restablecer el flag de mensaje activo



    #-------------------- MANEJO DE MENSAJES PARA EL RECONOCIMIENTO FACIAL ----------------
 
    """
    Muestra un mensaje en una ventana emergente dentro de la sección 2 de la interfaz de usuario, con un fondo 
    de color correspondiente al tipo de mensaje, y gestiona la interfaz según el método de verificación.

    Args:
        title (str): El título del mensaje.
        message (str): El contenido del mensaje.
        message_type (str): El tipo de mensaje ('escaneando', 'error', 'éxito', 'retardo', 'fueraderango').

    Returns:
        None

    Detalles:
        - La función define colores para diferentes tipos de mensajes.
        - Limpia el contenido actual de `section2_frame` antes de mostrar el nuevo mensaje.
        - El mensaje se muestra en un contenedor configurado para no cambiar de tamaño y se posiciona en el centro de la sección.
        - Cambia temporalmente el color de fondo del label correspondiente y lo restablece después de 4 segundos.
        - Programa una llamada para limpiar el mensaje después de 4 segundos.
        - Captura cualquier excepción y muestra un mensaje de error básico en caso de fallo.
    """  
    def msg_box(self, title, message, message_type):
        colors = {
            'escaneando': '#D3D3D3',  # Gris
            'error': '#FF0000',  # Rojo
            'éxito': '#008000',  # Verde
            'retardo': '#E3DB1B',  # Amarillo
            'fueraderango': '#941EDD'  # Morado
        }
        background_color = colors.get(message_type, '#D3D3D3')

        try:
            # Limpiar los widgets existentes en section2_frame
            for widget in self.section2_frame.winfo_children():
                widget.destroy()

            self.message_active = True  # Establecer el flag de mensaje activo

            # Configurar el contenedor de mensajes con tamaño fijo
            msg_container = tk.Frame(self.section2_frame, bg='white', borderwidth=2, relief="groove", width=800, height=380)
            msg_container.pack_propagate(False)  # Para asegurar que el tamaño no cambie
            msg_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            msg_container.place(relx=0.5, rely=0.5, anchor='center')

            # Formatear y mostrar el mensaje
            current_time = datetime.now().strftime("%H:%M:%S")
            current_day = dias_espanol[datetime.now().strftime("%A")]
            current_date = datetime.now().strftime("%d / %m / %Y")
            full_message = f"{title}\n{message}\n\nHora: {current_time}\nDía: {current_day}\nFecha: {current_date}"

            message_label = tk.Label(msg_container, text=full_message, bg='white', font=('Arial', 18), wraplength=480, justify=tk.CENTER)
            message_label.pack(expand=True, fill='both', padx=20, pady=20)

            # Cambiar el color de fondo del label correspondiente y restablecerlo después de un tiempo
            if self.metodo_verificacion == 'facial':
                label_a_cambiar = self.image_label2
            elif self.metodo_verificacion == 'huella':
                label_a_cambiar = self.image_label1
            else:
                label_a_cambiar = None

            if label_a_cambiar:
                label_a_cambiar.config(bg=background_color)
                self.root.after(3000, lambda: label_a_cambiar.config(bg='#D3D3D3'))

            # Restablecer el fondo del section2_frame después de 5 segundos
            self.root.after(3000, self.clear_message)
        except Exception as e:
            print(f"Error mostrando el mensaje: {e}")
            # Mostrar un mensaje de error básico si algo falla
            msg_container = tk.Frame(self.section2_frame, bg='white', borderwidth=2, relief="groove", width=800, height=380)
            msg_container.pack_propagate(False)
            msg_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
            error_message_label = tk.Label(msg_container, text="Error mostrando el mensaje", bg='white', font=('Arial', 18), wraplength=480, justify=tk.CENTER)
            error_message_label.pack(expand=True, fill='both', padx=20, pady=20)
            if label_a_cambiar:
                label_a_cambiar.config(bg='#D3D3D3')
            self.message_active = False  # Restablecer el flag de mensaje activo en caso de error

    def clear_message(self):
        for widget in self.section2_frame.winfo_children():
            widget.destroy()
        self.update_section2()
        self.message_active = False  # Restablecer el flag de mensaje activo


#----------------------- MANEJO DEL PROCESO DE LA CAMARA Y ESCANEO FACIAL ----------------------
    """
        Inicializa la captura de video desde la cámara web y configura el procesamiento continuo de la misma.

        Args:
            label (tk.Label): El label de Tkinter donde se mostrará el video de la cámara web.
        
        Funcionalidad:
        - Abre la cámara web utilizando OpenCV.
        - Asigna el label de Tkinter donde se mostrará el video.
        - Inicia el procesamiento continuo de los frames de video capturados por la cámara web.
    """

    def add_webcam(self, label):
        try:
            self.cap = cv2.VideoCapture(1)  # Asegúrate de usar el índice correcto para tu cámara
            if not self.cap.isOpened():
                raise Exception("Failed to open webcam")
            self._label = label
            self.process_webcam()
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            self.cap.release()


   
    
   
    def scan_effect(self, frame, face_landmarks, color=(0, 255, 0), alpha_points=0.4, alpha_lines=0.23):
        if face_landmarks:
            overlay_points = frame.copy()
            overlay_lines = frame.copy()
            
            # Especificaciones para dibujar puntos normales
            drawing_spec_points = self.mp_drawing.DrawingSpec(
                thickness=1,
                circle_radius=1,
                color=color
            )

            # Especificaciones para dibujar puntos brillantes (solo se usarán si el color no es rojo)
            bright_color = (255, 255, 255)  # Color blanco para el brillo
            drawing_spec_bright = self.mp_drawing.DrawingSpec(
                thickness=1,
                circle_radius=2,  # Puntos más grandes para el brillo
                color=bright_color
            )

            # Dibujar puntos en los landmarks
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(overlay_points, (x, y), 1, color, -1)
                
                # Solo aplicar brillo si el color no es rojo
                if color != (0, 0, 255) and random.random() < 0.1:  # 10% de probabilidad de brillo
                    cv2.circle(overlay_points, (x, y), drawing_spec_bright.circle_radius, bright_color, -1)
            
            # Especificaciones para dibujar líneas más delgadas
            drawing_spec_lines = self.mp_drawing.DrawingSpec(
                thickness=1,
                circle_radius=1,
                color=color
            )

            self.mp_drawing.draw_landmarks(
                image=overlay_lines,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec_lines
            )

            # Ajustar transparencia y combinar las superposiciones
            frame = cv2.addWeighted(overlay_points, alpha_points, frame, 1 - alpha_points, 0)
            frame = cv2.addWeighted(overlay_lines, alpha_lines, frame, 1 - alpha_lines, 0)
        
        return frame

    """  _process_webcam_

        Captura y procesa fotogramas de la cámara web para realizar el reconocimiento facial en tiempo real.

        Esta función se ejecuta continuamente para leer fotogramas de la cámara web, procesarlos y mostrar
        la salida en una etiqueta de la interfaz gráfica de usuario. Los pasos clave incluyen:

        1. Inicialización y verificación: Asegura que los atributos necesarios están inicializados y verifica
        si hay un mensaje de advertencia activo.
        2. Captura de fotogramas: Lee un fotograma de la cámara web.
        3. Preprocesamiento: Invierte el fotograma horizontalmente y lo convierte a escala de grises.
        4. Ajuste de brillo: Ajusta el brillo del fotograma basado en la detección del rostro.
        5. Detección de rostros: Utiliza un detector de rostros para encontrar rostros en el fotograma.
        6. Seguimiento de rostros: Inicia el seguimiento del rostro más grande detectado.
        7. Procesamiento de landmarks: Procesa los landmarks del rostro si se detectan.
        8. Efectos de escaneo: Aplica un efecto de escaneo al fotograma.
        9. Validación de posición: Verifica si el rostro está bien posicionado y muestra mensajes de 
        superposición si es necesario.
        10. Actualización de seguimiento: Actualiza el seguimiento del rostro.
        11. Visualización: Convierte el fotograma procesado a un formato compatible con Tkinter y actualiza 
            la etiqueta de la cámara web.
        12. Manejo de errores: Captura y muestra cualquier error que ocurra durante el procesamiento.
        13. Repetición: Programa la función para ejecutarse de nuevo después de un breve retraso.

        La función también ajusta el brillo de una copia limpia del fotograma para su uso en el reconocimiento 
        facial y reinicia la línea de escaneo si no se detecta ningún rostro.

        Args:
            None

        Returns:
            None
    """
    def crop_and_resize_face(self, frame, face, padding_percent=25):
        """
        Recorta y redimensiona el área del rostro de manera precisa.
        """
        try:
            height, width = frame.shape[:2]

            # Calcular padding basado en el porcentaje del tamaño del rostro
            basic_padding = int((face.width() * padding_percent) / 100)
            padding_x = basic_padding
            padding_y = int(basic_padding * 1.618)

            # Calcular el centro del rostro
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2

            # Calcular dimensiones del recorte incluyendo padding
            crop_width = face.width() + (2 * padding_x)
            crop_height = face.height() + (2 * padding_y)

            # Añadir espacio adicional a la derecha y reducir más espacio a la izquierda
            extra_right_padding = int(face.width() * 0.15)  # Añadir 15% más a la derecha
            reduce_left_padding = int(face.width() * 0.15)  # Reducir 15% de la izquierda

            # Ajustar para mantener el rostro centrado, con más espacio a la derecha y menos a la izquierda
            x1 = max(0, face_center_x - (crop_width // 2) + reduce_left_padding)
            x2 = face_center_x + (crop_width // 2) + extra_right_padding

            # Ajustar si se sale del marco
            if x2 > width:
                x2 = width
                x1 = max(0, x2 - crop_width)

            y1 = max(0, face_center_y - (crop_height // 2))
            y2 = min(height, y1 + crop_height)
            if y2 > height:
                y2 = height
                y1 = max(0, y2 - crop_height)

            # Validar dimensiones después del ajuste
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                return None, None

            # Recortar el rostro
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                return None, None

            # Calcular dimensiones finales manteniendo proporción
            target_height = 400
            aspect_ratio = crop_width / crop_height
            target_width = int(target_height * aspect_ratio)

            # Asegurar dimensiones mínimas y máximas
            target_width = min(max(target_width, 300), 600)
            target_height = min(max(target_height, 300), 600)

            # Redimensionar con interpolación de alta calidad
            face_crop_resized = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

            return face_crop_resized, (x1, y1, x2, y2)

        except Exception as e:
            logging.error(f"Error en crop_and_resize_face: {e}")
            return None, None



    def draw_face_outline(self, frame, coords, color=(0, 255, 0), thickness=2):
        """
        Dibuja un marco estético alrededor del área de recorte.
        """
        try:
            if coords is None:
                return frame

            x1, y1, x2, y2 = coords
            h, w = frame.shape[:2]

            # Validar coordenadas
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Calcular dimensiones para las esquinas
            rect_width = x2 - x1
            rect_height = y2 - y1
            corner_length = int(min(rect_width, rect_height) * 0.2)

            # Crear una copia del frame para el overlay
            overlay = frame.copy()

            # Dibujar rectángulo semitransparente
            alpha = 0.2
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Dibujar esquinas
            corners = [
                [(x1, y1), (x1 + corner_length, y1), (x1, y1 + corner_length)],  # Superior izquierda
                [(x2, y1), (x2 - corner_length, y1), (x2, y1 + corner_length)],  # Superior derecha
                [(x1, y2), (x1 + corner_length, y2), (x1, y2 - corner_length)],  # Inferior izquierda
                [(x2, y2), (x2 - corner_length, y2), (x2, y2 - corner_length)]   # Inferior derecha
            ]

            for corner in corners:
                cv2.line(frame, corner[0], corner[1], color, thickness + 1)
                cv2.line(frame, corner[0], corner[2], color, thickness + 1)

            return frame

        except Exception as e:
            logging.error(f"Error en draw_face_outline: {e}")
            return frame

        
        
    def is_face_in_capture_zone(self, face, frame_shape):
        """
        Verifica si el rostro está dentro de la zona de captura inmediata (10-20cm).
        Usa el tamaño del rostro como aproximación de la distancia física.
        
        Args:
            face: Rectángulo del rostro detectado
            frame_shape: Dimensiones del frame
        
        Returns:
            bool: True si el rostro está en la zona de captura inmediata
        """
        try:
            # A 10-20cm, un rostro adulto típicamente ocupa estas proporciones del frame
            MIN_FACE_RATIO = 0.15  # Aproximadamente 20cm
            MAX_FACE_RATIO = 0.30  # Aproximadamente 10cm
            
            frame_height = frame_shape[0]
            face_height_ratio = face.height() / frame_height
            
            return MIN_FACE_RATIO <= face_height_ratio <= MAX_FACE_RATIO
            
        except Exception as e:
            logging.error(f"Error en is_face_in_capture_zone: {e}")
            return False
    def ajustar_brillo_adaptativo(self, img, face):
        """
        Ajusta dinámicamente el brillo y contraste basado en las condiciones de iluminación.
        """
        try:
            # Extraer región del rostro
            x, y, w, h = max(0, face.left()), max(0, face.top()), face.width(), face.height()
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            rostro = img[y:y+h, x:x+w]

            # Convertir a YCrCb para mejor manejo de luminancia
            ycrcb = cv2.cvtColor(rostro, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:,:,0]

            # Calcular estadísticas de luminancia
            brillo_promedio = np.mean(y_channel)
            desviacion = np.std(y_channel)
            
            # Parámetros de ajuste
            alpha = 1.0  # Contraste
            beta = 0     # Brillo

            # Ajuste para luz baja (rostros oscuros)
            if brillo_promedio < 80:  # Valor bajo de brillo
                if desviacion < 30:  # Bajo contraste
                    alpha = 1.5
                    beta = 30
                else:
                    alpha = 1.3
                    beta = 20
            
            # Ajuste para luz alta (rostros muy brillantes)
            elif brillo_promedio > 200:  # Valor alto de brillo
                if desviacion < 30:  # Bajo contraste
                    alpha = 0.6
                    beta = -30
                else:
                    alpha = 0.7
                    beta = -20
            
            # Ajuste para condiciones intermedias pero problemáticas
            elif desviacion < 40:  # Bajo contraste en general
                if brillo_promedio < 120:
                    alpha = 1.2
                    beta = 10
                else:
                    alpha = 0.9
                    beta = -10

            # Aplicar corrección gamma para mejorar detalles en sombras
            gamma = 1.0
            if brillo_promedio < 100:
                gamma = 0.8
            elif brillo_promedio > 180:
                gamma = 1.2

            # Aplicar ajustes
            rostro_ajustado = cv2.convertScaleAbs(rostro, alpha=alpha, beta=beta)
            
            # Aplicar corrección gamma
            look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            rostro_ajustado = cv2.LUT(rostro_ajustado, look_up_table)

            # Aplicar ecualización adaptativa solo en casos extremos
            if desviacion < 20 or brillo_promedio < 60 or brillo_promedio > 200:
                lab = cv2.cvtColor(rostro_ajustado, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                rostro_ajustado = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Suavizado para reducir ruido
            rostro_ajustado = cv2.bilateralFilter(rostro_ajustado, d=5, sigmaColor=50, sigmaSpace=50)

            # Reintegrar el rostro ajustado a la imagen original
            img[y:y+h, x:x+w] = rostro_ajustado

            return img
        except Exception as e:
            logging.error(f"Error en ajustar_brillo_adaptativo: {e}")
            return img
    
    # def process_webcam(self):
    #     try:
    #         self.check_initialized()
    #         if self.is_warning_message_active:
    #             self.webcam_label.after(10, self.process_webcam)
    #             return

    #         self.metodo_verificacion = 'facial'

    #         ret, frame = self.cap.read()
    #         if not ret:
    #             logging.error("Failed to grab frame")
    #             self.webcam_label.after(10, self.process_webcam)
    #             return

    #         frame = cv2.flip(frame, 1)
    #         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    #         clean_frame = frame.copy()
    #         display_frame = frame.copy()

    #         frame = self.zoom_image(frame, zoom_factor=1)
    #         label_width = self.webcam_label.winfo_width() or 1
    #         label_height = self.webcam_label.winfo_height() or 1

    #         aspect_ratio = frame.shape[1] / frame.shape[0]
    #         new_width = max(int(label_width * 0.97), 1)
    #         new_height = max(int(new_width / aspect_ratio), 1)

    #         if new_height > int(label_height * 0.97):
    #             new_height = max(int(label_height * 0.97), 1)
    #             new_width = max(int(new_height * aspect_ratio), 1)

    #         if new_width > 0 and new_height > 0:
    #             display_frame = cv2.resize(frame, (new_width, new_height))
    #         else:
    #             logging.error(f"Dimensiones inválidas para redimensionar: ancho={new_width}, alto={new_height}")

    #         faces = self.detector(gray_frame)
    #         feedback_color = COLOR_RED
    #         mesh_color = (0, 0, 255)  # Red for mesh
    #         face_detected = False

    #         # Rangos ajustados para permitir más distancia
    #         PRE_MIN_FACE_SIZE = 110    # Comienza a mostrar malla roja
    #         MIN_FACE_SIZE = 130        # Tamaño mínimo para captura
    #         MAX_FACE_SIZE = 250        # Tamaño máximo para captura

    #         # Procesamos la malla facial
    #         results_display = self.face_mesh.process(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

    #         if faces:
    #             closest_face = max(faces, key=lambda rect: rect.width() * rect.height())
    #             face_width = closest_face.width()
    #             face_height = closest_face.height()

    #             # Si el rostro está al menos en el rango de pre-detección
    #             if face_width >= PRE_MIN_FACE_SIZE or face_height >= PRE_MIN_FACE_SIZE:
    #                 face_detected = True
                    
    #                 if face_width < MIN_FACE_SIZE and face_height < MIN_FACE_SIZE:
    #                     # Mostrar malla roja y mensaje cuando está muy lejos
    #                     self.overlay_message(display_frame, "Por favor, acerquese mas.")
    #                     feedback_color = COLOR_RED
    #                     mesh_color = (0, 0, 255)
    #                     if results_display.multi_face_landmarks:
    #                         display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)
    #                     self.face_detected_time = None
    #                     self.capturing = False
    #                 elif face_width > MAX_FACE_SIZE or face_height > MAX_FACE_SIZE:
    #                     # Cuando está demasiado cerca
    #                     self.overlay_message(display_frame, "Por favor, alejese un poco.")
    #                     feedback_color = COLOR_RED
    #                     mesh_color = (0, 0, 255)
    #                     if results_display.multi_face_landmarks:
    #                         display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)
    #                     self.face_detected_time = None
    #                     self.capturing = False
    #                 else:
    #                     # En rango válido para captura
    #                     clean_cropped, crop_coords = self.crop_and_resize_face(clean_frame, closest_face)
    #                     if clean_cropped is not None:
    #                         clean_cropped = self.ajustar_brillo_adaptativo(clean_cropped, closest_face)
    #                         self.clean_frame = clean_cropped

    #                     display_cropped, _ = self.crop_and_resize_face(display_frame, closest_face)
    #                     if display_cropped is not None:
    #                         display_frame = self.draw_face_outline(display_frame, crop_coords)

    #                     self.handle_face_tracking(display_frame, closest_face)
                        
    #                     self.clear_overlay_message()
    #                     feedback_color = COLOR_GREEN
    #                     mesh_color = (0, 255, 0)

    #                     current_time = time.time()
    #                     if self.face_detected_time is None:
    #                         self.face_detected_time = current_time
    #                         self.capturing = True

    #                     if current_time - self.last_capture_time > 3:
    #                         if current_time - self.face_detected_time >= 0.5:
    #                             self.most_recent_capture_arr = self.clean_frame.copy()
    #                             try:
    #                                 self.login()
    #                             except Exception as e:
    #                                 logging.error(f"Error during login process: {e}")
    #                             self.last_capture_time = current_time
    #                             self.face_detected_time = None
    #                             self.capturing = False

    #                     if results_display.multi_face_landmarks:
    #                         display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)
                
    #             if len(faces) > 1:
    #                 self.overlay_message(display_frame, "Por favor, una persona a la vez")
    #                 feedback_color = COLOR_RED
    #                 self.face_detected_time = None
    #                 self.capturing = False
    #         else:
    #             self.clear_overlay_message()
    #             feedback_color = COLOR_RED
    #             mesh_color = (0, 0, 255)
    #             self.face_detected_time = None
    #             self.capturing = False

    #         # Aplicar feedback final
    #         self.apply_feedback(display_frame, feedback_color, face_detected)

    #     except Exception as e:
    #         logging.error(f"Error al procesar la imagen: {e}")

    #     self.webcam_label.after(10, self.process_webcam)
    def process_webcam(self):
        try:
            self.check_initialized()
            if self.is_warning_message_active:
                self.webcam_label.after(10, self.process_webcam)
                return

            self.metodo_verificacion = 'facial'

            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                self.webcam_label.after(10, self.process_webcam)
                return

            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            clean_frame = frame.copy()
            display_frame = frame.copy()

            frame = self.zoom_image(frame, zoom_factor=1)
            label_width = self.webcam_label.winfo_width() or 1
            label_height = self.webcam_label.winfo_height() or 1

            aspect_ratio = frame.shape[1] / frame.shape[0]
            new_width = max(int(label_width * 0.97), 1)
            new_height = max(int(new_width / aspect_ratio), 1)

            if new_height > int(label_height * 0.97):
                new_height = max(int(label_height * 0.97), 1)
                new_width = max(int(new_height * aspect_ratio), 1)

            if new_width > 0 and new_height > 0:
                display_frame = cv2.resize(frame, (new_width, new_height))
            else:
                logging.error(f"Dimensiones inválidas para redimensionar: ancho={new_width}, alto={new_height}")

            faces = self.detector(gray_frame)
            feedback_color = COLOR_RED
            mesh_color = (0, 0, 255)  # Red for mesh
            face_detected = False

            PRE_MIN_FACE_SIZE = 110      # Comienza detección con malla roja
            MIN_FACE_SIZE = 140          # Comienza captura con malla amarilla
            IDEAL_MIN_SIZE = 170         # Rango ideal con malla verde
            IDEAL_MAX_SIZE = 250         # Máximo ideal
            MAX_FACE_SIZE = 300          # Máximo absoluto

            # Procesamos la malla facial
            results_display = self.face_mesh.process(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))

            if faces:
                closest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                face_width = closest_face.width()
                face_height = closest_face.height()

                face_detected = True

                # Secuencia exacta de validaciones
                if face_width < PRE_MIN_FACE_SIZE:
                    # Rostro muy lejano, sin malla
                    self.overlay_message(display_frame, "Por favor, acerquese mas.")
                    feedback_color = COLOR_RED
                    mesh_color = (0, 0, 255)
                    self.face_detected_time = None
                    self.capturing = False

                elif PRE_MIN_FACE_SIZE <= face_width < MIN_FACE_SIZE:
                    # Entre PRE_MIN y MIN: malla roja
                    self.overlay_message(display_frame, "Acerquese un poco mas.")
                    feedback_color = COLOR_RED
                    mesh_color = (0, 0, 255)
                    if results_display.multi_face_landmarks:
                        display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)
                    self.face_detected_time = None
                    self.capturing = False

                elif MIN_FACE_SIZE <= face_width < IDEAL_MIN_SIZE:
                    # Entre MIN y IDEAL_MIN: malla amarilla + recorte
                    clean_cropped, crop_coords = self.crop_and_resize_face(clean_frame, closest_face)
                    if clean_cropped is not None:
                        clean_cropped = self.ajustar_brillo_adaptativo(clean_cropped, closest_face)
                        self.clean_frame = clean_cropped

                    display_cropped, _ = self.crop_and_resize_face(display_frame, closest_face)
                    if display_cropped is not None:
                        display_frame = self.draw_face_outline(display_frame, crop_coords)

                    feedback_color = COLOR_YELLOW
                    mesh_color = (0, 255, 255)
                    self.overlay_message(display_frame, "Captura aceptable.")
                    if results_display.multi_face_landmarks:
                        display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)

                    # Iniciar captura
                    current_time = time.time()
                    if self.face_detected_time is None:
                        self.face_detected_time = current_time
                        self.capturing = True

                    if current_time - self.last_capture_time > 3:
                        if current_time - self.face_detected_time >= 0.5:
                            self.most_recent_capture_arr = self.clean_frame.copy()
                            try:
                                self.login()
                            except Exception as e:
                                logging.error(f"Error during login process: {e}")
                            self.last_capture_time = current_time
                            self.face_detected_time = None
                            self.capturing = False

                elif IDEAL_MIN_SIZE <= face_width <= IDEAL_MAX_SIZE:
                    # Entre IDEAL_MIN y IDEAL_MAX: malla verde + recorte
                    clean_cropped, crop_coords = self.crop_and_resize_face(clean_frame, closest_face)
                    if clean_cropped is not None:
                        clean_cropped = self.ajustar_brillo_adaptativo(clean_cropped, closest_face)
                        self.clean_frame = clean_cropped

                    display_cropped, _ = self.crop_and_resize_face(display_frame, closest_face)
                    if display_cropped is not None:
                        display_frame = self.draw_face_outline(display_frame, crop_coords)

                    feedback_color = COLOR_GREEN
                    mesh_color = (0, 255, 0)
                    self.overlay_message(display_frame, "Captura ideal.")
                    if results_display.multi_face_landmarks:
                        display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)

                    # Iniciar captura
                    current_time = time.time()
                    if self.face_detected_time is None:
                        self.face_detected_time = current_time
                        self.capturing = True

                    if current_time - self.last_capture_time > 3:
                        if current_time - self.face_detected_time >= 0.5:
                            self.most_recent_capture_arr = self.clean_frame.copy()
                            try:
                                self.login()
                            except Exception as e:
                                logging.error(f"Error during login process: {e}")
                            self.last_capture_time = current_time
                            self.face_detected_time = None
                            self.capturing = False

                elif IDEAL_MAX_SIZE < face_width <= MAX_FACE_SIZE:
                    # Entre IDEAL_MAX y MAX: malla amarilla + recorte
                    clean_cropped, crop_coords = self.crop_and_resize_face(clean_frame, closest_face)
                    if clean_cropped is not None:
                        clean_cropped = self.ajustar_brillo_adaptativo(clean_cropped, closest_face)
                        self.clean_frame = clean_cropped

                    display_cropped, _ = self.crop_and_resize_face(display_frame, closest_face)
                    if display_cropped is not None:
                        display_frame = self.draw_face_outline(display_frame, crop_coords)

                    feedback_color = COLOR_YELLOW
                    mesh_color = (0, 255, 255)
                    self.overlay_message(display_frame, "Un poco mas lejos.")
                    if results_display.multi_face_landmarks:
                        display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)

                else:  # > MAX_FACE_SIZE
                    # Por encima de MAX: malla roja
                    self.overlay_message(display_frame, "Por favor, alejese un poco.")
                    feedback_color = COLOR_RED
                    mesh_color = (0, 0, 255)
                    if results_display.multi_face_landmarks:
                        display_frame = self.scan_effect(display_frame, results_display.multi_face_landmarks[0], mesh_color)
                    self.face_detected_time = None
                    self.capturing = False

                if len(faces) > 1:
                    self.overlay_message(display_frame, "Por favor, una persona a la vez")
                    feedback_color = COLOR_RED
                    self.face_detected_time = None
                    self.capturing = False

            else:
                self.clear_overlay_message()
                feedback_color = COLOR_RED
                mesh_color = (0, 0, 255)
                self.face_detected_time = None
                self.capturing = False

            # Aplicar feedback final
            self.apply_feedback(display_frame, feedback_color, face_detected)

        except Exception as e:
            logging.error(f"Error al procesar la imagen: {e}")

        self.webcam_label.after(10, self.process_webcam)

#---------------------------------------FUNCIONES AUXIALIARES------------------------------------
   
   
    def adjust_brightness(self, frame, clean_frame, face):
        frame = self.ajustar_brillo_adaptativo(frame, face)
        clean_frame = self.ajustar_brillo_adaptativo(clean_frame, face)
        return frame, clean_frame

    def handle_face_tracking(self, frame, largest_face):
        if not self.tracking_face or (self.tracking_face and self.calculate_distance(self.tracked_face_rect, largest_face) > 50):
            self.tracker = cv2.TrackerCSRT_create()
            self.tracker.init(frame, (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()))
            self.tracking_face = True
            self.tracked_face_rect = largest_face

    def evaluate_face_position(self, largest_face, frame_shape, gray_frame):
        MIN_FACE_SIZE = 130  # Umbral mínimo de tamaño del rostro en píxeles
        if largest_face.width() < MIN_FACE_SIZE or largest_face.height() < MIN_FACE_SIZE:
            self.overlay_message(self.clean_frame, "Por favor, acérquese más.")
            return COLOR_RED
        if not self.is_face_well_positioned(largest_face, frame_shape):
            self.overlay_message(self.clean_frame, "Coloque su rostro dentro del área central.")
            return COLOR_RED
        elif not self.is_facing_front(largest_face, gray_frame):
            self.overlay_message(self.clean_frame, "Por favor, mire hacia la cámara.")
            return COLOR_YELLOW
        else:
            self.clear_overlay_message()
            return COLOR_GREEN


    def update_tracking(self, frame):
        if self.tracking_face:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                self.tracked_face_rect = dlib.rectangle(x, y, x + w, y + h)
            else:
                self.tracking_face = False

    def apply_feedback(self, frame, feedback_color, face_detected):
        frame = draw_feedback(frame, feedback_color, face_detected)
        img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)
        self.most_recent_capture_arr = self.clean_frame




    # Función auxiliar para calcular la distancia entre dos rectángulos
    def calculate_distance(self, rect1, rect2):
        x1, y1, w1, h1 = rect1.left(), rect1.top(), rect1.width(), rect1.height()
        x2, y2, w2, h2 = rect2.left(), rect2.top(), rect2.width(), rect2.height()
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        return distance

    """  _zoom_image_

        Aplica un efecto de zoom a una imagen, recortando y redimensionando la región central de la imagen original.

        Args:
            image (numpy.ndarray): La imagen de entrada en formato de matriz numpy.
            zoom_factor (float): El factor de zoom. Valores mayores a 1 aumentan el zoom, mientras que valores menores a 1 lo disminuyen.

        Returns:
            numpy.ndarray: La imagen redimensionada con el efecto de zoom aplicado.
    """
    def zoom_image(self,image, zoom_factor):
        height, width = image.shape[:2] # Obtiene las dimensiones de la imagen
        center_x, center_y = width // 2, height // 2# Calcula el centro de la imagen

         # Calcula el cuadro delimitador de la región a mantener después del zoom
        new_width = int(width / zoom_factor)
        new_height = int(height / zoom_factor)
        x1 = center_x - new_width // 2
        y1 = center_y - new_height // 2
        x2 = center_x + new_width // 2
        y2 = center_y + new_height // 2

         # Asegura que el cuadro delimitador esté dentro de las dimensiones de la imagen
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

         # Recorta y redimensiona la imagen para lograr el efecto de zoom
        cropped_image = image[y1:y2, x1:x2]
        zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        return zoomed_image



    """  
    _overlay_message_

    Superpone un mensaje de texto sobre un cuadro de imagen.

    Args:
        frame (numpy.ndarray): La imagen de entrada en formato de matriz numpy.
        message (str): El mensaje de texto a superponer.
        padding (int, optional): El espacio de relleno alrededor del texto. Por defecto es 10.

    Returns:
        None
    """

    def overlay_message(self, frame, message, padding=10):
        self.overlay_message_text = message
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 50

        overlay = frame.copy()

        # Dibuja un rectángulo semitransparente detrás del texto
        cv2.rectangle(overlay, (text_x - padding, text_y - text_size[1] - padding), 
                    (text_x + text_size[0] + padding, text_y + padding), (255, 255, 255), -1)
        alpha = 0.6  # Opacidad del rectángulo

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Dibuja el texto sobre el rectángulo
        cv2.putText(frame, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    """ _clear_overlay_message_
       
        Limpia el mensaje superpuesto actualmente en el cuadro de imagen.

        Args:
            None

        Returns:
            None
    """
    def clear_overlay_message(self):
        self.overlay_message_text = None

    """ _is_face_well_positioned_
        
        Verifica si un rostro está bien posicionado dentro del marco de la imagen.

        Args:
            face (dlib.rectangle): El rectángulo que delimita la cara detectada.
            frame_shape (tuple): Las dimensiones de la imagen en formato (altura, anchura).

        Returns:
            bool: True si la cara está bien posicionada, False en caso contrario.
    """
    def is_face_well_positioned(self, face, frame_shape):
        frame_height, frame_width = frame_shape[:2] # Obtiene la altura y anchura de la imagen
        x, y, w, h = face.left(), face.top(), face.width(), face.height() # Obtiene las coordenadas y dimensiones del rectángulo que delimita la cara

        horizontal_center = frame_width / 2  # Calcula el centro horizontal de la imagen
        vertical_center = frame_height / 2  # Calcula el centro vertical de la imagen
        margin_x = frame_width * 0.2  # Calcula el margen horizontal permitido
        margin_y = frame_height * 0.2  # Calcula el margen vertical permitido

        face_center_x = x + w / 2# Calcula el centro horizontal del rostro
        face_center_y = y + h / 2 # Calcula el centro vertical del rostro
        # Verifica si el centro del rostro está dentro de los márgenes permitidos
        well_positioned = (
            (horizontal_center - margin_x < face_center_x < horizontal_center + margin_x) and
            (vertical_center - margin_y < face_center_y < vertical_center + margin_y)
        )
        return well_positioned

    # def is_facing_front(self, face, gray_frame):
    #     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    #     x, y, w, h = face.left(), face.top(), face.width(), face.height()
    #     roi_gray = gray_frame[y:y+h, x:x+w]
    #     eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #     return len(eyes) >= 2


    """_is_facing_front_
        

        Evalúa si la persona está mirando hacia el frente usando detección de ojos.

        Args:
            face (dlib.rectangle): El rectángulo que delimita la cara detectada.
            gray_frame (numpy.ndarray): La imagen en escala de grises donde se realizará la detección.

        Returns:
            bool: True si se detectan al menos dos ojos en la región de la cara, False en caso contrario.
    """


    def is_facing_front(self, face, gray_frame):
        
         # Obtener la ruta absoluta al archivo haarcascade_eye.xml
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eye_cascade_path = os.path.join(current_dir, "utils", "haarcascade_eye.xml")

        # Carga el clasificador de ojos pre-entrenado
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


        # Obtiene las coordenadas y dimensiones del rectángulo que delimita la cara
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Definir la región de interés (ROI) para la detección de ojos
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Aplicar ecualización del histograma para mejorar el contraste
        roi_gray = cv2.equalizeHist(roi_gray)
        
        # Detectar los ojos en la ROI
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(90, 90)
        ) # Aplica el detector de ojos sobre la región de interés

        # Retorna True si se detectan al menos dos ojos, indicando que la 
        # persona está mirando hacia el frente
        return len(eyes) >= 2

    

    """_show_positioning_message_

        Muestra un mensaje de posicionamiento en la interfaz de usuario para indicar al usuario cómo colocar su rostro.

        Args:
            message (str): El mensaje a mostrar.

    """
    def show_positioning_message(self, message):
        # Limpia cualquier mensaje de posicionamiento existente
        self.clear_positioning_message()
         # Crea y configura una nueva etiqueta para mostrar el mensaje
        self.positioning_label = tk.Label(self.main_frame, text=message, 
                                        fg='red', font=('Arial', 12))
        self.positioning_label.grid(row=1, column=0, columnspan=2, sticky='nsew', pady=10)
        # Establece un temporizador para borrar el mensaje después de 4000 milisegundos (4 segundos)
        self.positioning_label.after(4000, self.clear_positioning_message)

    """_clear_positioning_message_
        
        Limpia el mensaje de posicionamiento de la interfaz de usuario si existe.
        
        Esta función verifica si el objeto self tiene un atributo 'positioning_label' 
        y si dicho atributo existe en el contexto de la interfaz de usuario. 
        Si es así, destruye el widget y elimina el atributo 'positioning_label' de self.
        """
    
    def clear_positioning_message(self):

        if hasattr(self, 'positioning_label') and self.positioning_label.winfo_exists():
            # Destruye el widget de la etiqueta de posicionamiento
            self.positioning_label.destroy()
            # Elimina el atributo 'positioning_label' de self
            del self.positioning_label

    """_reset_scan_line_

        Reinicia la posición y dirección de la línea de escaneo.
        
        Esta función establece la posición de escaneo a 0 y la dirección de escaneo a 1. 
        Es utilizada para reiniciar el estado de la línea de escaneo cuando se inicia 
        o se reinicia un proceso de escaneo.
    """

    # def reset_scan_line(self):
    #     # Establece la posición de escaneo a 0
    #     self.scan_position = 0
    #      # Establece la dirección de escaneo a 1
    #     self.scan_direction = 1

    """_draw_scan_line_
        
        Dibuja una línea de escaneo en el frame y actualiza su posición.

        Esta función dibuja una línea de escaneo horizontalmente a través del frame, 
        variando la intensidad del color verde a medida que se aleja del centro horizontal.
        La posición de la línea de escaneo se actualiza en cada llamada y la dirección 
        de la línea se invierte cuando alcanza el borde superior o inferior del frame.
        Además, se realiza una captura limpia del frame y se llama a la función de login 
        cuando la línea de escaneo cambia de dirección.

        Args:
            frame (numpy.ndarray): La imagen de entrada en formato de matriz numpy.

        Returns:
            numpy.ndarray: La imagen con la línea de escaneo dibujada.
    """
  
   
    
    """_scan_effect_

    Aplica un efecto de escaneo al frame y resalta los landmarks de la cara si están presentes.

    Args:
        frame (numpy.ndarray): La imagen de entrada en formato de matriz numpy.
        face_landmarks (object): Los landmarks de la cara detectados en el frame.

    Returns:
        numpy.ndarray: La imagen con el efecto de escaneo aplicado.
    """
 

    """ _login_
        Realiza el proceso de inicio de sesión basado en reconocimiento facial, validación de antispoofing
        y verificación de horarios, gestionando distintos tipos de errores y respuestas basadas en el 
        resultado de cada paso del proceso.
    """
    
    def login(self):
        def antispoofing_thread():
            try:
                label = self.run_antispoofing_test()
                self.result_queue.put(('antispoofing', label))
            except Exception as e:
                logging.error(f"Error in antispoofing_thread: {str(e)}")
                self.result_queue.put(('antispoofing_error', str(e)))

        threading.Thread(target=antispoofing_thread).start()

    def run_antispoofing_test(self):
        try:
            result = test(
                image=self.most_recent_capture_arr,
                model_dir='./resources/anti_spoof_models',
                device_id=0
            )
            return result
        except Exception as e:
            logging.error(f'Error en el modelo antispoofing: {str(e)}')
            raise RuntimeError(f'Error en el modelo antispoofing: {str(e)}')

    def check_result_queue(self):
        try:
            while not self.result_queue.empty():
                result_type, result = self.result_queue.get_nowait()
                if result_type == 'antispoofing':
                    self.handle_antispoofing_result(result)
                elif result_type == 'antispoofing_error':
                    self.handle_antispoofing_error(result)
                elif result_type == 'recognition':
                    self.handle_recognition_result(result)
                elif result_type == 'recognition_error':
                    self.handle_recognition_error(result)
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error in check_result_queue: {str(e)}")
            self.msg_box('ERROR', f'Error in check_result_queue: {str(e)}', 'error')
            play_error_escaneo()
        self.root.after(100, self.check_result_queue)  # Revisa la cola cada 100 ms

    def handle_antispoofing_result(self, result):
        try:
            if result != 1:
                self.handle_spoofing_error()
            else:
                threading.Thread(target=self.run_face_recognition_thread).start()
        except Exception as e:
            logging.error(f"Error handling antispoofing result: {str(e)}")
            self.msg_box('ERROR', f'Error handling antispoofing result: {str(e)}', 'error')
            play_error_escaneo()

    def handle_spoofing_error(self):
        self.msg_box('Imagen Falsa', 'No use fotos o intentlo de nuevo.', 'error')
        play_error_escaneo()

    def run_face_recognition_thread(self):
        try:
            rfc = self.run_face_recognition()
            self.result_queue.put(('recognition', rfc))
        except Exception as e:
            logging.error(f"Error in run_face_recognition_thread: {str(e)}")
            self.result_queue.put(('recognition_error', str(e)))

    def run_face_recognition(self):
        try:
            return recognize_with_embeddings(self.most_recent_capture_arr, self.embeddings_dict)
        except Exception as e:
            raise RuntimeError(f'Error en el reconocimiento facial: {str(e)}')

    def handle_antispoofing_error(self, error_message):
        logging.error(f"Error in antispoofing: {error_message}")
        self.msg_box('ERROR', f'Error en el modelo antispoofing: {error_message}', 'error')
        play_error_escaneo()

    def handle_recognition_result(self, result):
        try:
            if result in ['unknown_person', 'no_persons_found']:
                self.handle_recognition_error()
            else:
                self.process_recognition_result(result)
        except Exception as e:
            logging.error(f"Error handling recognition result: {str(e)}")
            self.msg_box('ERROR', f'Error handling recognition result: {str(e)}', 'error')
            play_error_escaneo()

    def handle_recognition_error(self, error_message=None):
        self.recognition_error_count += 1
        if self.recognition_error_count >= 2:
            self.msg_box('Ups...', 'No registrado', 'error')
            play_error_escaneo()
            self.register_facial_entry(None, 'No registrado', False)
            self.recognition_error_count = 0

    def process_recognition_result(self, rfc):
        try:
            schedule_type = self.get_schedule_type(rfc)
            print(f"Usuario: {rfc}, Tipo de Horario: {schedule_type}")  # Impresión adicional
            if not schedule_type:
                self.register_facial_entry(rfc, 'Error de Horario', False)
                return
            if schedule_type == 'Abierto':
                self.handle_open_schedule(rfc)
            elif schedule_type == 'Cerrado':
                self.handle_closed_schedule(rfc)
        except Exception as e:
            self.handle_unexpected_error(e)

    def get_schedule_type(self, rfc):
        try:
            success, result = get_employee_schedule_type(self.db, rfc)
            if not success:
                self.msg_box('ERROR', result, 'error')
                play_error_sound()
                return None
            return result
        except Exception as e:
            error_message = f'Error al obtener el tipo de horario: {str(e)}'
            self.msg_box('ERROR', error_message, 'error')
            play_error_escaneo()
            return None

    def handle_open_schedule(self, rfc):
        try:
            current_time = datetime.now()
            aviso = self.db.get_collection('avisos').find_one({
                'RFC': rfc,
                'Fecha_inicial': {'$lte': current_time},
                'Fecha_Final': {'$gte': current_time}
            })

            message = add_open_schedule_check(self.db, rfc, "entrada")
            
            # Verificar si el mensaje es una tupla y extraer el mensaje real si es necesario
            if isinstance(message, tuple):
                message = message[0]

            # Primero verificamos si es una entrada ya registrada
            if "ya registrada" in message.lower():
                play_ya_scaneado()
                self.msg_box('Registro de Asistencia', message, 'error')
                return

            # Determinamos el tipo de mensaje y reproducimos el sonido correspondiente
            if "entrada" in message.lower() or "bienvenido" in message.lower():
                self.msg_box('Registro de Asistencia', message, 'éxito')
                play_normal_sound()
            elif "salida" in message.lower():
                self.msg_box('Registro de Asistencia', message, 'éxito')
                play_sa_normal()

            # Después mostramos el aviso si existe
            if aviso and 'mensaje_administrador' in aviso:
                self.show_warning_message(aviso['mensaje_administrador'], duration=7, color='gray')

        except Exception as e:
            logging.error(f"Error al registrar entrada abierta: {str(e)}")
            self.msg_box('ERROR', f'Error al registrar entrada abierta: {str(e)}', 'error')
            play_error_escaneo()
            self.register_facial_entry(rfc, 'Entrada Fallida', False)


    def handle_closed_schedule(self, rfc):
        try:
            resultado = verificar_y_actualizar_horario_fechas(self.db, rfc)
            if isinstance(resultado, tuple):
                estatus, action_type = resultado
                print(f"Usuario: {rfc}, Estatus: {estatus}, Acción: {action_type}")  # Impresión adicional
                self.handle_status_messages(rfc, estatus, action_type)
            else:
                self.msg_box('Error', resultado, 'error')
        except Exception as e:
            logging.error(f"Error al verificar horario cerrado: {str(e)}")
            self.msg_box('ERROR', f'Error al verificar horario cerrado: {str(e)}', 'error')
            play_error_escaneo()
            self.register_facial_entry(rfc, 'Entrada Fallida', False)

    def handle_status_messages(self, rfc, estatus, action_type):
        status_messages = {
            "NORMAL": {
                "entrada": f"Bienvenido {rfc}, llegaste a tiempo, asistencia tomada.",
                "salida": f"Hasta luego {rfc}, salida registrada a tiempo."
            },
            "RETARDO": {
                "entrada": f"¡CASI! {rfc}, llegaste un poco tarde, asistencia tomada con retardo.",
                "salida": f"¡CUIDADO! {rfc}, has salido tarde."
            },
            "NOTA MALA": {
                "entrada": f"¡UPSS! {rfc}, esta vez tienes nota mala, llegaste tarde.",
                "salida": f"¡ALERTA! {rfc}, has salido mucho más tarde de lo previsto."
            }
        }
        message_types = {
            "NORMAL": "éxito",
            "RETARDO": "retardo",
            "NOTA MALA": "fueraderango"
        }
        message = status_messages.get(estatus, {}).get(action_type, "Ya escaneado o fuera de rango.")
        message_type = message_types.get(estatus, "error")

        if message == "Ya escaneado o fuera de rango.":
            play_ya_scaneado()
        elif message == status_messages["NORMAL"]["entrada"]:
            play_normal_sound()
        elif message == status_messages["NORMAL"]["salida"]:
            play_sa_normal()
        elif message == status_messages["RETARDO"]["entrada"]:
            play_retardo_sound()
        elif message == status_messages["RETARDO"]["salida"]:
            play_sa_retardo()
        elif message == status_messages["NOTA MALA"]["entrada"]:
            play_nota_mala_sound()
        elif message == status_messages["NOTA MALA"]["salida"]:
            play_sa_notamala()

        self.msg_box('Registro de Asistencia', message, message_type)
        entry_success = estatus in ["NORMAL", "RETARDO"]
        entry_type = 'Entrada Exitosa' if entry_success else 'Entrada Fallida'
        self.register_facial_entry(rfc, entry_type, entry_success)

    def handle_unexpected_error(self, e):
        logging.error(f"Error inesperado: {str(e)}")
        self.msg_box('ERROR', f'Error inesperado: {str(e)}', 'error')
        play_error_escaneo()
        self.register_facial_entry(None, 'Entrada Fallida', False)



#----------------------------------------------------------------------------------------------------------#
"""_create_window_
    Inicializa la ventana principal para la aplicación del Instituto Tecnológico de Tuxtepec.
    Configura el diseño, el encabezado, la alimentación de video y las secciones para mostrar mensajes.
"""
def create_window():
    global external_process
    
    # Create the main window
    root = tk.Tk()
    root.state('zoomed')  
    root.title("Instituto Tecnológico de Tuxtepec")
    root.attributes('-topmost', True)

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()


    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)

 
    position_x = int((screen_width - window_width) / 2)
    position_y = int((screen_height - window_height) / 2)


    root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")


    root.grid_rowconfigure(1, weight=1) 
    root.grid_columnconfigure(0, weight=45, minsize=window_width*0.45) 
    root.grid_columnconfigure(1, weight=50, minsize=window_width*0.50) 

   
    header_frame = tk.Frame(root, bg='white', height=50) 
    header_frame.grid(row=0, column=0, columnspan=2, sticky='ew')
    header_frame.grid_propagate(False)

     
    logo_image_left = load_resized_image('RECURSOS/imagenes/logo_ittux.png', (50, 50))
    logo_image_right = load_resized_image('RECURSOS/imagenes/LOGO_TECNM.png', (50, 50))

    # Logo izquierdo
    logo_label_left = tk.Label(header_frame, image=logo_image_left, bg='white')
    logo_label_left.pack(side='left', padx=10)

    # Logo derecho
    logo_label_right = tk.Label(header_frame, image=logo_image_right, bg='white')
    logo_label_right.pack(side='right', padx=10)

    # Institute name label - centrado en el medio
    name_label = tk.Label(header_frame, text='TECNOLÓGICO NACIONAL DE MÉXICO  CAMPUS TUXTEPEC ', bg='white', fg='black', font=('Roboto', 20))
    name_label.pack(expand=True)

    def redraw_gradient(canvas, start_color, end_color):
        canvas.delete("gradient") 
        width = canvas.winfo_width()  
        create_gradient(canvas, start_color, end_color, width)  

   
    gradient_canvas = Canvas(header_frame, bg='white', height=10, bd=0, highlightthickness=0)
    gradient_canvas.pack(fill='x', side='bottom')

   
    create_gradient(gradient_canvas, 'green', 'blue', gradient_canvas.winfo_reqwidth())


    gradient_canvas.bind("<Configure>", lambda event, canvas=gradient_canvas, start_color='green', end_color='blue': redraw_gradient(canvas, start_color, end_color))

    logo_label_left.image = logo_image_left
    logo_label_right.image = logo_image_right

    # Central Frame Configuration
    central_frame = tk.Frame(root, bg='white', bd=2, relief='groove')
    central_frame.grid(row=1, column=0, sticky='nswe')
    central_frame.grid_propagate(False)
    central_frame.grid_columnconfigure(0, weight=1)
    central_frame.grid_rowconfigure(0, weight=1)
    central_frame.grid_columnconfigure(0, weight=1)


    # Frame para la cámara
    top_left_frame = tk.Frame(central_frame, bg='green', bd=2, relief='groove')
    top_left_frame.grid(row=0, column=0, sticky='nsew')  # Este frame es el del video
    top_left_frame.grid_propagate(False)
    top_left_frame.grid_rowconfigure(0, weight=1)
    top_left_frame.grid_columnconfigure(0, weight=1)
 
    # Label para mostrar el video de la cámara
    webcam_label = tk.Label(top_left_frame)
    webcam_label.grid(row=0, column=0, sticky="nsew")
    webcam_label.pack_propagate(False)
    
    # Mantener el tamaño mínimo para el frame y el label
    min_width, min_height = 640, 480
    webcam_label.config(width=min_width, height=min_height)
    top_left_frame.config(width=min_width, height=min_height)


    def resize_image(event):
        # Obtener el tamaño del frame
        frame_width, frame_height = event.width, event.height
        # Calcular la proporción de la imagen
        ratio = min(frame_width/min_width, frame_height/min_height)
        # Redimensionar la imagen
        new_width, new_height = int(min_width * ratio), int(min_height * ratio)
        webcam_label.config(width=new_width, height=new_height)
    
    # Vincular el evento de redimensionar al frame
    top_left_frame.bind('<Configure>', resize_image)



#------------------------------------------------------------------------------------------------------#

    bottom_left_frame = tk.Frame(central_frame, bg='#EFEFEF', bd=2, relief='groove')
    bottom_left_frame.grid(row=1, column=0, sticky='nswe')
    bottom_left_frame.grid_propagate(False)


    bottom_left_frame.grid_columnconfigure(0, weight=1)


    bottom_left_frame.grid_rowconfigure(0, minsize=60) 
    bottom_left_frame.grid_rowconfigure(1, minsize=30)  
    
    font_style = ('digital-7', 50)  
    time_label = tk.Label(bottom_left_frame, font=font_style, fg='black', bg='#EFEFEF')
    time_label.pack(side='top', fill='x', expand=False, pady=(10, 0))  
    
    date_font_style = ('Helvetica', 18) 
    date_label = tk.Label(bottom_left_frame, font=date_font_style, fg='black', bg='#EFEFEF')
    date_label.pack(side='top', fill='x', expand=True, pady=(5, 10))  

 
    update_time(time_label, root)
    update_date(date_label)

 
    right_frame = tk.Frame(root, bg='white', bd=2, relief='groove')
    right_frame.grid(row=1, column=1, sticky='nswe')
    right_frame.grid_propagate(False)

    right_frame = tk.Frame(root, bg='white', bd=2, relief='groove')
    right_frame.grid(row=1, column=1, sticky='nswe')
    right_frame.grid_columnconfigure(0, weight=1)  

    """# Configura las filas del right_frame para las secciones y los separadores
    right_frame.grid_rowconfigure(0, weight=10)  # 10% altura para la primera sección
    right_frame.grid_rowconfigure(1, weight=1)   # Pequeño peso para el primer separador
    right_frame.grid_rowconfigure(2, weight=55)  # 55% altura para la segunda sección
    right_frame.grid_rowconfigure(3, weight=1)   # Pequeño peso para el segundo separador
    right_frame.grid_rowconfigure(4, weight=10)  # 10% altura para la tercera sección
    right_frame.grid_rowconfigure(5, weight=1)   # Pequeño peso para el tercer separador
    right_frame.grid_rowconfigure(6, weight=25)  # 25% altura para la cuarta sección"""

    # Configuración de la fila para Sección 1
    right_frame.grid_rowconfigure(0, minsize=50, weight=10)  # Tamaño fijo para Sección 1

    # Configuración de la fila para el Separador 1
    right_frame.grid_rowconfigure(1, minsize=2, weight=0)  # Altura fija para el separador

    # Configuración de la fila para Sección 2
    right_frame.grid_rowconfigure(2, minsize=250, weight=55)  # Tamaño fijo para Sección 2

    # Configuración de la fila para el Separador 2
    right_frame.grid_rowconfigure(3, minsize=2, weight=0)  # Altura fija para el separador

    # Configuración de la fila para Sección 3
    right_frame.grid_rowconfigure(4, minsize=50, weight=10)  # Tamaño fijo para Sección 3

    # Configuración de la fila para el Separador 3
    right_frame.grid_rowconfigure(5, minsize=2, weight=0)  # Altura fija para el separador

    # Configuración de la fila para Sección 4
    right_frame.grid_rowconfigure(6, minsize=150, weight=25)  # Tamaño fijo para Sección 4

    # Añade los separadores
    separator1 = ttk.Separator(right_frame, orient='horizontal')
    separator1.grid(row=1, column=0, sticky='ew')
    

    separator2 = ttk.Separator(right_frame, orient='horizontal')
    separator2.grid(row=3, column=0, sticky='ew')

    separator3 = ttk.Separator(right_frame, orient='horizontal')
    separator3.grid(row=5, column=0, sticky='ew')


    section1_frame = tk.Frame(right_frame, bg='#079073') 
    section1_frame.grid(row=0, column=0, sticky='nswe')
    section1_label = tk.Label(section1_frame, text='AVISOS', bg='#079073', fg='black', anchor='center', font=('Roboto', 20))  
    section1_label.pack(expand=True, fill='both')  


    section2_frame = tk.Frame(right_frame, bg='#EFEFEF')  
    section2_frame.grid(row=2, column=0, sticky='nswe')
    #section2_label = tk.Label(section2_frame, text='SIN NOVEDAD', bg='#EFEFEF', fg='black', anchor='center', font=('Roboto', 20))  
    #section2_label.pack(expand=True, fill='both')  


    section3_frame = tk.Frame(right_frame, bg='#CFF2EA') 
    section3_frame.grid(row=4, column=0, sticky='nswe')
    section3_label = tk.Label(section3_frame, text='ORGULLOSANTE TECNM', bg='#CFF2EA', fg='black', anchor='center', font=('Roboto', 20)) 
    section3_label.pack(expand=True, fill='both') 


    section4_frame = tk.Frame(right_frame, bg='#EFEFEF')
    section4_frame.grid(row=6, column=0, sticky='nswe')
    

    section4_frame.grid_columnconfigure(0, weight=1)
    section4_frame.grid_columnconfigure(1, weight=0)  
    section4_frame.grid_columnconfigure(2, weight=1)
    section4_frame.grid_rowconfigure(0, weight=1)


    separator4 = ttk.Separator(section4_frame, orient='vertical')
    separator4.grid(row=0, column=1, sticky='ns')

    """
    Maneja el evento de cierre de la ventana principal.
    Termina cualquier proceso externo en ejecución y libera los recursos de
    la cámara antes de destruir la ventana.
    """
    
    def on_close():
        global external_process

        if external_process is not None:
            external_process.terminate()
            external_process.wait()

        
        if app_instance.cap.isOpened():
            app_instance.cap.release() 

        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)


    app_instance = App(root, top_left_frame, section2_frame, section4_frame)
    app_instance.check_result_queue() 
    
    app_instance.check_for_messages()


    try:
      external_process = subprocess.Popen(["DemoDP4500_k_1/DemoDP4500/bin/Debug/DemoDP4500.exe", "verificar"])
        
    except Exception as e:
        print(f"Failed to start external process: {e}")
        external_process = None
   
    root.mainloop()

if __name__ == "__main__":
    pygame.mixer.init()
    create_window()
