import os
import pathlib
from pyexpat import model
import string
from tarfile import GNUTYPE_LONGLINK

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time

from importlib.util import LazyLoader
from multiprocessing import connection
from multiprocessing.sharedctypes import Value
from optparse import Values
import mysql.connector
import operator

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk



def decode_audio(audio_binary):
  # dekodowanie plików WAV na float 32 [-1;1]
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  #drop the `channels` - single channel
 
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

def plot_waveform4all():
  rows = 3
  cols = 3
  n = rows * cols
  fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

  for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

def logowanie(loginIN, hasloIN):
    global dane_uzytkownik
    mycursor = mydb.cursor()
    sql_login= "SELECT login FROM uzytkownik WHERE login = \"{}\" and haslo=\"{}\"". format(loginIN,hasloIN)
    mycursor.execute(sql_login)
    login_poprawne= mycursor.fetchall()

    if login_poprawne:
        pobierzDane(loginIN)
        return 1
    
    else:
        return 0

def pobierzDane(loginIN):
    global dane_uzytkownik
    mycursor = mydb.cursor()
    dane= "SELECT id,imie,nazwisko,login,haslo,data_edycji,dostep FROM uzytkownik WHERE login = \"{}\"". format(loginIN)
    mycursor.execute(dane)
    for row in mycursor:
        dane_uzytkownik= {"ID":row[0], "login":row[3], "haslo":row[4], "imie":row[1],"nazwisko":row[2],"data_edycji":row[5],"dostep":row[6]}
    
    pobierzKomendy()

def rejestracja(daneIN):

    mylist_records= ",".join(["%s"]*len(daneIN))
    cursor = mydb.cursor()
    sql =(f"INSERT INTO uzytkownik (imie,nazwisko,login, haslo) VALUES ({mylist_records})")
    
    cursor.execute(sql, daneIN)
    mydb.commit()
    pobierzDane(daneIN[2])

    return 1

def pobierzKomendy():
  global komendy_slownik
  komendy_slownik=[]
  mycursor = mydb.cursor()
  sql="SELECT komenda,polecenie FROM komendy WHERE id_uzytkownik = \"{}\"". format(dane_uzytkownik["ID"])
  mycursor.execute(sql)

  for komenda, polecenie in mycursor:
   polecenieDict={"komenda":komenda,"polecenie":polecenie}
   komendy_slownik.append(polecenieDict)

def edycjaKomend(tryb, komenda, polecenie=""):
  
  if tryb=="Dodaj":    
      mycursor = mydb.cursor()
      sql="INSERT INTO komendy (id_uzytkownik, komenda, polecenie) VALUES(\"{}\",\"{}\",\"{}\")". format(dane_uzytkownik["ID"], komenda, polecenie)
      mycursor.execute(sql)
  elif tryb=="Popraw":
      mycursor = mydb.cursor()
      sql="UPDATE komendy set polecenie = \"{}\" where id_uzytkownik=\"{}\" and komenda = \"{}\"". format(polecenie,dane_uzytkownik["ID"],komenda)
      mycursor.execute(sql)    
  elif tryb=="Usuń":
      mycursor = mydb.cursor()
      sql="DELETE from komendy where id_uzytkownik=\"{}\" and komenda = \"{}\"". format(dane_uzytkownik["ID"],komenda)
      mycursor.execute(sql)
  
  mydb.commit()
  pobierzKomendy()


def bttn(slf, x, y, text, ecolor, lcolor, func):
    def on_entera(e):
        myButton1['background'] = ecolor  # ffcc66
        myButton1['foreground'] = lcolor  # 000d33

    def on_leavea(e):
        myButton1['background'] = lcolor
        myButton1['foreground'] = ecolor

    myButton1 = tk.Button(slf, text=text,
                       width=25,
                       height=2,
                       fg=ecolor,
                       border=0,
                       bg=lcolor,
                       activeforeground=lcolor,
                       activebackground=ecolor,
                       command=func)

    myButton1.bind("<Enter>", on_entera)
    myButton1.bind("<Leave>", on_leavea)

    myButton1.pack(pady=10)

def wykonajPolecenie():

    global  model
    freq = 16000
    duration = 2
    sciezka='C:\\Users\\igisk\\Desktop\\xxx\\XX\\recording1.wav'
    recording = sd.rec(int(duration * freq), 
                       samplerate=freq, channels=1, dtype='int16')

    sd.wait()
    # numpy2audio
    write("recording0.wav", freq, recording)
    wv.write("recording1.wav", recording, freq, sampwidth=2)

    time.sleep(1)

    try:
      
        sample_file = sciezka

        sample_ds = preprocess_dataset([str(sample_file)])

        for spectrogram, label in sample_ds.batch(1):
          prediction = model(spectrogram)
          print(prediction)


          pred=prediction        
    except:
        print("Exception in wykonajPolecenie() 1!")

    try:
        i=0
        for x in komendyProb:
          aa=str(prediction[0][i])

          komendyProb[x]=aa[10:17]
          i+=1

    except:
        print("Exception in wykonajPolecenie() 2!")

    maxValue=max(komendyProb.items(), key=operator.itemgetter(1))[0]
    print("Największa wartość:"+maxValue)
    
    mycursor = mydb.cursor()
    sql="SELECT polecenie FROM komendy WHERE id_uzytkownik = \"{}\" and komenda=\"{}\"". format(dane_uzytkownik["ID"],maxValue)
    mycursor.execute(sql)
    sciezka_polecenia=mycursor.fetchall()

    for k in sciezka_polecenia:
        print(k)
        os.startfile(k[0])


#XXX DB
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  database = "project1"
)

komendy_slownik=[]
dane_uzytkownik=[]

komendyProb={'down':0, 'go':0, 'left':0, 'no':0, 'right':0,'stop':0, 'up':0, 'yes':0}


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.place_forget()
            self._frame.pack_forget()
            self._frame.grid_forget()
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class StartPage(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self, master)
        
        tk.Label(self, text="Strona Powitalna").pack(side="top", fill="x", pady=10)

        bttn(self, 100, 345, 'Logowanie', 'white', '#227999', lambda:master.switch_frame(LoginPage))
        bttn(self, 100, 445, 'Rejestracja', 'white', '#227999',lambda:master.switch_frame(SignUpPage))
        


class LoginPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        tk.Label(self, text="LOGIN").pack(side="top", fill="x", pady=10)
        
        tk.Label(self, text='Username', font=('Calibri', 13)).pack()
        e1 = tk.Entry(self, width=20, border=0, font=('Consolas', 13))
        e1.pack()

        tk.Label(self, text='Password', font=('Consolas', 13)).pack()
        e2 = tk.Entry(self, width=20, border=0, show='*', font=('Calibri', 13))
        e2.pack()


        def cmd():
            if logowanie(e1.get(),e2.get())==1:
                messagebox.showinfo("LOGIN SUCCESSFULLY", "         W E L C O M E        ")
                master.switch_frame(WorkingPage)
            else:
                messagebox.showwarning("LOGIN FAILED", "        PLEASE TRY AGAIN        ")


        bttn(self, 450, 345, 'L O G I N', 'white', '#227999', cmd)
        bttn(self, 450, 445, 'Powrot do strony powitalnej', 'white', '#227999', lambda: master.switch_frame(StartPage))

        

class WorkingPage(tk.Frame):
    

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        def cmd():
                messagebox.showinfo("Potwierdzenie", "         Słuchanie      ")
                wykonajPolecenie()
        def cmd2():
                dane_uzytkownik=[]
                master.switch_frame(StartPage)

        bttn(self, 200, 445, 'Wysłuchanie komendy', 'white', '#227999',cmd)
        bttn(self, 200, 545, 'Edycja komend', 'white', '#227999',lambda: master.switch_frame(EditCommandsPage))
        bttn(self, 200, 645, 'Wylogowanie', 'white', '#227999',cmd2)


class SignUpPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Rejestracja").pack(side="top", fill="x", pady=10)
        
        tk.Label(self, text='Username', bg='white', font=('Calibri', 13)).pack()
        e1 = tk.Entry(self, width=20, border=0, font=('Consolas', 13))
        e1.pack()

        tk.Label(self, text='Password', bg='white', font=('Consolas', 13)).pack()
        e2 = tk.Entry(self, width=20, border=0, show='*', font=('Calibri', 13))
        e2.pack()


        tk.Label(self, text='Imie', bg='white', font=('Consolas', 13)).pack()
        e3 = tk.Entry(self, width=20, border=0, font=('Calibri', 13))
        e3.pack()


        tk.Label(self, text='Nazwisko', bg='white', font=('Consolas', 13)).pack()
        e4 = tk.Entry(self, width=20, border=0, font=('Calibri', 13))
        e4.pack()


        def cmd():
            if rejestracja([e3.get(),e4.get(),e1.get(),e2.get()])==1:
                print(dane_uzytkownik)
                messagebox.showinfo("ZAREJESTROWANO POMYSLNIE", " Witamy "+ dane_uzytkownik["login"])
                master.switch_frame(WorkingPage)


        bttn(self,100, 345, 'Z A R E J E S T R U J', 'white', '#227999',cmd)
        bttn(self,100, 345, 'Powrot do strony powitalnej', 'white', '#227999',lambda: master.switch_frame(StartPage))

class EditCommandsPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        
        # Add a Treeview widget
        tree = ttk.Treeview(self, column=("c1", "c2"), show='headings', height=5)

        tree.column("# 1", anchor=CENTER)
        tree.heading("# 1", text="Komenda")
        tree.column("# 2", anchor=CENTER)
        tree.heading("# 2", text="Polecenie")

        for row in komendy_slownik:

            tree.insert('', 'end', text="1", values=(row["komenda"],row["polecenie"]))

        tree.pack()

        OPTIONS = [
        "Dodaj",
        "Usuń",
        "Popraw"
        ] 
        KOMENDY = [
        "left",
        "right",
        "up",
        "down",
        "stop",
        "yes",
        "go",
        "no"
        ]

        def callback(*args):
            if(mode.get()=="Usuń"):

                e2.config(state='disabled')
            else: 

                e2.config(state='normal')

        mode = StringVar(self)
        mode.set(OPTIONS[0]) 
        mode.trace("w", callback)
        komenda = StringVar(self)
        komenda.set("---")

        w = OptionMenu(self, mode, *OPTIONS)
        w.pack()

        # test do schowania
        tk.Label(self, text='Przypisanie czynności: ', font=('Calibri', 13)).pack()
        # tk.Label(self, text='Edycja komend ' + nazwisko, bg='grey', font=('Consolas', 13)).pack()

        tk.Label(self, text='Komenda', font=('Calibri', 10)).pack()
        e1 = OptionMenu(self, komenda, *KOMENDY)
        e1.pack()

        e2Label=tk.Label(self, text='Polecenie', font=('Consolas', 10)).pack()
        e2 = tk.Entry(self, width=20, border=0, font=('Calibri', 13))
        e2.pack()

        def cmd():
            edycjaKomend(mode.get(), komenda.get(), e2.get())
            master.switch_frame(WorkingPage)


        bttn(self,400, 345, 'Zapisz zmiany', 'white', '#227999',cmd )
        bttn(self,400, 345, 'Wróć do menu', 'white', '#227999',lambda: master.switch_frame(WorkingPage))


def trainModel():
    global commands
    global model
    global AUTOTUNE
    global waveform_ds
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    DATASET_PATH = 'data/mini_speech_commands'

    data_dir = pathlib.Path(DATASET_PATH)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir='.', cache_subdir='data')


    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    print('Commands:', commands)

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Number of examples per label:',
        len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
    print('Example file tensor:', filenames[0])

    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    test_file = tf.io.read_file(DATASET_PATH+'/down/0a9f9af7_nohash_0.wav')
    test_audio, _ = tf.audio.decode_wav(contents=test_file)
    test_audio.shape

    AUTOTUNE = tf.data.AUTOTUNE

    files_ds = tf.data.Dataset.from_tensor_slices(train_files)

    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)


    

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    print('Label:', label)
    print('Waveform shape:', waveform.shape)
    print('Spectrogram shape:', spectrogram.shape)
    print('Audio playback')
    display.display(display.Audio(waveform, rate=16000))



    spectrogram_ds = waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
        print('Input shape:', input_shape)
    num_labels = len(commands)

    #normalizacja ??
    norm_layer = layers.Normalization()

    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # sampling input
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 20
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    #
    #metrics = history.history
    #plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    #plt.legend(['loss', 'val_loss'])
    #plt.show()

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')



#trainModel()
app = SampleApp()
app.geometry('450x500')
app.title('DREAM TEAM PROJECT')
app.mainloop()
