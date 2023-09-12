import tkinter as tk
import cv2
import threading
import numpy as np

class Aplicacao:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title("Aplicacao OpenCV + Tkinter")

        self.frame = tk.Frame(self.janela)
        self.frame.pack()

        self.status_label = tk.Label(self.frame, text="Status: Aguardando...")
        self.status_label.pack()

        # Crie uma área para exibir o vídeo (ou imagem) usando OpenCV
        self.video_canvas = tk.Canvas(self.frame, width=216, height=1024) #TODO: Implement dynamic canvas size
        self.video_canvas.pack()

        self.zoom = 1.0

        self.mouse_x = 0
        self.mouse_y = 0

        # Inicie a thread para executar o OpenCV
        self.iniciar_thread_opencv()

    def iniciar_thread_opencv(self):
        self.thread_opencv = threading.Thread(target=self.executar_opencv)
        self.thread_opencv.daemon = True
        self.thread_opencv.start()

    def executar_opencv(self):
        frame = cv2.imread(r"/home/sinmec/imgs/out_000001.jpg")
        # W, H = frame.shape
        # cap = cv2.VideoCapture(0)
        #
        while True:
            self.atualizar_video(frame)
        #     ret, frame = cap.read()
        #     if not ret:
        #         continue
        #
        #     # Realize o zoom na imagem com base no valor de self.zoom
        #     frame = self.aplicar_zoom(frame, self.zoom)
        #     if (self.zoom) > 1.1:
        #         print(self.zoom)
        #
        #     # Atualize o vídeo no canvas do Tkinter
        #     self.atualizar_video(frame)

    def aplicar_zoom(self, frame, zoom):
        altura, largura, _ = frame.shape

        # Calcule as novas dimensões da imagem com base no zoom
        nova_altura = int(altura * zoom)
        nova_largura = int(largura * zoom)

        # if zoom < 1.01:
        #     return frame

        # Calcule as coordenadas do canto superior esquerdo da ROI
        x1 = int(self.mouse_x - nova_largura / 2)
        y1 = int(self.mouse_y - nova_altura / 2)

        # Garanta que a ROI esteja dentro dos limites da imagem
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(largura, x1 + nova_largura)
        y2 = min(altura, y1 + nova_altura)

        # Recorte a ROI da imagem original
        roi = frame[y1:y2, x1:x2]

        # Redimensione a ROI
        frame_zoom = cv2.resize(roi, (largura, altura))

        return frame_zoom

    def atualizar_video(self, frame):
        # Converte o frame do OpenCV para um formato que o Tkinter possa exibir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem = tk.PhotoImage(data=cv2.imencode('.png', frame_rgb)[1].tobytes())

        # Atualize o canvas do Tkinter
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imagem)
        self.video_canvas.imagem = imagem  # Mantém uma referência para evitar a coleta de lixo

    def on_mouse_wheel(self, event):
        # Verifica se a tecla Ctrl está pressionada e atualiza o zoom com base na rolagem do mouse
        # if event.state == 4:  # 4 é o valor para a tecla Ctrl
        if event.delta > 0:
            self.zoom += 0.1  # Zoom in
        else:
            self.zoom -= 0.1  # Zoom out

        print('eu to zoomando!!!')
        self.zoom = max(0.1, self.zoom)  # Limita o zoom mínimo
        self.zoom = min(2.0, self.zoom)  # Limita o zoom máximo

    def on_mouse_motion(self, event):
        # Atualiza a posição do mouse

        self.mouse_x = event.x
        self.mouse_y = event.y

        print(self.mouse_x, self.mouse_y)

if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacao(root)

    # Associe a rolagem do mouse ao método de zoom
    # app.video_canvas.bind("<MouseWheel>", app.on_mouse_wheel)
    app.video_canvas.bind("<Motion>", app.on_mouse_motion)




    root.mainloop()