import tkinter as tk
import cv2
import threading
import numpy as np

class Aplicacao:
    def __init__(self, janela):
        self.count_wheel = None
        self.scroll_para_cima_ativado = None
        self.retangulo_y = None
        self.retangulo_x = None
        self.wheel = None
        self.frame = None
        self.height = None
        self.width = None
        self.channels = None


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
        self.frame = cv2.imread(r"/home/sinmec/imgs/out_000001.jpg")
        self.rodar_imagem(self.frame)

    def rodar_imagem(self, img):
        #self.create_rectangle(img, self.mouse_x, self.mouse_y)
        self.retangulo_x = self.mouse_x
        self.retangulo_y = self.mouse_y
        count = 1
        img_rectangle = img
        while True:
            if self.wheel == 4:
                self.retangulo_x = self.mouse_x
                self.retangulo_y = self.mouse_y
                count += 0.2
                self.wheel = None

                if (self.retangulo_x + (216 / count) / 2) > 216:
                    self.retangulo_x = self.retangulo_x - ((self.retangulo_x + (216 / count) / 2) - 216)
                if (self.retangulo_x - (216/count)/2) < 0:
                    self.retangulo_x = self.retangulo_x + (0 - (self.retangulo_x - ((216/count) / 2)))

                if (self.retangulo_y + (1024 / count) / 2) > 1024:
                    self.retangulo_y = self.retangulo_y - ((self.retangulo_y + (1024 / count) / 2) - 1024)
                if (self.retangulo_y - (1024/count)/2) < 0:
                    self.retangulo_y = self.retangulo_y + (0 - (self.retangulo_y - ((1024/count) / 2)))

                img_rectangle = self.criar_retangulo(img, self.retangulo_x, self.retangulo_y, 216 / count,
                                                         1024 / count)
                self.atualizar_video(img_rectangle)
            elif self.wheel == 5:
                self.retangulo_x = self.mouse_x
                self.retangulo_y = self.mouse_y
                count -= 0.2
                if count < 0:
                    count = 1
                self.wheel = None

                if (self.retangulo_x + (216/count)/2) > 216:
                        self.retangulo_x = self.retangulo_x - (( self.retangulo_x + (216/count)/2) - 216)
                if (self.retangulo_x - (216/count)/2) < 0:
                    self.retangulo_x = self.retangulo_x + (0 - (self.retangulo_x - ((216/count) / 2)))

                if (self.retangulo_y + (1024 / count) / 2) > 1024:
                    self.retangulo_y = self.retangulo_y - ((self.retangulo_y + (1024 / count) / 2) - 1024)
                if (self.retangulo_y - (1024/count)/2) < 0:
                    self.retangulo_y = self.retangulo_y + (0 - (self.retangulo_y - ((1024/count) / 2)))

                img_rectangle = self.criar_retangulo(img, self.retangulo_x, self.retangulo_y, 216 / count,
                                                         1024 / count)
                self.atualizar_video(img_rectangle)
            else:
                self.atualizar_video(img_rectangle)



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

    def criar_retangulo(self, imagem, x, y, largura, altura):
        imagem_com_retangulo = imagem.copy()
        cor = (0, 255, 0)
        espessura = 2

        x1 = int(x - largura / 2)
        y1 = int(y - altura / 2)
        x2 = int(x + largura / 2)
        y2 = int(y + altura / 2)

        cv2.rectangle(imagem_com_retangulo, (x1, y1), (x2, y2), cor, espessura)

        return imagem_com_retangulo

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
        if event.num == 4:
            print('zoom para cima')
        elif event.num == 5:
            print('zoom para baixo')

        self.wheel = event.num

    def on_mouse_motion(self, event):
        # Atualiza a posição do mouse

        self.mouse_x = event.x
        self.mouse_y = event.y
        print(self.mouse_x, self.mouse_y)




if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacao(root)

    # Associe a rolagem do mouse ao método de zoom
    app.video_canvas.bind("<Button-4>", app.on_mouse_wheel)
    app.video_canvas.bind("<Button-5>", app.on_mouse_wheel)
    app.video_canvas.bind("<Motion>", app.on_mouse_motion)
    app.video_canvas.bind("<Button-1>", app.on_mouse_wheel)




    root.mainloop()