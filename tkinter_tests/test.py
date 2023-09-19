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
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

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

    def criar_segunda_janela(self):
        segunda_janela = tk.Toplevel(self.janela)

        print('Estou abrindo uma janela')

        self.segunda_janela = segunda_janela
        self.segunda_janela.title('Zoom')
        self.frame2 = tk.Frame(self.segunda_janela)

        self.status_label_2 = tk.Label(self.segunda_janela, text="Status: Aguardando...")
        self.status_label_2.pack()

        # Crie uma área para exibir o vídeo (ou imagem) usando OpenCV
        self.video_canvas_2 = tk.Canvas(self.segunda_janela, width=216, height=1024) #TODO: Implement dynamic canvas size
        self.video_canvas_2.pack()




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
        self.criar_segunda_janela()
        while True:
            if self.wheel == 4:
                self.retangulo_x = self.mouse_x
                self.retangulo_y = self.mouse_y
                count += 0.2
                self.wheel = None

                self.retangulo_x = self.check_borders(216, 1024, self.retangulo_x, self.retangulo_y, count)[0]
                self.retangulo_y = self.check_borders(216, 1024, self.retangulo_x, self.retangulo_y, count)[1]

                img_rectangle = self.criar_retangulo(img, self.retangulo_x, self.retangulo_y, 216 / count,
                                                         1024 / count)
                self.atualizar_video(img_rectangle, img)
            elif self.wheel == 5:
                self.retangulo_x = self.mouse_x
                self.retangulo_y = self.mouse_y
                count -= 0.2
                if count < 0:
                    count = 1
                self.wheel = None

                self.retangulo_x = self.check_borders(216, 1024, self.retangulo_x, self.retangulo_y, count)[0]
                self.retangulo_y = self.check_borders(216, 1024, self.retangulo_x, self.retangulo_y, count)[1]


                img_rectangle = self.criar_retangulo(img, self.retangulo_x, self.retangulo_y, 216 / count,
                                                         1024 / count)
                self.atualizar_video(img_rectangle, img)
            else:
                self.atualizar_video(img_rectangle, img)




    def check_borders(self, width, height, x, y, count):
        if (x + (width / count) / 2) > width:
            x = x - ((x + (width / count) / 2) - width)
        if (x - (width / count) / 2) < 0:
            x = x + (0 - (x - ((width / count) / 2)))

        if (y + (height / count) / 2) > height:
            y = y - ((y + (height / count) / 2) - height)
        if (y - (height / count) / 2) < 0:
            y = y + (0 - (y - ((height / count) / 2)))
        return x, y
        
    def criar_retangulo(self, imagem, x, y, largura, altura):
        imagem_com_retangulo = imagem.copy()
        cor = (0, 255, 0)
        espessura = 2

        x1 = int(x - largura / 2)
        y1 = int(y - altura / 2)
        x2 = int(x + largura / 2)
        y2 = int(y + altura / 2)

        self.x1 = max(0, x1)
        self.y1 = max(0, y1)
        self.x2 = min(216, x2)
        self.y2 = min(1024, y2)

        cv2.rectangle(imagem_com_retangulo, (self.x1, self.y1), (self.x2, self.y2), cor, espessura)

        return imagem_com_retangulo


    def atualizar_video(self, frame_rectangle, frame):
        # Converte o frame do OpenCV para um formato que o Tkinter possa exibir
        frame_rgb = cv2.cvtColor(frame_rectangle, cv2.COLOR_BGR2RGB)
        imagem = tk.PhotoImage(data=cv2.imencode('.png', frame_rgb)[1].tobytes())

        # Atualize o canvas do Tkinter
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imagem)
        self.video_canvas.imagem = imagem  # Mantém uma referência para evitar a coleta de lixo

        frame_rgb_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_zoom = frame_rgb_2[self.y1:self.y2, self.x1:self.x2]

        resize = cv2.resize(frame_rgb_zoom, (216, 1024))

        image_2 = tk.PhotoImage(data=cv2.imencode('.png', resize)[1].tobytes())

        self.video_canvas_2.create_image(0, 0, anchor=tk.NW, image=image_2)
        self.video_canvas_2.imagem = image_2  # Mantém uma referência para evitar a coleta de lixo



    def on_mouse_wheel(self, event):
        #if event.num == 4:
            #print('zoom para cima')
        #elif event.num == 5:
            #print('zoom para baixo')

        self.wheel = event.num

    def on_mouse_motion(self, event):
        # Atualiza a posição do mouse

        self.mouse_x = event.x
        self.mouse_y = event.y
        #print(self.mouse_x, self.mouse_y)




if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacao(root)

    # Associe a rolagem do mouse ao método de zoom
    app.video_canvas.bind("<Button-4>", app.on_mouse_wheel)
    app.video_canvas.bind("<Button-5>", app.on_mouse_wheel)
    app.video_canvas.bind("<Motion>", app.on_mouse_motion)
    app.video_canvas.bind("<Button-1>", app.on_mouse_wheel)

    root.mainloop()