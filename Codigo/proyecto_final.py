import numpy as np
import cv2
import copy

def main():
    #cap = cv2.VideoCapture(0)
    cap= cv2.VideoCapture('y2mate.com-TownCentreXVID_rfkGy6dwWJs_360p.avi')
    # pip install opencv-contrib-python
    fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False) #Funcion para sustraccion de imagenes, existen varias.
    bandera = True


    primer_frame = 1
    while(bandera==True):
        if (primer_frame == 1):
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2] #propiedades de cuantas filas y columnas tiene la imagen gray, en este caso el alto y ancho
            imagen_acumulada = np.zeros((height, width), np.uint8) #una variable llamada imagen acumulada, donde np.zeros hace que cree un fondo negro de las dimensiones anteriores
            primer_frame = 0
        else:
            ret, frame = cap.read()  # lee el frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convierte a escala de grises

            fgmask = fgbg.apply(gray)  # remueve el fondo

            cv2.imshow('diff-bkgnd-frame', fgmask)     #metodo de diferencia de frame, muestra el metodo,resultado de la sustraccion de fondo
            cv2.imshow('frame', frame)                 #Muestra el frame que esta tomando

            # se aplica un umbral binario.
            thresh = 2
            maxValue = 1
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY) #si el pixel se pasa del umbral es 1 y si no es 0 en este caso.
            # muestra la imagen umbral
            #cv2.imwrite('diff-th1.jpg', th1)

            # agregar la imagen acumulada.
            imagen_acumulada = cv2.add(imagen_acumulada, th1)
            # muestra la imagen acumulada
            # cv2.imwrite('diff-accum.jpg', imagen_acumulada)



            # muestra el frame actual en escala de grises
            #cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # aplicar mapa de calor
    color_image = im_color = cv2.applyColorMap(imagen_acumulada, cv2.COLORMAP_JET)
    # mustra la iamgen del color de mapa
    #cv2.imwrite('diff-color.jpg', color_image)

    # cubrir el mapa de color a el primer frame
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # guardar imagen final
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # limpiar
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()