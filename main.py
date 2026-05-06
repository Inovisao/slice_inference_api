import cv2 

def verifica_recorte(altura, largura, stride):
    if altura % stride == 0 or largura % stride == 0:
        return True
    return False

TILES_SIZE = 640
OVERLAP = 18 #porcentagem

img = cv2.imread("1.jpg")
altura = img.shape[0]
largura = img.shape[1]

print(f"Altura: {altura}, Largura: {largura}") #wxh

OVERLAP_px = TILES_SIZE * (OVERLAP/100) 
stride = TILES_SIZE - OVERLAP_px

print(f"Overlap (px): {OVERLAP_px}, Stride: {stride}") #wxh

n_tiles_largura = largura/TILES_SIZE
print(f"Maior número de tiles para a largura: {int(n_tiles_largura)}")

print(largura % stride)

if verifica_recorte(altura, largura, stride) == False:
    overlap_adc= TILES_SIZE - (largura % stride)
    overlap_final = (OVERLAP_px + overlap_adc) / n_tiles_largura
    new_stride = TILES_SIZE - overlap_final
    
    print(f"Overlap Adicional: {overlap_adc} px"),
    print(f"Overlap Final (px): {overlap_final} px"), 
    print(f"Stride: {new_stride}") #wxh
    verifica_recorte(altura,largura, new_stride)
