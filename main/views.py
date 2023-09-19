import cv2
import face_recognition
from django.shortcuts import render
from scipy.spatial.distance import pdist


def main(request):
    result = ''
    fimg1 = ''
    fimg2 = ''
    if request.method == 'POST':
        img1 = request.FILES['img1']
        img2 = request.FILES['img2']
        fimg1 = f'media/{str(img1)}'
        fimg2 = f'media/{str(img2)}'
        with open(f'media/{str(img1)}', 'wb') as f:
            f.write(img1.read())
        with open(f'media/{str(img2)}', 'wb') as f:
            f.write(img2.read())
        img1 = cv2.imread(f'media/{str(img1)}')
        img2 = cv2.imread(f'media/{str(img2)}')
        vector1 = face_recognition.face_encodings(img1)[0]
        vector2 = face_recognition.face_encodings(img2)[0]
        a = pdist([vector1, vector2], 'euclidean')
        if a > 0.6:
            result = ({
                "PDIST": str(a),
                "RESULT": "Лица не совпадают"
            })
        else:
            result = ({
                "PDIST": str(a),
                "RESULT": "Один и тот же человек"
            })
    return render(request, 'index.html', context={'result': result, 'img1': fimg1, 'img2': fimg2})
