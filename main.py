import cv2, face_recognition, aiofiles
from scipy.spatial.distance import pdist
from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.post("/")
async def check(img1: UploadFile, img2: UploadFile):
    async with aiofiles.open(img1.filename, 'wb') as out_file:
        content = await img1.read()
        await out_file.write(content)
    async with aiofiles.open(img2.filename, 'wb') as out_file:
        content = await img2.read()
        await out_file.write(content)
    img1 = cv2.imread(img1.filename)
    img2 = cv2.imread(img2.filename)
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
    return result
