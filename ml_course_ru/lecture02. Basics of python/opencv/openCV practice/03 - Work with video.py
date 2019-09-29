import cv2

# need to FFMPEG CODEC
video_stream = cv2.VideoCapture('Road.mp4')
width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_stream.get(cv2.CAP_PROP_FPS))

print('VIDEO FRAME SHAPE')
print('height', height)
print('width', width)
print('fps', fps)

frame_exist = True

while frame_exist:
    frame_exist, frame = video_stream.read()
    if frame_exist:
        frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(frame, 'Video capture!!!!',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow('video', frame)
        cv2.waitKey(40)

cv2.destroyAllWindows()
