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

motion_detector = cv2.createBackgroundSubtractorKNN()

frame_exist = True


def get_only_big_contours(contours):
    biggest_contours = []
    area_threshold = 120

    for c in contours:
        if cv2.contourArea(c) > area_threshold:
            biggest_contours.append(c)

    return biggest_contours


def get_bounding_rects(contours):
    rects = []
    for c in contours:
        rect = cv2.boundingRect(c)
        rects.append(rect)
    return rects

def draw_rects(source_image, rects):
    image = source_image.copy()
    for r in rects:
        (x, y, w, h) = r
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(image, p1, p2, (0, 200, 0), 1)
    return image

while frame_exist:
    frame_exist, frame = video_stream.read()
    if frame_exist:
        frame = cv2.resize(frame, dsize=(640, 360), interpolation=cv2.INTER_CUBIC)

        # is not necessary but next work with gray image will be faster
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        foreground_mask = motion_detector.apply(frame_gray)

        frame_m_contours, m_contours, contours_hierarchy = cv2.findContours(foreground_mask,
                                                                            cv2.RETR_EXTERNAL,
                                                                            cv2.CHAIN_APPROX_SIMPLE)

        # note that drawContours get a frame copy instead original object
        # because function don't copy object and change object by reference
        frame_contours = cv2.drawContours(frame.copy(), m_contours, -1, (0, 0, 200), 2)

        biggest_contours = get_only_big_contours(m_contours)
        object_rects = get_bounding_rects(biggest_contours)
        frame_objects = draw_rects(frame, object_rects)

        # drawing all results
        cv2.imshow('video', frame)
        cv2.imshow('foreground', foreground_mask)
        cv2.imshow('contours', frame_contours)
        cv2.imshow('objects', frame_objects)
        cv2.waitKey(50)

cv2.destroyAllWindows()
