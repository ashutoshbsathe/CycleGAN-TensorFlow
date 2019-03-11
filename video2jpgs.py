import cv2
VIDEOA_DIR ='./pop1topop2/trainA'
VIDEOB_DIR = './pop1topop2/trainB'

def video2jpg(vid_dir):
    """
    Reads a video file called vid_dir/video.mp4 
    extracts and reshapes the images into 320x200 size and saves them as jpgs in the same vid_dir
    """
    vidcap = cv2.VideoCapture(vid_dir + '/video.mp4')
    success, image = vidcap.read()
    image = cv2.resize(image, (320, 200))
    count = 0
    while success:
        cv2.imwrite(vid_dir + '/frame{}.jpg'.format(count), image)
        success, image = vidcap.read()
        if not success:
            print("Read last frame, count {}".format(count))
        if count % 100 == 0:
            print("Read {} frames".format(count))
        count += 1
    print("Video ended")
    print('-'*25)
    return 

def main():
    video2jpg(VIDEOA_DIR)
    video2jpg(VIDEOB_DIR)
    return

if __name__ == '__main__':
    main()